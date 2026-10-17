"""领域路由：把任务分派到最相关的工具子集。

三级策略（由 config.ROUTER_MODE / 环境变量 OMNIPILOT_ROUTER 控制）：
- heuristic：只用关键词启发式（零算力、确定性、快）。
- decision ：只用决策层（Jev 式 Choice 六域，本项目的 omnipilot16 决策模型）。
- hybrid（默认）：关键词命中就快返回；未命中再交给决策层；决策层不可信才回退到大模型分类。

为什么保留启发式作为快路：实测关键词 baseline 在本项目 demo 路由集上 acc≈0.97，
且零延迟零算力；决策层的真正增量在于关键词覆盖不到的表述、以及意图/情感/事实核验
等启发式根本无法表达的能力。这不是"能省则省"，而是明确的算力/准确率取舍。

决策层为什么可能返回 None：letter-logit 依赖模型按字母作答。实测思考型模型（qwen3）
首 token 是推理词、未经字母训练的原模型首 token 是候选原文，此时概率不可信，
必须回退而不是硬给一个域——静默给错域比慢一点危险得多。
"""
from __future__ import annotations

import re

from ..config import DOMAIN_DESC, DOMAINS, ROUTER_MODE
from ..decision.engine import DecisionEngine

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "browser": ("浏览器", "网页", "网站", "网址", "链接", "登录", "搜索", "百度", "谷歌", "淘宝",
                "京东", "知乎", "微博", "http", "www", "标签页", "表单",
                # 联网检索/读网页类诉求，也应路由到浏览器域（含 web_search / web_fetch）
                "检索", "联网", "搜一下", "查资料", "搜资料", "web",
                # 页面/标题类描述也应指向浏览器，避免只命中 system 而缺失浏览器工具
                "页面", "标题", "首页", "跳转"),
    "office": ("word", "excel", "ppt", "powerpoint", "表格", "单元格", "文档", "幻灯片", "演示",
               "报表", "wps", "工作表", "写一份", "生成一份", "excel表"),
    "code": ("代码", "程序", "脚本", "运行", "python", "bug", "编译", "调试", "测试", "git",
             "仓库", "报错", "重构", "单元测试", "构建", "命令行", "终端",
             # 文件落盘类诉求：由 code_python / code_run 完成最稳
             "写入", "写到", "写进", "保存到", "存到", "记录到", "写入文件", "文本文件", "txt"),
    "media": ("视频", "图片", "图像", "剪辑", "转码", "截图", "裁剪", "缩放", "音频", "抽帧",
              "ffmpeg", "裁剪视频", "分辨率", "格式转换"),
    "blender": ("3d", "建模", "blender", "场景", "网格", "材质", "渲染", "模型文件", "blend"),
    "system": ("打开", "关闭", "窗口", "桌面", "文件夹", "文件", "点击", "输入", "快捷键",
               "任务栏", "应用", "电脑", "系统设置", "剪贴板", "鼠标", "键盘"),
}

# 形如 example.com / https://a.b.cn 的网址，出现即视为浏览器任务
_URL_RE = re.compile(
    r"(?:https?://)?\b[\w-]+(?:\.[\w-]+)*\.(?:com|cn|org|net|io|edu|gov|co|dev|ai)\b", re.IGNORECASE
)

# 这些词一旦出现，说明是"操作浏览器页面"，即使用户没说"浏览器"
_PAGE_WORDS = ("页面", "标题", "首页", "跳转", "链接", "标签页")


def classify_heuristic(task: str) -> list[str]:
    """返回命中的领域列表（可能多个）。"""
    low = task.lower()
    hits = [d for d, kws in _KEYWORDS.items() if any(k in low for k in kws)]
    # 网址或页面类词汇 → 一定是浏览器任务（且补上 system 以提供截图/键鼠兜底）
    if _URL_RE.search(task) or any(w in task for w in _PAGE_WORDS):
        for d in ("browser", "system"):
            if d not in hits:
                hits.append(d)
    return hits


def _model_classify(task: str, client, planner_model: str | None) -> list[str]:
    """大模型兜底分类（原逻辑保持不变）。"""
    if client is not None and planner_model:
        try:
            prompt = (
                "把任务归类到以下领域之一(可多选，逗号分隔)："
                "browser,office,code,media,blender,system。\n"
                f"任务：{task}\n只输出领域名。"
            )
            out = client.chat(planner_model, [{"role": "user", "content": prompt}],
                              options={"temperature": 0, "num_predict": 32, "num_ctx": 2048}, think=False)
            picked = [d for d in DOMAINS if d in out]
            if picked:
                return picked
        except Exception:
            pass
    return []


def _decision_model() -> str:
    from ..config import DECISION_MODEL

    return DECISION_MODEL


# 决策可信度下限：低于它宁可回退，也不把一个"其实很犹豫"的域当真。
_DECISION_MIN_CONF = 0.35


def classify_with_decision(task: str, engine: DecisionEngine | None = None) -> list[str] | None:
    """用决策层（Choice 六域）判定领域。不可信时返回 None（由调用方回退）。

    判据是 `letter_ok`：模型必须真的用字母作答，概率才与候选一一对应。
    """
    try:
        eng = engine or DecisionEngine(model=_decision_model())
        sc = eng.score_candidates(
            task, "这条用户请求应该交给哪个工具域处理？",
            list(DOMAINS), dict(DOMAIN_DESC))
    except Exception:
        return None
    if not sc.letter_ok or sc.probs[sc.best_index()] < _DECISION_MIN_CONF:
        return None
    return [DOMAINS[sc.best_index()]]


def classify(task: str, client=None, planner_model: str | None = None, *,
             engine: DecisionEngine | None = None, mode: str | None = None) -> list[str]:
    """领域分类（三级策略，见模块 docstring）。"""
    m = (mode or ROUTER_MODE)
    hits = classify_heuristic(task)

    if m == "heuristic":
        return hits or _model_classify(task, client, planner_model) or ["system"]

    if m == "decision":
        return (classify_with_decision(task, engine)
                or hits
                or _model_classify(task, client, planner_model)
                or ["system"])

    # hybrid（默认）：启发式命中即快返回；未命中才走决策层，最后才用大模型兜底。
    if hits:
        return hits
    return (classify_with_decision(task, engine)
            or _model_classify(task, client, planner_model)
            or ["system"])


def tools_for(domains: list[str]) -> tuple[str, ...] | None:
    """领域 -> 传给注册表的领域过滤。None 表示不限制。"""
    valid = tuple(d for d in domains if d in DOMAINS)
    return valid or None

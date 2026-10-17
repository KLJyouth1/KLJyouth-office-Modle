"""智能体主循环：感知 → 决策 → 执行 → 验证。

每一步：读取环境观察(浏览器DOM / 桌面UI树) → 让规划模型输出一个 JSON 动作
→ 执行工具 → 把结果写入历史 → 再观察。直到模型给出 final 或达到步数上限。

低运行环境适配：默认关闭思维链(think=False)、限制观察与历史长度、上下文可控。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from ..config import MAX_STEPS, PLANNER_MODEL, VISION_MODEL
from ..prompts import SYSTEM_PROMPT, _DECIDE_PROMPT, _PLAN_PROMPT, _PLAN_SYSTEM
from ..perception import describe_for_model, dump_elements
from ..runtime.llm import OllamaClient, OllamaError
from ..tools import ToolRegistry, build_registry
from ..personal import PersonalProfile, infer_preferences_from_text
from .memory import Episode, Memory, Step
from .router import classify, tools_for

# 提示词常量（SYSTEM_PROMPT / _DECIDE_PROMPT / _PLAN_*）已上移到 ..prompts，
# 以免只想取提示词的调用方被迫导入 perception/tools 的 GUI 依赖。
# 这里仍从 ..prompts 导入，保持本模块对外接口不变。


def _repair_json(s: str) -> str:
    """修常见的 JSON 手写毛病：尾逗号、单引号、Python 字面量。

    低配小模型偶尔会输出不完全合法的 JSON，直接判失败会让整条轨迹空转；
    先做无损语义修复再解析，能显著降低"决策解析失败"率。
    """
    s = re.sub(r",\s*([}\]])", r"\1", s)        # 尾随逗号
    s = s.replace("'", '"')                      # 单引号键值
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    return s


_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _extract_action(text: str) -> dict[str, Any] | None:
    """从模型输出里取出"下一步动作"。

    P2-A 新协议（标准 Qwen / Ollama 工具调用）：
        <tool_call>{"name": "工具名", "arguments": {…}}</tool_call>
        → 归一为内部动作 {"tool": …, "args": {…}}
    旧协议仍兼容：直接输出 {"tool":…,"args":…} / {"final":…} 的 JSON。
    既没有工具调用块、也不是动作 JSON 时，视为**最终答复**（{"final": 原文}）——
    这正是 Qwen 协议里"任务完成就直接说话"的语义。
    """
    if not text or not text.strip():
        return None
    m = _TOOL_CALL_BLOCK_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
        except Exception:
            obj = None
        if isinstance(obj, dict):
            name = obj.get("name") or obj.get("tool")
            args = obj.get("arguments") if "arguments" in obj else obj.get("args")
            if name:
                return {"tool": str(name), "args": args if isinstance(args, dict) else {}}
    legacy = _extract_json(text)
    if isinstance(legacy, dict) and ("tool" in legacy or "final" in legacy):
        return legacy
    return {"final": text.strip()}


def _action_from_message(msg: dict) -> dict[str, Any] | None:
    """从 Ollama 的 assistant message 取动作：**优先原生 tool_calls**，其次解析文本块。

    O1：以前客户端从不发 tools，所以永远只有文本可解析；现在两路都接。
    """
    for tc in (msg.get("tool_calls") or []):
        fn = (tc or {}).get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        raw = fn.get("arguments")
        if isinstance(raw, str):
            try:
                args = json.loads(raw) if raw.strip() else {}
            except Exception:
                args = {}
        else:
            args = raw if isinstance(raw, dict) else {}
        return {"tool": str(name), "args": args}
    return _extract_action(msg.get("content") or "")


def _extract_json(text: str) -> dict[str, Any] | None:
    """从模型输出里稳健地抠出第一个 JSON 对象（带字符串感知的括号配对）。"""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start : i + 1]
                for cand in (chunk, _repair_json(chunk)):
                    try:
                        obj = json.loads(cand)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        continue
                return None
    return None


# 表示"没做完/做不了"的措辞：不能当成成功终止
_NOT_DONE_MARKERS = (
    "尚未完成", "未完成", "还没有完成", "没有完成", "无法完成", "不能完成",
    "无法继续", "无法执行", "not completed", "cannot complete", "unable to complete",
)


def _is_unfinished(final_text: str) -> bool:
    t = (final_text or "").strip().lower()
    return any(m in t for m in _NOT_DONE_MARKERS)


# 声称"已完成"的措辞。若它出现在 final 中、而上一个工具步骤其实失败，
# 就是"谎报成功"——实测小模型会这样做（Office 写入失败却回"已写入"），必须拦下。
# 注意：不能把裸的"成功"当作成功断言——实测"任务未能成功执行"是诚实报告失败，
#       却因含"成功"二字被误判（本机复现过的假阳性），故只保留复合肯定式。
_SUCCESS_RE = re.compile(
    r"(已完成|搞定|写好了|做好了|处理好了"
    r"|已[把将].{0,16}(写入|保存|生成|创建|读取|打开|完成|删除|修改|运行|执行|转换|缩放|裁剪)"
    r"|已(写入|保存|生成|创建|读取|打开|删除|修改|运行|执行|转换|缩放|裁剪)"
    r"|(写入|保存|生成|创建|读取|转换|渲染|执行)成功"
    r"|成功(地)?(完成|写入|保存|生成|创建|读取|转换|渲染|执行))",
    re.IGNORECASE,
)
# 明确提到"失败/做不到/未成"的答复是诚实的，不算谎报成功。
_FAIL_MARKERS = ("失败", "不存在", "无法", "找不到", "没有找到", "出错", "错误", "未完成", "尚未",
                 "未能", "未成功", "不成功", "没有成功", "不能",
                 "error", "failed", "cannot", "not found", "unable")


def _claims_success(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    low = t.lower()
    if any(f in low for f in _FAIL_MARKERS):
        return False
    return bool(_SUCCESS_RE.search(t))


# 历史里这些是"元步骤"，不是真实工具调用；判断"上一步"时要跳过。
_META_STEPS = ("(none)", "(guard)", "(unfinished)", "(false-success)", "(fabrication)", "(repeat)")

# 数值断言检测：3 位以上的数字（如 827115、1920x1080 中的 1920）
_NUM_RE = re.compile(r"\d{3,}")
# 度量/计数类数字（"467 字""3 条""12 步"）：多为模型对工具输出的汇总，不按编造处理
_METRIC_NUM_RE = re.compile(r"\d{3,}\s*(?:字|条|行|个|步|秒|次|字节|字符|项|页)")


def _looks_like_fabrication(final_text: str, history: list,
                            task: str = "", plan: str = "") -> bool:
    """检测"未经工具验证的数值断言"。

    实测缺陷（用户直接对话时发现）：模型被问 12345×67 时凭记忆编出 99240
    （正确为 827115）。这是幻觉。

    但它必须**只针对"事实性数值"**：实测又踩到误伤——模型回"已写入 467 字"，
    467 是 172+203+92 的合计（工具只报过那几个），没有任何工具直接产出过 467，
    于是被判成编造。度量/计数类数字（N 字/条/行/个/步/秒）常是模型对工具输出的
    汇总，不该按编造处理，这里先剔除；证据面同时纳入任务描述与已定计划
    （模型合法地会引用它们）。
    """
    text = _METRIC_NUM_RE.sub(" ", final_text or "")
    nums = set(_NUM_RE.findall(text))
    if not nums:
        return False
    evidence = " ".join(
        [(s.output or "") for s in history if getattr(s, "ok", False)]
        + [task or "", plan or ""]
        + [str(s.args) for s in history]
    )
    return not any(n in evidence for n in nums)


# 声称"已写/已存/已生成文件"的措辞。
# 实测缺陷：模型只搜索了一次，就回"results are stored in E:/temp/yolo11_notes.txt"，
# 而它从没调用过任何写入工具——典型的"虚报副作用"。必须拦下。
_WRITE_CLAIM_RE = re.compile(
    r"(已写入|写入文件|已保存|保存到|已存到|已创建|已生成文件|写好了|存好了"
    r"|stored in|written to|saved to|has been saved|file created)",
    re.IGNORECASE,
)
# 能真正产生"落盘/产出物"的工具。
# 注意：**不含 file_read**。旧版本把 file_read 列在这里，于是"读过一个文件"也算
# "已交付"，交付物闸门形同虚设（假成功成因之一）。
_WRITE_TOOLS = frozenset((
    "file_write", "code_python", "code_run", "excel_write", "word_create", "ppt_create",
    "image_resize", "image_crop", "video_convert", "video_trim",
    "video_extract_audio", "video_extract_frames", "blender_run_script", "blender_render",
))


def _claims_write_without_evidence(final_text: str, history: list) -> bool:
    """声称写了文件，但没有任何写入类工具成功执行过 → 虚报副作用。"""
    if not _WRITE_CLAIM_RE.search(final_text or ""):
        return False
    return not any(getattr(s, "ok", False) and s.tool in _WRITE_TOOLS for s in history)


# "零执行直接瞎答"：一个真实工具都没调用，却断言具体位置/坐标/界面状态。
# 实测缺陷：任务"操控我的电脑打开微信"→ 0 步就回"微信位于第32行，第14列…"（纯编造）。
_ZERO_STEP_CLAIM_RE = re.compile(
    r"(坐标|位于|第\s*\d+\s*(?:行|列|个)|界面上|页面上|图标|已打开|已点击|已输入|屏幕|工具栏|任务栏)"
)


def _claims_unobserved_state(final_text: str, history: list) -> bool:
    """没有任何真实工具步骤，却断言了必须"看过/操作过"才能知道的状态。"""
    has_real_step = any(s.tool not in _META_STEPS for s in history)
    if has_real_step:
        return False
    return bool(_ZERO_STEP_CLAIM_RE.search(final_text or ""))


# 任务本身要求"产出文件"（写入/保存到 + 一个像路径的目标）。
# 实测缺陷：模型搜完、总结完就 final 收尾，交不出文件；必须把它推到最后一步。
_WRITE_TASK_RE = re.compile(
    r"(写入|写到|写进|保存到|保存为|存到|生成文件|导出到)[^\n]{0,60}?[\w:/\\\\.-]+\.(txt|md|csv|json|xlsx|docx|pptx|py|html|pdf|png|jpg)",
    re.IGNORECASE,
)


# 写入类动词（与顺序无关的判定用）与"带扩展名的文件名"。
_WRITE_VERB_RE = re.compile(
    r"(写入|写到|写进|写下来|保存到|保存为|存到|存为|生成|导出|输出|落盘|追加|记录下来)",
)
_FILE_EXT_RE = re.compile(
    r"\.(txt|md|csv|json|jsonl|xlsx|xls|docx|pptx|py|html|pdf|png|jpg|jpeg|webp|mp4|mp3|wav|glb|obj)\b",
    re.IGNORECASE,
)


def _task_needs_write(task: str) -> bool:
    """任务是否要求产出文件。

    为什么需要第二套判据（实测缺陷）：旧实现只用 `_WRITE_TASK_RE`，它要求**动词在前、
    路径紧随其后**（`写入 D:\\a\\out.txt`）。可真实任务常写成
    「在 D:\\a\\points.txt **写入**三条要点」——路径在前、动词在后，旧正则判 False，
    于是**所有交付物闸门被整体跳过**：模型零步、零工具调用直接 final，agent 还报
    `成功: True`（实测复现）。故补一条**顺序无关**的判据：任务里同时出现写入类动词与
    带扩展名的文件名，即视为要产出文件。宁可多催一次「请把结果落盘」，也不能放过假成功。
    """
    t = task or ""
    if _WRITE_TASK_RE.search(t):
        return True
    return bool(_WRITE_VERB_RE.search(t) and _FILE_EXT_RE.search(t))


# 任务里的数量词：中文数字/阿拉伯数字 + 单位（条/个/点/项/行/步）。
_NUM_TOKENS = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4,
               "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_ITEM_COUNT_RE = re.compile(r"(\d+|[一两二三四五六七八九十])\s*(?:条|个|点|项|行|步)")
# 任务里显式给出的目标路径（Windows 盘符 或 POSIX 绝对路径）。
_TASK_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|/)[^\s，,。；;：:）)】\]\"'`]+")


def _required_item_count(task: str) -> int:
    """从任务里解析"要几条/几项/几行"。

    取所有命中的**最大值**：任务常写「三条要点，每条一行」——顺序扫描会先撞上"三"，
    但取最大值对两种顺序都稳。解析不到返回 0（表示"无规格可校验"，而不是"要 0 条"）。
    上限钳到 20：更大的数字多半不是计数要求，宁可不校验也不要造出假约束。
    """
    counts = []
    for m in _ITEM_COUNT_RE.finditer(task or ""):
        tok = m.group(1)
        n = int(tok) if tok.isdigit() else _NUM_TOKENS.get(tok, 0)
        if 1 <= n <= 20:
            counts.append(n)
    return max(counts) if counts else 0


def _norm_path(p: str) -> str:
    """路径比较用归一化：统一分隔符、去尾部标点、忽略大小写。"""
    return p.strip().rstrip("。；;,，").replace("/", "\\").lower()


def _task_target_path(task: str) -> str | None:
    """任务里**唯一**明确给出的目标路径；给出多个（或一个都没有）时返回 None。

    只在"唯一"时才校验：任务里出现多个路径时无法判定哪个是交付目标，此时不做断言，
    避免造出假约束把模型逼进死循环。
    """
    uniq: dict[str, str] = {}
    for p in _TASK_PATH_RE.findall(task or ""):
        uniq.setdefault(_norm_path(p), p)
    return next(iter(uniq.values())) if len(uniq) == 1 else None


def _should_realign_path(got: str, target: str | None) -> bool:
    """写入路径是否该改回任务给定路径。

    判据（见 [`Agent._align_write_path`]）：任务唯一给了目标路径、两者归一化后不同、
    且**文件名一致**。文件名不同就不是同一个交付物，绝不代改——那会变成"猜路径"。
    抽成纯函数是为了能脱离 Agent 实例单测。
    """
    if not target or not got:
        return False
    if _norm_path(got) == _norm_path(target):
        return False
    return Path(got).name.lower() == Path(target).name.lower()


# 交付物规格核验的补写预算（有界，防死循环）。
#
# 为什么是 5 而不是 2（实测）：v2 模型学会的是「一条要点一次 file_write(append)」的轨迹
# （见 synth 链 D/E），写 3 条要点就需要 3 步。旧的预算 2 会在写完第 2 条时放弃并判失败，
# 把一个本可完成的 3 条任务错杀。预算必须 ≥ 任务要求的条数，故取 5（覆盖到 5 条要点），
# 同时仍受 max_steps 总量约束。
_SPEC_RETRY_BUDGET = 5


def _deliverable_spec_gap(task: str, history: list) -> str | None:
    """核对**产物本身**是否满足任务规格：目标路径一致 + 条目数达标。

    为什么必须读回产物：旧的交付物闸门只问"有没有写过文件"。实测（1.5B）因此出现
    假成功——模型只写 1 条要点（要求 3 条）、还把路径写成 `…\\yolo-ultralytics_\\points.txt`
    （真项目目录的同级兄弟），而 agent 照样报 `成功: True`。这里改成按任务规格验收真实
    文件内容，把"假成功"变成"真成功"或"如实失败"。

    返回 None 表示通过（或无法判定）；否则返回给模型的纠正说明。
    """
    writes = [s for s in history if getattr(s, "ok", False) and s.tool in _WRITE_TOOLS]
    if not writes:
        return None                      # 完全没有写入：由既有交付物闸门负责催促
    got = str((writes[-1].args or {}).get("path") or "").strip()
    if not got:
        return None

    want = _task_target_path(task)
    if want and _norm_path(got) != _norm_path(want):
        return (f"你把结果写到了「{got}」，但任务指定的目标是「{want}」。"
                f"请下一步用任务给定的那个路径重新写入；路径请逐字复制，"
                f"不要改写、不要省略中间目录。")

    need = _required_item_count(task)
    if not need:
        return None
    p = Path(got)
    if not p.is_file():
        return None
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) >= need:
        return None
    return (f"你只写入了 {len(lines)} 条，任务要求 {need} 条。"
            f"请继续用 file_write 以 append=true 追加剩余条目，"
            f"凑满 {need} 条且各条内容互不相同，再收尾。")


# 瞬时性故障（超时/网络）值得自动重试；系统性错误（参数错、工具不存在）则不该重试。
_TRANSIENT_MARKERS = (
    "timeout", "timed out", "超时", "temporarily", "connection", "econnreset",
    "wsarecv", "connection reset", "network", "502", "503", "504",
)


def _is_transient(msg: str) -> bool:
    m = (msg or "").lower()
    return any(k in m for k in _TRANSIENT_MARKERS)


def _match_tool(name: str, valid: list[str]) -> str | None:
    """把模型写歪的工具名对齐到真实工具名。

    实测小模型会丢掉域前缀（如把 browser_click_text 写成 click_text）。
    这类"近义词"完全可以自动纠正，不必浪费一整步。
    """
    if not name:
        return None
    if name in valid:
        return name
    n = name.strip().lower().replace("-", "_").replace(" ", "_")
    norm = {v.lower().replace("-", "_"): v for v in valid}
    if n in norm:
        return norm[n]
    suffix = [v for v in valid if v.lower().endswith("_" + n)]   # 丢前缀
    if len(suffix) == 1:
        return suffix[0]
    subs = [v for v in valid if n in v.lower()]                   # 唯一子串
    if len(subs) == 1:
        return subs[0]
    pref = [v for v in valid if v.lower().startswith(n)]          # 唯一前缀
    if len(pref) == 1:
        return pref[0]
    return None


class Agent:
    def __init__(
        self,
        client: OllamaClient | None = None,
        *,
        planner_model: str = PLANNER_MODEL,
        vision_model: str = VISION_MODEL,
        domains: tuple[str, ...] | None = None,
        max_steps: int = MAX_STEPS,
        headless: bool = True,
        verbose: bool = True,
        memory: Memory | None = None,
        think: bool = False,
        use_plan: bool = True,
        collector=None,
        personal: bool = True,
    ) -> None:
        self.client = client or OllamaClient()
        self.planner_model = planner_model
        self.vision_model = vision_model
        self.max_steps = max_steps
        self.headless = headless
        self.verbose = verbose
        self.memory = memory or Memory()
        self.think = think
        self.use_plan = use_plan
        self.collector = collector  # 可选：把成功的"观察->动作"自动沉淀为训练样本
        # Personal Agent：长期画像，使智能体主动适配用户（偏好/事实/技能）
        self.personal = PersonalProfile.load() if personal else None
        self.registry: ToolRegistry = build_registry(headless=headless, domains=domains)
        self.domains: list[str] = list(domains) if domains else []
        self._browser_started = False
        self.last_decision: dict | None = None

    # ------------------------------------------------------------------ 输出
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # ------------------------------------------------------------------ 感知
    def _observe(self, has_history: bool = False) -> str:
        """构造给模型的"环境观察"。

        关键：观察的格式必须与训练数据一致，否则微调过的小模型无法识别状态。
        - 浏览器任务：紧凑的 URL/标题（与训练样本同格式）；需要坐标时模型自行调 browser_snapshot。
        - 桌面类任务：给 UI 元素树（含坐标）。
        - 纯工具类任务（代码/媒体）：首步提示"可直接调用工具"，已有执行结果时提示"参考上述执行结果"，
          与训练样本的措辞逐字对齐，避免分布外导致模型误判状态。
        """
        parts: list[str] = []
        if "browser" in self.domains:
            if self._browser_started:
                tool = self.registry.get("browser_url")
                if tool:
                    res = tool.run()
                    if res.ok:
                        parts.append("【浏览器状态】\n" + res.output)
            else:
                parts.append("（本任务无需屏幕观察，可直接调用工具）")
        elif any(d in self.domains for d in ("system", "office", "blender")):
            try:
                els = dump_elements(max_depth=3, max_windows=3, limit=120)
                parts.append("【桌面UI元素(含坐标)】\n" + describe_for_model(els, limit=35))
            except Exception:
                pass
        if not parts:
            parts.append("（参考上述执行结果）" if has_history else "（本任务无需屏幕观察，可直接调用工具）")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ 决策
    def _plan(self, task: str) -> str:
        if not self.use_plan:
            return "（未显式规划）"
        try:
            raw = self.client.chat(
                self.planner_model,
                [{"role": "user", "content": _PLAN_PROMPT.format(
                    task=task,
                    # 必须给工具清单：不给的话原生基座会把文件任务规划成"打开记事本"这类
                    # 工具清单里不存在的界面操作，后续 decide 照着错计划走 → 必然失败（实测）。
                    tools=self.registry.describe(self.domains or None),
                )}],
                system=_PLAN_SYSTEM,
                options={"temperature": 0.2, "num_predict": 256, "num_ctx": 4096},
                think=False,
            )
        except OllamaError:
            return "（规划失败，直接执行）"
        # 清洗：只保留形如 "1. xxx" / "- xxx" 的步骤行，最多 6 条
        steps: list[str] = []
        for line in raw.splitlines():
            s = line.strip()
            if not s:
                continue
            if s[0].isdigit() or s.startswith(("-", "•", "*")):
                steps.append(s[:100])
            if len(steps) >= 6:
                break
        if not steps:
            # 兜底：规划失败时给一个通用两步计划，避免把乱码塞进决策提示
            return "1. 完成用户任务\n2. 汇报结果"
        return "\n".join(steps)

    def _tools_schema(self) -> list[dict] | None:
        """按当前领域挑工具，包成 OpenAI function 声明，供 Ollama 走原生工具调用。

        只发当前领域的工具：45 个工具全发会撑爆 8K 上下文（这也是工作台那边
        做"首轮按需披露"的原因）。
        """
        try:
            tools = list(self.registry.all())
        except Exception:
            return None
        doms = set(self.domains or [])
        picked = [t for t in tools if not doms or getattr(t, "domain", None) in doms] or tools
        out = []
        for t in picked:
            params = getattr(t, "parameters", None) or {"type": "object", "properties": {}}
            out.append({"type": "function", "function": {
                "name": getattr(t, "name", ""),
                "description": getattr(t, "description", "") or "",
                "parameters": params,
            }})
        return out or None

    def _decide(self, task: str, plan: str, observation: str, history: list[Step]) -> dict | None:
        # 历史里必须让模型"看得见上一步的结果"，否则它会重复同一个动作、永远推进不了。
        # 实测缺陷：原来每步输出只截断到 160 字符，web_search 的结果被腰斩，
        # 模型以为没搜到 → 反复重搜 → 被防重复闸门拦死（0.5B / 1.5B 都中招）。
        # 现在：最近一步给足（1600 字符），更早的步子保留摘要，兼顾上下文长度。
        hist_lines = []
        recent = history[-5:]
        for i, s in enumerate(recent):
            args = json.dumps(s.args, ensure_ascii=False)[:120]
            budget = 1600 if i >= len(recent) - 2 else 400   # 最近两步给足，更早只留摘要
            out = (s.output or "")[:budget]
            hist_lines.append(f"#{s.n} {s.tool}({args}) -> {'成功' if s.ok else '失败'}: {out}")
        prompt = _DECIDE_PROMPT.format(
            task=task, plan=plan, tools=self.registry.describe(self.domains or None),
            observation=observation, history="\n".join(hist_lines) or "（无）",
        )
        # O1：把工具 schema 真正发给 Ollama，拿原生 tool_calls；拿不到再回退解析文本块。
        msg = self.client.chat_message(
            self.planner_model,
            [{"role": "user", "content": prompt}],
            system=SYSTEM_PROMPT,
            tools=self._tools_schema(),
            options={"temperature": 0.2, "num_predict": 1024, "num_ctx": 8192},
            think=self.think,
        )
        action = _action_from_message(msg)
        self.last_decision = {"prompt": prompt, "action": action}
        return action

    # ------------------------------------------------------------------ 执行
    def _align_write_path(self, action: dict) -> None:
        """把"同名不同目录"的写入路径改回任务唯一给定的目标路径。

        为什么需要（实测）：1.5B 无法逐字复制长绝对路径——任务给
        `…\\yolo-ultralytics\\_t\\points.txt`，它连续两轮写成 `…\\yolo-ultralytics_\\t\\points.txt`，
        连"路径请逐字复制"的纠正提示都照做不了。这是小模型的能力上限，靠提示修不好，
        只能由宿主在落盘前纠正（数据侧另加监督样本，见 train/synth.py）。

        判据刻意收紧到两条**同时**成立才纠正，避免退化成"猜路径"：
          1. 任务里**唯一**给出一个目标路径（有多个候选时不判）；
          2. 模型给的路径与目标**文件名相同**（basename 一致）——它显然在指同一个交付物。
        纠正一律打日志、绝不静默改；改完仍要过交付物规格闸门，不达标照样如实失败。
        """
        target = getattr(self, "_task_target", None)
        if not target:
            return
        if str(action.get("tool", "")) not in _WRITE_TOOLS:
            return
        args = action.get("args") or {}
        got = str(args.get("path") or "").strip()
        if not _should_realign_path(got, target):
            return
        args["path"] = target
        action["args"] = args
        self._log(f"[路径纠正] 写入路径「{got}」与任务目标同名不同目录，"
                  f"已按任务给定路径改为「{target}」")

    def _execute(self, action: dict) -> tuple[Step, bool]:
        raw_name = str(action.get("tool", "")).strip()
        args = action.get("args") or {}
        # 实测缺陷：模型偶尔输出 {"query": …} 这种**没有 tool 字段**的动作，
        # 旧提示只说"未知工具「」"，模型看不懂；这里给出明确、可纠正的错误。
        if not raw_name:
            return Step(-1, "", args, False,
                        "你的动作里缺少 tool 字段。必须严格输出 "
                        "{\"thought\":\"简短理由\",\"tool\":\"工具名\",\"args\":{…}}，"
                        "其中 tool 是可用清单里的工具名。"), False
        valid_names = [t.name for t in self.registry.all()]
        # 工具名自动纠错：小模型常丢域前缀（click_text -> browser_click_text）
        repaired = _match_tool(raw_name, valid_names)
        name = repaired or raw_name
        tool = self.registry.get(name)
        if tool is None:
            return Step(-1, raw_name, args, False,
                        f"未知工具「{raw_name}」：不在可用清单中，请只使用清单内的工具名"), False
        if name.startswith("browser_") and name != "browser_snapshot":
            self._browser_started = True

        # 参数校验：只把该工具 schema 声明过的参数传进去，丢弃模型编造的参数。
        # 否则一个不存在的参数名(如 snapshot 的 monitor)会把整次调用变成 TypeError，
        # 而小模型往往会反复重试同一个错误调用，导致空转。
        valid = set(((tool.parameters or {}).get("properties") or {}).keys())
        clean = {k: v for k, v in args.items() if k in valid}
        dropped = [k for k in args if k not in valid]

        t0 = time.time()
        res = tool.run(**clean)
        # 瞬时性故障（页面超时、网络抖动）自动重试一次——这类错误重试即可成功，
        # 不该浪费一步让模型去"换招"；系统性错误（参数错）则不重试。
        if not res.ok and _is_transient(res.output):
            time.sleep(1.0)
            res = tool.run(**clean)
        out = res.output
        notes = []
        if repaired and repaired != raw_name:
            notes.append(f"（已自动纠正工具名 {raw_name} → {repaired}）")
        if dropped:
            notes.append(f"（已忽略无效参数 {dropped}；本工具可用参数：{sorted(valid) if valid else '无'}）")
        if notes:
            out = f"{out}\n{' '.join(notes)}" if out else " ".join(notes)
        step = Step(0, name, clean, res.ok, out, round(time.time() - t0, 2))
        return step, True

    # ------------------------------------------------------------------ 主循环
    def run(self, task: str, domains: list[str] | None = None) -> dict:
        self.domains = domains or classify(task, self.client, self.planner_model)
        self._log(f"[路由] 任务领域: {self.domains}")

        # 记忆召回：把相似历史经验作为参考。注意只放进"计划"字段，
        # 不放进"观察"字段——观察是模型判断状态的关键，必须与训练措辞逐字一致。
        recalled = self.memory.recall(task, n=2)
        plan = self._plan(task)
        if recalled:
            plan += "\n（历史参考：" + "；".join(
                f"{e.get('task')}" for e in recalled
            ) + "）"
        # Personal Agent：把长期画像注入"计划"字段（不污染观察字段，保持训练/推理一致）
        if self.personal is not None:
            # 先从任务文本里抽取显式偏好信号（零算力冷启动规则）
            for key, val, w in infer_preferences_from_text(task):
                self.personal.observe("preference", key, val, signal=w)
            prof = self.personal.summary()
            if prof:
                # 加"仅供参考、不代表当前任务状态"这句是**必要**的：实测小模型会把画像里的
                # 历史失败记录当成"我现在正在失败"，于是规划跑偏（去"打开记事本"而不是调
                # file_write）。画像只是背景，必须与当前任务状态显式区分，并钳制长度。
                plan += ("\n（背景画像（仅供措辞参考，不代表当前任务状态）：\n"
                         + prof[:200] + "\n）")
        self._log(f"[计划]\n{plan}")

        # 任务唯一给定的目标路径：供写入路径纠正使用（见 _align_write_path）。
        self._task_target = _task_target_path(task)

        history: list[Step] = []
        result = ""
        ok = False
        fail_counts: dict[str, int] = {}   # 同一动作重复计数：失败防死循环，成功防"原地打转"
        zero_step_nudged = False           # 零执行瞎答只提醒一次，避免死循环
        deliverable_nudged = 0               # 交付物未落盘的可提醒次数（有界，防死循环）
        spec_checks = 0                    # 交付物规格核验的补写次数（有界，防死循环）

        for n in range(1, self.max_steps + 1):
            observation = self._observe(has_history=bool(history))
            action = self._decide(task, plan, observation, history)

            if not action:
                self._log(f"[{n}] 决策解析失败，重试…")
                history.append(Step(n, "(none)", {}, False, "模型未输出合法 JSON"))
                continue

            thought = str(action.get("thought", ""))[:200]
            if "final" in action:
                final_text = str(action["final"])
                # 拒绝"未完成类"终止：小模型常以"任务尚未完成"当作 final 提前收尾，
                # 这会伪装成成功。此时应继续推进，而不是结束。
                if _is_unfinished(final_text):
                    self._log(f"[{n}] 模型声称未完成，继续推进：{final_text[:80]}")
                    history.append(Step(n, "(unfinished)", {}, False,
                                        f"你上一步以未完成结束，请继续实际执行而非声明失败。原始措辞：{final_text[:120]}"))
                    continue
                # 谎报成功拦截：上一个真实工具步骤失败，却声称已完成 → 不放过
                last_tool = next((s for s in reversed(history) if s.tool not in _META_STEPS), None)
                if last_tool is not None and not last_tool.ok and _claims_success(final_text):
                    self._log(f"[{n}] 上一步失败却声称成功，拦下：{final_text[:80]}")
                    history.append(Step(
                        n, "(false-success)", {}, False,
                        f"你上一步工具「{last_tool.tool}」实际失败了（{(last_tool.output or '')[:100]}），"
                        f"但 final 声称成功。请先修复这一步（换参数/换工具），或如实说明失败，不要谎报成功。",
                    ))
                    continue
                # 编造拦截：final 里的数值没有任何工具产出过 → 疑似凭记忆瞎编
                if _looks_like_fabrication(final_text, history, task, plan):
                    self._log(f"[{n}] 数值未经工具验证，拦下：{final_text[:80]}")
                    history.append(Step(
                        n, "(fabrication)", {}, False,
                        f"你给出的数值「{final_text[:60]}」不是任何工具的执行结果，可能是编造的。"
                        f"请先调用工具实际获取数据，再据实回答；不要凭记忆报数。",
                    ))
                    continue
                # 虚报副作用拦截：声称写了文件，却没有任何写入类工具执行过
                if _claims_write_without_evidence(final_text, history):
                    self._log(f"[{n}] 声称已写文件但无写入动作，拦下：{final_text[:80]}")
                    history.append(Step(
                        n, "(fabrication)", {}, False,
                        "你声称已经把结果写入/保存到文件，但历史上没有任何写入类工具"
                        "（例如 code_python）成功执行过。请先真正调用写入工具把文件落盘，再回复；不要虚报已完成。",
                    ))
                    continue
                # 零执行瞎答拦截：一个真实工具都没调用，却断言了位置/坐标/界面状态
                if not zero_step_nudged and _claims_unobserved_state(final_text, history):
                    zero_step_nudged = True
                    self._log(f"[{n}] 零执行却断言具体状态，拦下：{final_text[:80]}")
                    history.append(Step(
                        n, "(fabrication)", {}, False,
                        "你还没有执行任何动作（既没观察屏幕，也没调用任何工具），却给出了具体的位置、"
                        "坐标或界面状态，这只能是编造。请先调用工具真实观察（例如 sys_screenshot 或相关工具），"
                        "或者如实说明你当前无法确定。",
                    ))
                    continue
                # 交付物闸门：任务要求产出文件，但没有任何写入类工具成功执行 → 不许收尾。
                # 提醒次数必须**有界而持久**：旧实现是"提醒一次"的布尔开关，模型只要重复
                # 同一个 final 就能蒙混过关（实测：第 1 步拦下、第 2 步原样重复即报成功）。
                # 现在提醒两次仍不落盘就**如实判失败** —— 宁可诚实失败，也绝不假成功。
                if (_task_needs_write(task)
                        and not any(s.ok and s.tool in _WRITE_TOOLS for s in history)):
                    if deliverable_nudged < 2:
                        deliverable_nudged += 1
                        self._log(f"[{n}] 任务要求产出文件但尚无写入动作，拦下：{final_text[:80]}")
                        history.append(Step(
                            n, "(unfinished)", {}, False,
                            "任务要求把结果写入/保存到文件，但至今没有任何写入类工具"
                            "（例如 file_write）成功执行过。请不要就此收尾：请调用写入工具"
                            "把内容真正落盘，然后再汇报文件路径。",
                        ))
                        continue
                    self._log(f"[{n}] 屡次不落盘，如实判失败：{final_text[:80]}")
                    result = "未完成：任务要求把结果写入文件，但模型始终没有真正调用写入工具落盘。"
                    ok = False
                    break
                # 交付物规格核验：读回真实产物，核对目标路径与条目数（有界补写）。
                gap = _deliverable_spec_gap(task, history)
                if gap:
                    if spec_checks < _SPEC_RETRY_BUDGET:
                        spec_checks += 1
                        self._log(f"[{n}] 交付物未达规格，要求补写：{gap[:80]}")
                        history.append(Step(n, "(unfinished)", {}, False, gap))
                        continue
                    # 预算用尽仍不达标 → 如实判失败，绝不谎报成功
                    self._log(f"[{n}] 交付物仍不达规格，如实判失败：{gap[:80]}")
                    result = f"未完成：{gap}"
                    ok = False
                    break
                result = final_text
                ok = True
                self._log(f"[{n}] 完成：{result}")
                if self.collector is not None:
                    try:
                        self.collector.add_step(self.last_decision, None)
                    except Exception:
                        pass
                break

            # 防死循环 / 防"原地打转"：同一动作（工具+参数）重复执行没有意义。
            # 实测小模型会"搜到结果却继续重复搜索"，必须在第 2 次催它用结果推进、
            # 第 3 次直接拦下不再执行，强制换路。
            # 路径纠正必须排在"算重复键"和"执行"之前：纠正后的参数才是真正要执行、
            # 也要拿去去重的参数（否则会对着错路径判重复）。
            self._align_write_path(action)
            key = f"{str(action.get('tool', ''))}:{json.dumps(action.get('args') or {}, ensure_ascii=False, sort_keys=True)}"
            if fail_counts.get(key, 0) >= 2:
                self._log(f"[{n}] 重复动作，拦下：{key[:80]}")
                history.append(Step(
                    n, "(repeat)", {}, False,
                    f"动作 {action.get('tool')} 你已经执行过两次，结果就在上方历史里。"
                    f"不要再重复它；请改用其它工具、换参数，或直接汇总已有结果。"))
                continue

            step, known = self._execute(action)
            step.n = n
            history.append(step)
            if self.collector is not None:
                try:
                    self.collector.add_step(self.last_decision, step)
                except Exception:
                    pass
            self._log(f"[{n}] {thought}\n    -> {step.tool} {json.dumps(step.args, ensure_ascii=False)[:100]}"
                      f" | {'成功' if step.ok else '失败'} ({step.elapsed}s)\n    {step.output[:200]}")

            fail_counts[key] = fail_counts.get(key, 0) + 1
            cnt = fail_counts[key]
            if step.ok and cnt == 2:
                history.append(Step(
                    n, "(repeat)", {}, False,
                    f"提示：{step.tool} 刚刚已经成功执行过，结果就在上方历史里。"
                    f"请基于该结果推进下一步（例如用 code_python 写入文件，或直接汇总答复），不要重复同一动作。"))
            elif not step.ok and cnt >= 2:
                history.append(Step(
                    n, "(guard)", {}, False,
                    f"提示：动作 {step.tool} 已连续失败 {cnt} 次，请更换参数或改用其它工具。",
                ))

            if not known:
                continue
        else:
            result = f"达到最大步数({self.max_steps})仍未完成。最后状态：{history[-1].output[:300] if history else '无'}"

        episode = Episode(task=task, domains=self.domains, steps=history, result=result, ok=ok)
        self.memory.record(episode)
        # Personal Agent：把本次结果沉淀为长期记忆（技能/事实），并持久化画像
        if self.personal is not None:
            try:
                self.personal.learn_from_episode(task, result or "", ok)
                self.personal.forget()          # 主动遗忘：只留重要的
                self.personal.save()
            except Exception:
                pass
        # 执行轨迹（供界面展示"它到底做了什么"；纯附加字段，不影响既有调用方）
        trace = [{"n": s.n, "tool": s.tool, "args": s.args, "ok": s.ok,
                  "output": (s.output or "")[:1500], "elapsed": s.elapsed} for s in history]
        return {"result": result, "ok": ok, "steps": len(history), "domains": self.domains, "trace": trace}

    # ------------------------------------------------------------------ 收尾
    def close(self) -> None:
        """关闭浏览器等常驻资源。"""
        for t in self.registry.all():
            if t.__class__.__name__ == "FuncTool" and hasattr(t, "_func"):
                owner = getattr(t._func, "__self__", None)
                if owner is not None and owner.__class__.__name__ == "BrowserTools":
                    try:
                        owner.close()
                    except Exception:
                        pass

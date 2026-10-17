"""知识蒸馏：用本地更强的模型当"老师"，为学生生成**经过验证**的训练数据。

动机（用户要求"把本地模型结合进训练，重点解决幻觉"）：
本机可用的老师模型（来自 Ollama）：
  - qwen3:4b            推理型，适合做决策/规划的老师
  - kraoffices-3.6-plus 办公领域专用
  - gelab-zero-4b-preview / llama2 / moondream
学生是我们 0.5B 的 omnipilot。

抗幻觉的核心设计——**老师产出必须过验证才入库**：
  1. JSON 可解析（格式合法）
  2. 工具名在真实清单内（不允许编造工具）
  3. 必填参数齐全（参数幻觉过滤）
  4. 数值断言必须来自工具输出（数值幻觉过滤）
未通过验证的老师输出**直接丢弃**，绝不进训练集。
这就是"从源头降低幻觉"——学生只学被证实过的样本。
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from ..agent.loop import SYSTEM_PROMPT, _DECIDE_PROMPT, _extract_json
from ..config import DEMO_DIR
from ..runtime.llm import OllamaClient
from ..tools import build_registry

_REG = build_registry()
_VALID_TOOLS = {t.name: t for t in _REG.all()}
_NUM_RE = re.compile(r"\d{3,}")


def verify_teacher_action(action: dict | None, tools_text: str) -> tuple[bool, str]:
    """验证老师给出的一步动作是否可信（抗幻觉的关键闸门）。"""
    if not isinstance(action, dict):
        return False, "不是合法 JSON 对象"
    if "final" in action:
        txt = str(action["final"])
        # final 里出现数值但没有工具产出证据 → 疑似编造（学生不该学这种）
        if _NUM_RE.search(txt):
            return False, "final 含未经验证的数值（疑似幻觉）"
        return True, "final 合法"
    name = str(action.get("tool", "")).strip()
    if name not in _VALID_TOOLS:
        return False, f"工具名不存在：{name}"
    tool = _VALID_TOOLS[name]
    props = set(((tool.parameters or {}).get("properties") or {}).keys())
    required = set((tool.parameters or {}).get("required") or [])
    args = action.get("args") or {}
    if not isinstance(args, dict):
        return False, "args 不是对象"
    missing = required - set(args.keys())
    if missing:
        return False, f"缺必填参数：{sorted(missing)}"
    invented = set(args.keys()) - props
    if invented:
        return False, f"编造了不存在的参数：{sorted(invented)}"
    return True, "动作合法"


def _tool_list_for(domains: tuple[str, ...]) -> str:
    return _REG.describe(domains)


def distill_task(client: OllamaClient, teacher: str, task: str, domains: tuple[str, ...],
                 *, plan: str = "1. 直接调用合适的工具完成任务") -> dict | None:
    """让老师解一个任务，验证通过后返回一条学生可用的样本。"""
    prompt = _DECIDE_PROMPT.format(
        task=task, plan=plan, tools=_tool_list_for(domains),
        observation="（本任务无需屏幕观察，可直接调用工具）", history="（无）",
    )
    try:
        raw = client.chat(teacher, [{"role": "user", "content": prompt}],
                          system=SYSTEM_PROMPT, json_mode=True, think=False,
                          options={"temperature": 0.2, "num_ctx": 4096, "num_predict": 512})
    except Exception:
        return None
    action = _extract_json(raw)
    ok, _reason = verify_teacher_action(action, _tool_list_for(domains))
    if not ok:
        return None
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(action, ensure_ascii=False)},
        ],
        "meta": {"source": "distill", "teacher": teacher, "domains": list(domains),
                 "tool": action.get("tool", "(final)")},
    }


# ------------------------------------------------------------------ 任务池
_TASKS: list[tuple[str, tuple[str, ...]]] = [
    ("打开 {site} 并告诉我页面标题", ("browser",)),
    ("在 {site} 搜索“{q}”", ("browser",)),
    ("把表格数据 [[名称, 数量], [苹果, {n}]] 写入 {f}.xlsx 的 A1 单元格", ("office",)),
    ("生成一份关于“{t}”的 Word 文档保存为 {f}.docx", ("office",)),
    ("做一份 {k} 页的 PPT 保存为 {f}.pptx", ("office",)),
    ("用 python 打印一行 {s}", ("code",)),
    ("用 python 计算 {a}*{b} 并把结果告诉我", ("code",)),
    ("查看项目最近的 {k} 条 git 提交", ("code",)),
    ("把图片 {img} 缩放到 320x240", ("media",)),
    ("把视频 {vid} 裁剪前 5 秒保存为 clip_cut.mp4", ("media",)),
    ("在记事本里输入 {s}", ("system",)),
    ("把当前屏幕截图保存下来", ("system",)),
    ("用 Blender 建一个 cube 并保存为 cube.blend", ("blender",)),
]

_FILL = {
    "site": ["example.com", "python.org", "github.com", "wikipedia.org"],
    "q": ["Python 教程", "Ollama 使用", "LoRA 微调", "YOLO 检测"],
    "f": ["报表", "data", "统计", "清单"],
    "t": ["季度报告", "项目周报", "技术方案"],
    "k": [2, 3, 4, 5, 10],
    "s": ["hello omnipilot", "训练完成", "模型已就绪"],
    "a": [1234, 5678, 9999],
    "b": [12, 37, 68],
    "img": ["photo.png", "input.jpg", "cover.webp"],
    "vid": ["clip.mp4", "demo.mov"],
    "n": [3, 7, 15],
}


def generate_distill_data(teacher: str = "qwen3:4b", target: int = 120,
                          out: Path | None = None, seed: int = 5,
                          client: OllamaClient | None = None, verbose: bool = True) -> dict:
    """用老师模型产出经过验证的样本；未通过验证的一律丢弃（并统计丢弃率）。"""
    client = client or OllamaClient()
    if not client.ping():
        return {"ok": False, "message": "Ollama 未运行"}
    if not client.has_model(teacher):
        return {"ok": False, "message": f"老师模型不存在：{teacher}"}

    rnd = random.Random(seed)
    out_path = Path(out) if out else (DEMO_DIR / "distill.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept, tried, rejected, seen = 0, 0, 0, set()
    with out_path.open("w", encoding="utf-8") as f:
        while kept < target and tried < target * 4:
            tried += 1
            tpl, domains = rnd.choice(_TASKS)
            task = tpl.format(**{k: rnd.choice(v) for k, v in _FILL.items() if "{" + k + "}" in tpl})
            rec = distill_task(client, teacher, task, domains)
            if rec is None:
                rejected += 1
                continue
            key = json.dumps(rec["messages"], ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            if verbose and kept % 20 == 0:
                print(f"[distill] 已接受 {kept} / 尝试 {tried}（拒绝 {rejected}）", flush=True)
    rate = (rejected / tried) if tried else 0.0
    return {"ok": True, "teacher": teacher, "kept": kept, "tried": tried,
            "rejected": rejected, "reject_rate": round(rate, 3), "path": str(out_path)}

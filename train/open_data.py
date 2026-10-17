"""定向训练 · 开源数据接入（真实数据）。

把公开的工具调用数据集（如 ToolACE / Team-ACE，apache-2.0）转换成 OmniPilot 的
智能体训练格式，用于让模型学会"读懂提示里的工具清单 → 输出正确的 JSON 动作"。

为什么这很关键：自造数据只有我们自己的 41 个工具，模型没见过"任意工具 schema"。
引入真实多样化的工具 schema 后，模型对未见工具名的泛化与 JSON 服从性显著提升。

ToolACE 原始格式：
    {"system": "....函数清单 JSON...", "conversations": [
        {"from": "user"|"assistant"|"tool", "value": "..."}]}
    助手工具调用形如： [Market Trends API(trend_type="MARKET_INDEXES", country="us")]
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..agent.loop import SYSTEM_PROMPT, _DECIDE_PROMPT
from ..config import DEMO_DIR

_CALL_RE = re.compile(r"\[?\s*([A-Za-z_][\w \-\.]*?)\s*\((.*?)\)\s*\]?", re.DOTALL)
_FUNC_BLOCK_RE = re.compile(r"(?:functions|tools)[^\[]*(\[.*)", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------- 解析
def _balanced_json_array(text: str) -> list | None:
    """从文本中找出第一个完整的 JSON 数组并解析。"""
    start = text.find("[")
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
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def parse_toolace_functions(system_text: str) -> list[dict]:
    arr = _balanced_json_array(system_text)
    if not isinstance(arr, list):
        return []
    out = []
    for f in arr:
        if isinstance(f, dict) and f.get("name"):
            out.append(f)
    return out


def render_tools(functions: list[dict], limit: int = 20) -> str:
    """把函数清单渲染成 OmniPilot 的工具清单文本。"""
    lines = []
    for f in functions[:limit]:
        params = f.get("parameters") or {}
        props = params.get("properties") or {}
        required = set(params.get("required") or [])
        arg_str = ", ".join(f"{k}*" if k in required else k for k in props)
        desc = (f.get("description") or "").strip().replace("\n", " ")[:80]
        lines.append(f"- {f['name']}({arg_str}): {desc}")
    return "\n".join(lines)


def _split_args(s: str) -> list[str]:
    """按顶层逗号切分参数串（忽略括号/引号内的逗号）。"""
    parts, depth, in_str, esc, cur = [], 0, False, False, []
    for c in s:
        if esc:
            cur.append(c)
            esc = False
            continue
        if c == "\\":
            cur.append(c)
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            cur.append(c)
            continue
        if not in_str:
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            elif c == "," and depth == 0:
                parts.append("".join(cur))
                cur = []
                continue
        cur.append(c)
    if cur:
        parts.append("".join(cur))
    return parts


def _coerce(v: str):
    v = v.strip()
    if not v:
        return ""
    try:
        return json.loads(v)
    except Exception:
        pass
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        return v[1:-1]
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() in ("none", "null"):
        return None
    return v


def parse_call(text: str) -> tuple[str, dict] | None:
    """解析 '[Name(a=1, b="x")]' -> (name, {a:1, b:"x"})。"""
    m = _CALL_RE.search(text or "")
    if not m:
        return None
    name = m.group(1).strip()
    args = {}
    for chunk in _split_args(m.group(2)):
        if "=" not in chunk:
            continue
        k, _, v = chunk.partition("=")
        k = k.strip()
        if k:
            args[k] = _coerce(v)
    return name, args


# ---------------------------------------------------------------------- 转换
def toolace_to_samples(record: dict, *, max_history: int = 4) -> list[dict]:
    """把一条 ToolACE 多轮记录转成若干条 OmniPilot 单步决策样本。"""
    system_text = record.get("system", "")
    funcs = parse_toolace_functions(system_text)
    if not funcs:
        return []
    tools_text = render_tools(funcs)

    conv = record.get("conversations") or []
    task = ""
    for m in conv:
        if m.get("from") == "user":
            task = str(m.get("value", "")).strip()
            break
    if not task:
        return []

    samples: list[dict] = []
    history: list[str] = []

    for m in conv:
        role = m.get("from")
        val = str(m.get("value", "")).strip()
        if role == "user":
            continue
        if role == "assistant":
            parsed = parse_call(val)
            if parsed:
                name, args = parsed
                action = {"thought": f"调用 {name} 完成任务",
                          "tool": name.replace(" ", "_"), "args": args}
                obs = "（本任务无需屏幕观察，可直接调用工具）" if not history else "（参考上述执行结果）"
                prompt = _DECIDE_PROMPT.format(task=task, plan=f"1. {task}",
                                               tools=tools_text, observation=obs,
                                               history="\n".join(history[-max_history:]) or "（无）")
                samples.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": json.dumps(action, ensure_ascii=False)},
                    ],
                    "meta": {"source": "toolace", "tool": name, "open_data": True},
                })
                history.append(f"#{len(history)+1} {name}({json.dumps(args, ensure_ascii=False)[:100]}) -> 成功: 已返回结果")
            else:
                # 纯文本回答 → final
                if not val:
                    continue
                prompt = _DECIDE_PROMPT.format(task=task, plan=f"1. {task}",
                                               tools=tools_text,
                                               observation="（参考上述执行结果）",
                                               history="\n".join(history[-max_history:]) or "（无）")
                samples.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant",
                         "content": json.dumps({"thought": "信息已足够，汇总答复",
                                                "final": val[:300]}, ensure_ascii=False)},
                    ],
                    "meta": {"source": "toolace", "tool": "(final)", "open_data": True},
                })
        elif role == "tool":
            history.append(f"#{len(history)+1} (工具返回) -> {val[:150]}")
    return samples


def load_json_records(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def convert_file(src: str | Path, out: str | Path | None = None,
                 limit: int = 3000, seed: int = 11) -> dict:
    """把 ToolACE 文件转成 OmniPilot 样本 JSONL。"""
    import random

    rnd = random.Random(seed)
    records = load_json_records(src)
    rnd.shuffle(records)
    out = Path(out) if out else (DEMO_DIR / "open_toolace.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    n, kept = 0, 0
    seen: set[str] = set()
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            n += 1
            for s in toolace_to_samples(rec):
                blob = json.dumps(s["messages"], ensure_ascii=False, sort_keys=True)
                if blob in seen:
                    continue
                seen.add(blob)
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
                kept += 1
                break  # 每条原始记录只取第一条样本，保证来源多样
            if kept >= limit:
                break
    return {"ok": True, "records_scanned": n, "samples": kept, "path": str(out)}

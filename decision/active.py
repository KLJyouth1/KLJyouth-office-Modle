"""主动求知决策：不确定时先查、再记、然后决策（对应"不知道就去搜"）。

三层行为递进，全部基于可观测信号（不是拍脑袋）：
1. **先查自己的记忆**：命中知识库就直接注入上下文再决策——零成本，且天然抗遗忘。
2. **记忆不够就上网查**：从外部取数 → 过抗幻觉闸门 → 入库 → 用新知识重决策一次。
   这就是"人能上网一步步搜、记下来、以后再用"的机器版本。
3. **查完仍不可信就如实报告**：绝不假装成功。返回的诊断里写清尝试了哪些途径、
   卡在哪一步，交给上层决定（回退启发式 / 直接弃答）。

为什么要独立成模块而不塞进 engine：求知是**有副作用的动作**（联网、写库），
而 engine 是纯函数式打分。把副作用隔离出来，engine 才能保持可测、可复现。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..knowledge.store import FACT, KnowledgeStore
from .engine import DecisionEngine

# 触发求知的条件
MIN_CONF_TO_ACCEPT = 0.55          # 首次决策置信度达到它就直接采信，不必求知


@dataclass
class ActiveResult:
    """一次主动求知决策的完整结果与过程记录。"""

    answer: dict
    attempts: list[dict] = field(default_factory=list)
    acquired: bool = False
    reliable: bool = False

    def as_dict(self) -> dict:
        return {"answer": self.answer, "attempts": self.attempts,
                "acquired": self.acquired, "reliable": self.reliable}


class ActiveDecider:
    """用"记忆 + 求知"增强的决策器。"""

    def __init__(self, engine: DecisionEngine, store: KnowledgeStore, *,
                 fetch: Callable[[str], str] | None = None,
                 min_conf: float = MIN_CONF_TO_ACCEPT) -> None:
        self.engine = engine
        self.store = store
        self.fetch = fetch
        self.min_conf = min_conf

    # ------------------------------------------------------------------ 内部
    def _probe(self, state: str, questions: dict) -> dict:
        """跑一次决策并提炼可信度。"""
        out = self.engine.decide(state, questions)
        diags = out.get("diagnostics") or {}
        letter_ok = all((d or {}).get("letter_ok") for d in diags.values()) if diags else False
        conf = 0.0
        for ans in (out.get("answers") or {}).values():
            t = ans.get("type")
            try:
                if t == "choice":
                    conf = max(conf, float((ans.get("probabilities") or {}).get(ans.get("choice"), 0.0)))
                elif t == "noul":
                    p = float(ans.get("noul", 0.0))
                    conf = max(conf, p if p >= 0.5 else 1 - p)
                elif t == "score":
                    probs = [float(v) for v in (ans.get("probabilities") or {}).values()]
                    conf = max(conf, max(probs) if probs else 0.0)
            except (TypeError, ValueError):
                continue
        return {"out": out, "letter_ok": letter_ok, "conf": round(conf, 4),
                "reliable": bool(letter_ok and conf >= self.min_conf)}

    # ------------------------------------------------------------------ 主流程
    def decide(self, state: str, questions: dict, *, topic: str = "") -> dict:
        """返回 {answer, attempts, acquired, reliable}（见 ActiveResult）。"""
        attempts: list[dict] = []

        # 1) 直接用模型（原始状态）
        first = self._probe(state, questions)
        attempts.append({"stage": "direct", "letter_ok": first["letter_ok"],
                         "conf": first["conf"], "reliable": first["reliable"]})
        acquired = False
        best = first

        # 2) 命中自身记忆 → 把知识注入状态再决策（零外部依赖，优先于联网）
        block = self.store.prompt_block(state or topic, k=5)
        if block:
            enriched = f"{block}\n\n{state}"
            mem = self._probe(enriched, questions)
            attempts.append({"stage": "memory", "letter_ok": mem["letter_ok"],
                             "conf": mem["conf"], "reliable": mem["reliable"],
                             "knowledge_chars": len(block)})
            if mem["reliable"] or mem["conf"] > best["conf"]:
                best = mem

        # 3) 仍不可信且有联网能力 → 主动求知，用新知识重决策一次
        if not best["reliable"] and self.fetch is not None:
            q = f"{topic or state} 相关信息"
            ok, why = self.store.acquire(topic or state[:60], q, self.fetch, kind=FACT, source="web")
            attempts.append({"stage": "acquire", "ok": ok, "why": why})
            if ok:
                acquired = True
                block2 = self.store.prompt_block(state or topic, k=5)
                if block2:
                    enriched2 = f"{block2}\n\n{state}"
                    after = self._probe(enriched2, questions)
                    attempts.append({"stage": "after_acquire", "letter_ok": after["letter_ok"],
                                     "conf": after["conf"], "reliable": after["reliable"]})
                    if after["reliable"] or after["conf"] > best["conf"]:
                        best = after

        return ActiveResult(answer=best["out"], attempts=attempts,
                            acquired=acquired, reliable=best["reliable"]).as_dict()

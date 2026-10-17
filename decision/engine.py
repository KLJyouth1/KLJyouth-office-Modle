"""Jev 式决策引擎：prefill-only 候选概率（本地 Ollama 实现）。

为什么是 prefill-only：NeoHorse-Jev 的核心卖点是**一次前向计算直接给出候选分布**，
而不是逐字生成答案（省掉解码等待，单次决策 ~74ms）。在 Ollama 上等价的做法：
把候选渲染成 A/B/C 字母选项，只解码极少量 token，读 `logprobs` 里各字母的
logprob → 各候选的概率。一次请求 = 一次决策。

为什么用"字母选项"而不是让模型输出候选原文：字母一定是单 token（ASCII），
不会因中文/多 token 候选而被拆成若干 token，从而保证概率能一对一映射回候选。
这也正是 NeoHorse-Jev 官方对基座模型采用的 letter-logit 提示法。

本机实测（Ollama 0.15.2）踩到的三个坑，均已在此处规避：
1. `num_predict=1` 会让 Ollama **丢掉 logprobs**（qwen3 必现、qwen2.5 不现）→ 取 2。
2. 思考型模型（qwen3）首 token 是 "We"/"First" 等推理词，`think=false` 在此无效，
   拿不到候选字母 → 必须能被显式识别（`letter_ok=False`），交给调用方回退，而不是假装成功。
3. 未经字母约定训练的模型（如 omnipilot-1.5b-v2）首 token 会是候选原文（"media"），
   此时按字母匹配会**静默给出错误结论** → `letter_ok` 就是用来暴露这一点的。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import PLANNER_MODEL
from ..runtime.llm import OllamaClient, OllamaError
from . import schema

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# 未出现在 top_logprobs 里的候选，给一个极小概率（后续会归一化）。
_FLOOR = 1e-9

# 见上文坑 1：必须 > 1，否则 Ollama 可能不返回 logprobs。
_NUM_PREDICT = 2


def render_options(candidates: list[str], criteria: dict[str, str] | None = None) -> str:
    """把候选渲染成 A) … / B) … 的选项块。criteria 存在时附上说明。"""
    lines = []
    for i, c in enumerate(candidates):
        desc = (criteria or {}).get(c)
        lines.append(f"{_LETTERS[i]}) {c}" + (f" — {desc}" if desc else ""))
    return "\n".join(lines)


def build_prompt(state: str, instructions: str, candidates: list[str],
                 criteria: dict[str, str] | None = None) -> str:
    """构造 letter-logit 提示：状态 + 问题 + 选项 + 只要字母。"""
    return (
        f"State:\n{state}\n\n"
        f"Question: {instructions}\n\n"
        f"Options:\n{render_options(candidates, criteria)}\n\n"
        f"Answer with only the option letter "
        f"({', '.join(_LETTERS[:len(candidates)])}), nothing else."
    )


@dataclass
class CandidateScores:
    """一次候选打分的结果：概率分布 + 可信度诊断。"""

    probs: list[float]                       # 长度 == 候选数，已归一化
    letter_ok: bool                          # 模型是否按字母作答（诚实性/格式有效性）
    sampled: str = ""                        # 实际生成的文本（诊断用）
    top_tokens: list[str] = field(default_factory=list)  # 首个位置 top-k 的 token 串

    def best_index(self) -> int:
        return max(range(len(self.probs)), key=lambda i: self.probs[i])


def _letter_probabilities(logprobs: list, n: int) -> tuple[list[float], bool, str, list[str]]:
    """把 Ollama 的 logprobs 转成 n 个候选的原始概率。

    返回 (raw, letter_ok, sampled, top_tokens)：
    - raw: exp(logprob)，未命中候选记 _FLOOR；
    - letter_ok: **采样到的**首个 token 本身就是一个候选字母（模型确实按字母作答）；
    - sampled: 实际生成的文本（可能为空，如思考型模型）；
    - top_tokens: 首个位置 top-k 的 token，便于诊断/评测。
    """
    if not logprobs:
        return [_FLOOR] * n, False, "", []
    first = logprobs[0] or {}
    tops = first.get("top_logprobs") or []
    raw = [_FLOOR] * n
    top_tokens: list[str] = []
    for entry in tops:
        tok = (entry.get("token") or "").strip()
        lp = entry.get("logprob")
        top_tokens.append((entry.get("token") or ""))
        if len(tok) != 1 or tok.upper() not in _LETTERS or lp is None:
            continue
        idx = _LETTERS.index(tok.upper())
        if idx >= n:
            continue
        raw[idx] = max(math.exp(float(lp)), _FLOOR)
    sampled = str(first.get("token") or "").strip()
    letter_ok = len(sampled) == 1 and sampled.upper() in _LETTERS[:n]
    return raw, letter_ok, sampled, top_tokens


def _normalize(raw: list[float]) -> list[float]:
    total = sum(raw)
    if total <= 0:
        n = len(raw)
        return [1.0 / n] * n
    return [p / total for p in raw]


class DecisionEngine:
    """把 {state, questions} 变成带概率的决策（Choice / Noul / Score）。"""

    def __init__(self, client: OllamaClient | None = None, model: str = PLANNER_MODEL,
                 *, top_logprobs: int = 20, keep_alive: str = "10m") -> None:
        self.client = client or OllamaClient()
        self.model = model
        # top_logprobs 至少要覆盖候选数：候选多于 top-k 时会漏掉字母 → 概率塌陷。
        self.top_logprobs = top_logprobs
        self.keep_alive = keep_alive

    # ------------------------------------------------------------------ 打分
    def score_candidates(self, state: str, instructions: str, candidates: list[str],
                         criteria: dict[str, str] | None = None) -> CandidateScores:
        """返回候选概率分布（已归一化，和为 1）与可信度诊断。"""
        if not candidates:
            raise schema.SchemaError("候选列表为空，无法决策")
        if len(candidates) > len(_LETTERS):
            raise schema.SchemaError(f"候选数 {len(candidates)} 超过 {len(_LETTERS)}，无法用字母编码")
        prompt = build_prompt(state, instructions, candidates, criteria)
        k = max(self.top_logprobs, len(candidates) + 2)
        data = self.client.chat_logprobs(
            self.model,
            [{"role": "user", "content": prompt}],
            top_logprobs=k,
            # temperature=0：只取极少量 token，采样温度不应改变候选分布；
            # think=False：对支持的模型关闭思考（对 qwen3 无效，由 letter_ok 暴露）。
            options={"temperature": 0, "num_predict": _NUM_PREDICT},
            keep_alive=self.keep_alive,
            think=False,
        )
        if "logprobs" not in data:
            raise OllamaError(
                f"Ollama({self.model}) 未返回 logprobs 字段；决策层需要 Ollama>=0.12 的 logprobs 支持"
            )
        raw, letter_ok, sampled, top_tokens = _letter_probabilities(data.get("logprobs") or [], len(candidates))
        if all(p <= _FLOOR for p in raw):
            raise OllamaError(
                f"Ollama({self.model}) 的首 token top_logprobs 中没有任何候选字母；"
                f"该模型可能不遵循字母作答（首 token={sampled!r}）"
            )
        return CandidateScores(probs=_normalize(raw), letter_ok=letter_ok, sampled=sampled,
                               top_tokens=top_tokens)

    # ------------------------------------------------------------------ 决策
    def decide(self, state: str, questions: dict) -> dict:
        """吃'原始问题 dict'（官方形状），解析后作答。"""
        parsed = {qid: schema.parse_question(qid, q) for qid, q in questions.items()}
        return self.answer(state, parsed)

    def answer(self, state: str, parsed: dict) -> dict:
        """对'已解析的问题'各做一次决策，返回官方形状响应（附 diagnostics）。"""
        answers: dict[str, dict] = {}
        diagnostics: dict[str, dict] = {}
        for qid, q in parsed.items():
            cands = q.candidates()
            if isinstance(q, schema.ChoiceQuestion):
                sc = self.score_candidates(state, q.instructions, cands, q.criteria)
                answers[qid] = schema.choice_answer(cands[sc.best_index()],
                                                    {c: p for c, p in zip(cands, sc.probs)})
            elif isinstance(q, schema.NoulQuestion):
                sc = self.score_candidates(state, q.instructions, cands)
                answers[qid] = schema.noul_answer(sc.probs[1], sc.probs[0])
            else:  # ScoreQuestion
                sc = self.score_candidates(state, q.instructions, cands)
                expected = sum(i * p for i, p in enumerate(sc.probs))
                answers[qid] = schema.score_answer(expected, cands, sc.probs)
            diagnostics[qid] = {"letter_ok": sc.letter_ok, "sampled": sc.sampled,
                                "top_tokens": sc.top_tokens}
        return schema.make_response(answers, model=self.model, input_tokens=None,
                                    diagnostics=diagnostics)

    def decide_request(self, request: dict) -> dict:
        """直接吃官方形状的请求 dict（先解析，再作答，不重复解析）。"""
        state, parsed = schema.parse_request(request)
        return self.answer(state, parsed)

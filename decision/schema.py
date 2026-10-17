"""决策契约：Choice / Noul / Score 三接口的请求与响应结构（零重依赖）。

对齐 NeoHorse-Jev（Apache-2.0）的 SystemOne 契约：请求是"状态 + 若干问题"，
响应是"每条问题一个答案 + 概率分布"，**不做逐字生成**。

为什么单独抽出契约：这套结构同时被 1) 本地决策引擎、2) 训练数据构造、
3) 评测台、4) 上层应用（omnipilot / Marguerite）消费。任何一处硬编码字段名
都会导致上下游契约错位，因此字段名与官方 example_request.json 逐字对齐：

请求：
    {"state": "...",
     "questions": {
        "move":    {"type": "choice", "instructions": "选一个安全路线",
                    "criteria": {"left": "左侧通畅", "forward": "前方堵塞"}},
        "blocked": {"type": "noul", "instructions": "前方是否堵塞？"},
        "risk":    {"type": "score", "instructions": "直行的风险等级",
                    "criteria": ["low", "medium", "high"]}}}

响应：
    {"answers": {
        "move":    {"type": "choice", "choice": "left",
                    "probabilities": {"left": 0.82, "forward": 0.18}},
        "blocked": {"type": "noul", "noul": 0.91,
                    "probabilities": {"false": 0.09, "true": 0.91}},
        "risk":    {"type": "score", "score": 1.2, "legend": ["low", "medium", "high"],
                    "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}}}}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

# 三种决策类型
CHOICE = "choice"
NOUL = "noul"
SCORE = "score"
_TYPES = (CHOICE, NOUL, SCORE)


class SchemaError(ValueError):
    """请求结构不合法（缺字段、类型不对、候选为空等）。"""


@dataclass
class ChoiceQuestion:
    """从若干候选里选一个。criteria 是候选 -> 说明的映射，键即返回的 choice。"""

    instructions: str
    criteria: dict[str, str]

    def candidates(self) -> list[str]:
        return list(self.criteria.keys())


@dataclass
class NoulQuestion:
    """判是非。返回 P(true)。"""

    instructions: str

    def candidates(self) -> list[str]:
        return ["false", "true"]


@dataclass
class ScoreQuestion:
    """按有序等级打分。criteria 是等级列表（低 -> 高），返回期望等级下标。"""

    instructions: str
    criteria: list[str]

    def candidates(self) -> list[str]:
        return list(self.criteria)


Question = Union[ChoiceQuestion, NoulQuestion, ScoreQuestion]


def parse_question(qid: str, raw: Any) -> Question:
    """把一条原始问题解析成 Question；结构不对则抛 SchemaError（不静默兜底）。"""
    if not isinstance(raw, dict):
        raise SchemaError(f"问题 {qid!r} 必须是对象，实际是 {type(raw).__name__}")
    qtype = raw.get("type")
    instructions = raw.get("instructions")
    if qtype not in _TYPES:
        raise SchemaError(f"问题 {qid!r} 的 type 必须是 {_TYPES} 之一，实际是 {qtype!r}")
    if not isinstance(instructions, str) or not instructions.strip():
        raise SchemaError(f"问题 {qid!r} 缺少非空 instructions")

    if qtype == CHOICE:
        crit = raw.get("criteria")
        if not isinstance(crit, dict) or not crit:
            raise SchemaError(f"choice 问题 {qid!r} 的 criteria 必须是非空对象")
        if not all(isinstance(k, str) and k for k in crit):
            raise SchemaError(f"choice 问题 {qid!r} 的 criteria 键必须是非空字符串")
        return ChoiceQuestion(instructions=instructions, criteria=dict(crit))

    if qtype == SCORE:
        crit = raw.get("criteria")
        if not isinstance(crit, list) or not crit:
            raise SchemaError(f"score 问题 {qid!r} 的 criteria 必须是非空列表")
        if not all(isinstance(c, str) and c.strip() for c in crit):
            raise SchemaError(f"score 问题 {qid!r} 的 criteria 每项必须是非空字符串")
        if len(crit) < 2:
            raise SchemaError(f"score 问题 {qid!r} 至少需要 2 个等级")
        return ScoreQuestion(instructions=instructions, criteria=list(crit))

    return NoulQuestion(instructions=instructions)


def parse_request(raw: Any) -> tuple[str, dict[str, Question]]:
    """解析 {state, questions}；返回 (state, {qid: Question})。"""
    if not isinstance(raw, dict):
        raise SchemaError(f"请求必须是对象，实际是 {type(raw).__name__}")
    state = raw.get("state")
    questions = raw.get("questions")
    if not isinstance(state, str) or not state.strip():
        raise SchemaError("请求缺少非空 state")
    if not isinstance(questions, dict) or not questions:
        raise SchemaError("请求缺少非空 questions")
    return state, {qid: parse_question(qid, q) for qid, q in questions.items()}


# ---------------------------------------------------------------- 响应构造
def choice_answer(choice: str, probabilities: dict[str, float]) -> dict:
    return {"type": CHOICE, "choice": choice, "probabilities": probabilities}


def noul_answer(prob_true: float, prob_false: float) -> dict:
    return {"type": NOUL, "noul": float(prob_true),
            "probabilities": {"false": float(prob_false), "true": float(prob_true)}}


def score_answer(expected: float, legend: list[str], probs: list[float]) -> dict:
    return {"type": SCORE, "score": float(expected), "legend": list(legend),
            "probabilities": {str(i): float(p) for i, p in enumerate(probs)}}


def make_response(answers: dict[str, dict], *, model: str | None = None,
                  input_tokens: int | None = None,
                  diagnostics: dict | None = None) -> dict:
    """组装响应。`answers` 严格保持官方形状；`diagnostics` 是本实现的附加诊断
    （官方没有该字段，但下游可用它判断决策是否可信、决定是否回退启发式）。"""
    out: dict = {"answers": answers}
    if model is not None:
        out["model"] = model
    if input_tokens is not None:
        out["input_tokens"] = int(input_tokens)
    if diagnostics is not None:
        out["diagnostics"] = diagnostics
    return out


# ---------------------------------------------------------------- 判分用工具
@dataclass
class Gold:
    """一条评测样本的标准答案（用于评测台判分）。"""

    qid: str
    type: str
    value: Any                     # choice: 候选键；noul: bool；score: 等级下标
    meta: dict = field(default_factory=dict)


def answer_is_correct(gold: Gold, answer: dict, *, score_tol: int = 0) -> bool:
    """判断一个答案是否与标准答案一致。

    - choice：返回的 choice 键必须与标准候选一致。
    - noul  ：以 0.5 阈值判是非。
    - score ：**按概率取 argmax 等级**再比较（官方 Nimble 的口径就是"评级等级精确匹配"）。
      不能拿浮点期望值直接比等级——期望值几乎不可能是整数，那样会让正确率恒为 0。
    """
    if not isinstance(answer, dict):
        return False
    if gold.type == CHOICE:
        return answer.get("choice") == gold.value
    if gold.type == NOUL:
        # 概率阈值 0.5 判是非；标准答案 True/False
        return (float(answer.get("noul", 0.0)) >= 0.5) == bool(gold.value)
    if gold.type == SCORE:
        probs = answer.get("probabilities")
        try:
            if isinstance(probs, dict) and probs:
                idx = max(range(len(probs)), key=lambda i: float(probs[str(i)]))
            else:
                idx = int(round(float(answer.get("score", -99.0))))
        except (TypeError, ValueError, KeyError):
            return False
        return abs(idx - int(gold.value)) <= score_tol
    return False

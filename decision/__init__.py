"""决策能力层：Choice / Noul / Score 三接口 + prefill-only 候选概率打分。

对外只暴露必要符号；契约细节见 schema.py，引擎实现见 engine.py。
"""
from __future__ import annotations

from .active import MIN_CONF_TO_ACCEPT, ActiveDecider, ActiveResult
from .engine import CandidateScores, DecisionEngine, build_prompt, render_options
from .experts import GATE_MIN_CONF, Expert, ExpertRegistry, ExpertRouter
from .schema import (
    CHOICE,
    NOUL,
    SCORE,
    ChoiceQuestion,
    Gold,
    NoulQuestion,
    Question,
    SchemaError,
    ScoreQuestion,
    answer_is_correct,
    make_response,
    parse_question,
    parse_request,
)

__all__ = [
    "CHOICE", "NOUL", "SCORE",
    "ChoiceQuestion", "NoulQuestion", "ScoreQuestion", "Question",
    "Gold", "SchemaError",
    "answer_is_correct", "make_response", "parse_question", "parse_request",
    "DecisionEngine", "CandidateScores", "build_prompt", "render_options",
    "Expert", "ExpertRegistry", "ExpertRouter", "GATE_MIN_CONF",
    "ActiveDecider", "ActiveResult", "MIN_CONF_TO_ACCEPT",
]

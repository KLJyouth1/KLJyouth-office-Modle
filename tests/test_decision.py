"""决策层（Choice/Noul/Score）回归测试 —— 不依赖 Ollama、模型与网络。

覆盖本轮实测暴露的三个真实缺陷及其修复：
1. **`num_predict=1` 会让 Ollama 丢掉 logprobs** → 引擎固定用 2，且缺字段时显式报错而非静默。
2. **思考型/未按字母训练的模型首 token 不是候选字母** → 必须靠 `letter_ok` 暴露，
   而不是静默按字母取概率给出错误结论。
3. **`decide_request` 二次解析** → 解析与作答分离，避免 SchemaError。

同时锁定契约字段（与官方 example_request.json 一致），防止上下游漂移。
"""
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from omnipilot.agent import router  # noqa: E402
from omnipilot.decision import schema  # noqa: E402
from omnipilot.decision.engine import (  # noqa: E402
    CandidateScores,
    DecisionEngine,
    _letter_probabilities,
    _normalize,
)
from omnipilot.decision.experts import Expert, ExpertRegistry, ExpertRouter  # noqa: E402
from omnipilot.runtime.llm import OllamaError  # noqa: E402


# ------------------------------------------------------------------ 替身
class FakeClient:
    """替身 OllamaClient：只伪造 chat_logprobs 的首位置 top_logprobs。"""

    def __init__(self, letters, *, drop_logprobs=False, content=None):
        self.letters = letters          # [(token, logprob), ...]
        self.drop = drop_logprobs
        self.content = content
        self.calls = 0

    def chat_logprobs(self, model, messages, **kw):
        self.calls += 1
        if self.drop:
            return {"message": {"content": "x"}}
        first = self.letters[0]
        return {
            "message": {"content": self.content if self.content is not None else first[0]},
            "logprobs": [{
                "token": first[0], "logprob": first[1],
                "top_logprobs": [{"token": t, "logprob": lp} for t, lp in self.letters],
            }],
        }


def _engine(letters, **kw) -> DecisionEngine:
    return DecisionEngine(client=FakeClient(letters, **kw), model="fake")


# ------------------------------------------------------------------ 契约
class TestSchemaContract:
    def test_parses_the_official_request_shape(self):
        state, qs = schema.parse_request({
            "state": "前方堵塞",
            "questions": {
                "move": {"type": "choice", "instructions": "选路线",
                         "criteria": {"left": "通畅", "forward": "堵塞"}},
                "blocked": {"type": "noul", "instructions": "是否堵塞？"},
                "risk": {"type": "score", "instructions": "风险", "criteria": ["低", "中", "高"]},
            },
        })
        assert state == "前方堵塞"
        assert qs["move"].candidates() == ["left", "forward"]
        assert qs["blocked"].candidates() == ["false", "true"]
        assert qs["risk"].candidates() == ["低", "中", "高"]

    @pytest.mark.parametrize("bad,why", [
        ({"type": "choise", "instructions": "x"}, "未知类型"),
        ({"type": "noul", "instructions": "  "}, "空 instructions"),
        ({"type": "choice", "instructions": "x", "criteria": {}}, "空 criteria"),
        ({"type": "choice", "instructions": "x", "criteria": ["a"]}, "criteria 非对象"),
        ({"type": "score", "instructions": "x", "criteria": ["只有一个"]}, "等级数不足"),
        ("不是对象", "非对象"),
    ])
    def test_rejects_malformed_questions(self, bad, why):
        with pytest.raises(schema.SchemaError):
            schema.parse_question("q", bad)

    def test_rejects_request_without_state(self):
        with pytest.raises(schema.SchemaError):
            schema.parse_request({"questions": {"a": {"type": "noul", "instructions": "x"}}})

    def test_answer_is_correct_covers_three_types(self):
        assert schema.answer_is_correct(schema.Gold("q", schema.CHOICE, "left"),
                                        {"choice": "left"})
        assert schema.answer_is_correct(schema.Gold("q", schema.NOUL, True), {"noul": 0.9})
        assert not schema.answer_is_correct(schema.Gold("q", schema.NOUL, False), {"noul": 0.9})
        assert schema.answer_is_correct(schema.Gold("q", schema.SCORE, 2), {"score": 2.1},
                                        score_tol=1)
        assert not schema.answer_is_correct(schema.Gold("q", schema.SCORE, 2), {"score": 0.4})

    def test_score_judged_by_argmax_level_not_expected_value(self):
        # 期望值是浮点（如 1.4），直接与整数等级比较会让正确率恒为 0。
        # 必须按概率 argmax 取等级（官方 Nimble 的口径）。
        ans = {"type": "score", "score": 1.4, "legend": ["低", "中", "高"],
               "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}}
        assert schema.answer_is_correct(schema.Gold("q", schema.SCORE, 1), ans)
        assert not schema.answer_is_correct(schema.Gold("q", schema.SCORE, 2), ans)
        assert schema.answer_is_correct(schema.Gold("q", schema.SCORE, 0), ans, score_tol=1)


# ------------------------------------------------------------------ 概率映射
class TestLetterProbabilities:
    def test_maps_letters_to_candidates_and_reports_letter_ok(self):
        lps = [{"token": "C", "logprob": math.log(0.7), "top_logprobs": [
            {"token": "C", "logprob": math.log(0.7)},
            {"token": "A", "logprob": math.log(0.2)},
            {"token": "B", "logprob": math.log(0.1)},
        ]}]
        raw, letter_ok, sampled, tops = _letter_probabilities(lps, 3)
        assert letter_ok is True and sampled == "C"
        assert raw == pytest.approx([0.2, 0.1, 0.7], abs=1e-9)
        assert tops[0] == "C"

    def test_marks_unreliable_when_first_token_is_candidate_text_not_a_letter(self):
        # 复现坑 2：未按字母训练的模型首 token 是候选原文（"media"）。
        lps = [{"token": "media", "logprob": -0.1, "top_logprobs": [
            {"token": "media", "logprob": -0.1},
            {"token": "A", "logprob": -5.0},
        ]}]
        _raw, letter_ok, sampled, _tops = _letter_probabilities(lps, 2)
        assert letter_ok is False and sampled == "media"

    def test_ignores_letters_beyond_candidate_count(self):
        lps = [{"token": "A", "logprob": -0.1, "top_logprobs": [
            {"token": "F", "logprob": -0.5},   # 越界：只有 2 个候选
            {"token": "A", "logprob": -0.1},
        ]}]
        raw, *_ = _letter_probabilities(lps, 2)
        assert raw[1] < raw[0]                  # B 未出现 → 极小值
        assert raw[0] > 0

    def test_empty_logprobs_returns_all_floor(self):
        raw, letter_ok, _s, _t = _letter_probabilities([], 3)
        assert letter_ok is False
        assert all(p <= 1e-9 for p in raw)

    @pytest.mark.parametrize("raw,expect", [
        ([0, 0, 0], [1 / 3, 1 / 3, 1 / 3]),
        ([1, 3], [0.25, 0.75]),
    ])
    def test_normalize(self, raw, expect):
        assert _normalize(raw) == pytest.approx(expect)


# ------------------------------------------------------------------ 引擎
class TestDecisionEngine:
    def test_choice_returns_official_shape_with_probabilities(self):
        out = _engine([("C", -0.1), ("A", -2.0), ("B", -3.0)]).decide(
            "状态", {"pick": {"type": "choice", "instructions": "选", "criteria": {"a": "", "b": "", "c": ""}}})
        ans = out["answers"]["pick"]
        assert ans["type"] == "choice" and ans["choice"] == "c"
        assert sum(ans["probabilities"].values()) == pytest.approx(1.0)
        assert out["diagnostics"]["pick"]["letter_ok"] is True

    def test_noul_returns_probability_of_true(self):
        out = _engine([("B", -0.05), ("A", -3.0)]).decide(
            "状态", {"q": {"type": "noul", "instructions": "是否？"}})
        ans = out["answers"]["q"]
        assert ans["type"] == "noul"
        assert ans["noul"] > 0.9
        assert ans["probabilities"]["false"] + ans["probabilities"]["true"] == pytest.approx(1.0)

    def test_score_returns_expected_level_index_and_legend(self):
        out = _engine([("B", -0.1), ("C", -2.0), ("A", -4.0)]).decide(
            "状态", {"r": {"type": "score", "instructions": "风险", "criteria": ["低", "中", "高"]}})
        ans = out["answers"]["r"]
        assert ans["type"] == "score" and ans["legend"] == ["低", "中", "高"]
        assert 0.0 <= ans["score"] <= 2.0
        assert set(ans["probabilities"]) == {"0", "1", "2"}

    def test_decide_request_does_not_reparse(self):
        # 复现坑 3：修复前 decide_request 会把已解析的 Question 再当原始 dict 解析 → SchemaError。
        out = _engine([("A", -0.05), ("B", -3.0)]).decide_request({
            "state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}})
        assert "q" in out["answers"]

    def test_raises_when_logprobs_missing_instead_of_silently_guessing(self):
        with pytest.raises(OllamaError):
            _engine([("A", -0.1)], drop_logprobs=True).decide(
                "s", {"q": {"type": "noul", "instructions": "x"}})

    def test_raises_when_no_candidate_letter_appears_at_all(self):
        with pytest.raises(OllamaError):
            _engine([("media", -0.1), ("office", -2.0)]).decide(
                "s", {"q": {"type": "choice", "instructions": "x", "criteria": {"m": "", "o": ""}}})

    def test_rejects_more_candidates_than_letters(self):
        crit = {f"c{i}": "" for i in range(27)}
        with pytest.raises(schema.SchemaError):
            _engine([("A", -0.1)]).decide(
                "s", {"q": {"type": "choice", "instructions": "x", "criteria": crit}})


# ------------------------------------------------------------------ 路由
class FakeEngine:
    def __init__(self, probs, letter_ok=True):
        self._probs, self._ok = probs, letter_ok

    def score_candidates(self, state, instructions, candidates, criteria=None):
        return CandidateScores(probs=list(self._probs), letter_ok=self._ok, sampled="A",
                               top_tokens=["A"])


class TestRouterModes:
    def test_heuristic_mode_returns_keyword_hits(self):
        assert router.classify("帮我裁一下视频分辨率", mode="heuristic") == ["media"]

    def test_hybrid_uses_decision_layer_only_when_heuristic_misses(self):
        # 关键词命中 → 快路（不调用决策层）
        assert router.classify("帮我裁一下视频分辨率", mode="hybrid") == ["media"]
        # 关键词未命中 → 决策层给出 domain
        probs = [0.01, 0.02, 0.01, 0.01, 0.01, 0.94]
        got = router.classify("请把这堆零散材料理一理", mode="hybrid",
                              engine=FakeEngine(probs))
        assert got == ["system"]

    def test_decision_layer_falls_back_when_model_is_not_letter_faithful(self):
        probs = [0.99, 0.001, 0.001, 0.001, 0.004, 0.003]
        assert router.classify_with_decision("x", FakeEngine(probs, letter_ok=False)) is None

    def test_decision_layer_falls_back_when_confidence_is_too_low(self):
        probs = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
        assert router.classify_with_decision("x", FakeEngine(probs)) is None

    def test_decision_layer_returns_domain_when_confident_and_letter_faithful(self):
        probs = [0.02, 0.05, 0.9, 0.01, 0.01, 0.01]
        assert router.classify_with_decision("x", FakeEngine(probs)) == ["code"]

    def test_hybrid_never_returns_empty(self):
        assert router.classify("", mode="hybrid") == ["system"]


# ------------------------------------------------------------------ 专家路由
class _BoomGate:
    def score_candidates(self, *a, **k):
        raise RuntimeError("gate down")


class _OnlyHas:
    def __init__(self, present):
        self.present = set(present)

    def has_model(self, name):
        return name in self.present


def _registry(names=("code_exp", "media_exp"), default="code_exp"):
    experts = [Expert(name=n, model=f"omnipilot16-{n}", desc=f"{n} 专家",
                      domain=n.split("_")[0]) for n in names]
    return ExpertRegistry(experts, default=default)


class TestExpertRouter:
    def test_single_expert_skips_the_gate_entirely(self):
        r = ExpertRouter(_registry(("only",)), client=FakeClient([("A", -0.1)]))
        e, diag = r.route("任意请求")
        assert e.name == "only" and diag["reliable"] is True

    def test_routes_to_the_highest_probability_expert(self):
        r = ExpertRouter(_registry(), client=FakeClient([("A", -0.1)]),
                         gate=FakeEngine([0.05, 0.95]))
        e, diag = r.route("任意请求")
        assert e.name == "media_exp"
        assert diag["reliable"] is True and diag["picked"] == "media_exp"

    def test_falls_back_to_default_when_gate_is_not_letter_faithful(self):
        # 复现坑 2 的后果：门控不可信时若照常路由，会把请求交给一个"其实没选对"的专家。
        r = ExpertRouter(_registry(), client=FakeClient([("A", -0.1)]),
                         gate=FakeEngine([0.05, 0.95], letter_ok=False))
        e, diag = r.route("任意请求")
        assert e.name == "code_exp" and diag["reliable"] is False

    def test_falls_back_to_default_when_gate_confidence_is_too_low(self):
        r = ExpertRouter(_registry(), client=FakeClient([("A", -0.1)]),
                         gate=FakeEngine([0.34, 0.33]))
        e, diag = r.route("任意请求")
        assert e.name == "code_exp" and diag["reliable"] is False

    def test_falls_back_to_default_when_gate_raises(self):
        r = ExpertRouter(_registry(), client=FakeClient([("A", -0.1)]), gate=_BoomGate())
        e, diag = r.route("任意请求")
        assert e.name == "code_exp" and diag["reliable"] is False
        assert "门控不可用" in diag["reason"]

    def test_returns_none_when_registry_is_empty(self):
        r = ExpertRouter(ExpertRegistry(), client=FakeClient([("A", -0.1)]))
        e, diag = r.route("任意请求")
        assert e is None and diag["reliable"] is False

    def test_prune_drops_experts_whose_ollama_model_is_missing(self):
        reg = _registry(("code_exp", "media_exp"), default="media_exp")
        missing = reg.prune_to_available(_OnlyHas({"omnipilot16-code_exp"}))
        assert missing == ["media_exp"]
        assert reg.names() == ["code_exp"]
        assert reg.default == "code_exp"          # 默认专家被剔掉时要改选

    def test_decide_routes_then_answers_with_the_selected_expert(self):
        r = ExpertRouter(_registry(), client=FakeClient([("A", -0.05), ("B", -3.0)]),
                         gate=FakeEngine([0.9, 0.1]))
        out = r.decide("状态", {"q": {"type": "noul", "instructions": "是否？"}})
        assert out["expert"] == "code_exp"
        assert out["answers"]["q"]["type"] == "noul"
        assert out["diagnostics"]["router"]["picked"] == "code_exp"

    def test_decide_reports_error_without_available_experts(self):
        r = ExpertRouter(ExpertRegistry(), client=FakeClient([("A", -0.1)]))
        out = r.decide("状态", {"q": {"type": "noul", "instructions": "是否？"}})
        assert out["answers"] == {} and "error" in out

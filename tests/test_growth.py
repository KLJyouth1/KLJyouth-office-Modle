"""成长层回归测试：在线增量学习 + 主动求知决策（不依赖 GPU/网络）。

锁定本轮最关键的三条设计底线：
1. **只学成功经验**：失败轨迹若要被内化，等于把错误行为写进权重。
2. **必须混入回放样本**：没有回放的增量微调会灾难性遗忘。
3. **增量版本不覆盖历史**：必须另存版本目录，发现退化可回退。
4. **求知有副作用但不假装成功**：取数失败要如实报告，并退回已有答案。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omnipilot.decision.active import ActiveDecider  # noqa: E402
from omnipilot.knowledge.store import FACT, KnowledgeItem, KnowledgeStore  # noqa: E402
from omnipilot.train import online as OL  # noqa: E402


# ------------------------------------------------------------------ 替身
class _ScriptedEngine:
    """按调用次数返回不同诊断，用来模拟"直接问不准、查完变准"。"""

    model = "fake"

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def decide(self, state, questions):
        idx = min(self.calls, len(self.script) - 1)
        letter_ok, conf = self.script[idx]
        self.calls += 1
        return {
            "answers": {"q": {"type": "choice", "choice": "a",
                              "probabilities": {"a": conf, "b": 1 - conf}}},
            "diagnostics": {"q": {"letter_ok": letter_ok, "sampled": "A" if letter_ok else "x"}},
        }


class TestEpisodeToSamples:
    def test_rejects_failed_episodes(self):
        # 底线 1：失败轨迹绝不能被学进权重。
        assert OL.episode_to_samples({"task": "x", "ok": False, "domains": ["code"]}) == []

    def test_emits_routing_sample_for_success(self):
        rows = OL.episode_to_samples({"task": "写入要点文件", "ok": True,
                                      "domains": ["code"], "steps": []})
        assert len(rows) == 1
        assert rows[0]["messages"][-1]["content"] == "code"
        assert rows[0]["meta"]["kind"] == "routing"

    def test_emits_tool_choice_only_for_successful_steps(self):
        ep = {"task": "t", "ok": True, "domains": [],
              "steps": [{"tool": "file_write", "ok": True},
                        {"tool": "bad_tool", "ok": False}]}
        rows = OL.episode_to_samples(ep)
        tools = [r["messages"][-1]["content"] for r in rows]
        assert tools == ["file_write"]

    def test_respects_max_samples(self):
        ep = {"task": "t", "ok": True, "domains": ["code"],
              "steps": [{"tool": f"t{i}", "ok": True} for i in range(10)]}
        assert len(OL.episode_to_samples(ep, max_samples=3)) == 3

    def test_empty_task_yields_nothing(self):
        assert OL.episode_to_samples({"task": "  ", "ok": True, "domains": ["code"]}) == []


class TestExperienceBuffer:
    def test_add_load_clear_roundtrip(self, tmp_path):
        buf = OL.ExperienceBuffer(tmp_path / "p.jsonl")
        assert buf.add([{"messages": [{"role": "user", "content": "a"}]}]) == 1
        assert buf.count() == 1
        buf.clear()
        assert buf.count() == 0

    def test_add_empty_is_noop(self, tmp_path):
        assert OL.ExperienceBuffer(tmp_path / "p.jsonl").add([]) == 0


class TestBuildOnlineDataset:
    def test_mixes_new_with_replay_to_prevent_forgetting(self, tmp_path, monkeypatch):
        # 底线 2：回放必须是真实混入的，否则增量训练会冲掉旧能力。
        ds = tmp_path / "datasets"
        ds.mkdir()
        (ds / "omnipilot16_train.jsonl").write_text(
            "\n".join('{"messages":[{"role":"user","content":"old%d"},'
                      '{"role":"assistant","content":"x"}]}' % i for i in range(10)),
            encoding="utf-8")
        pending = tmp_path / "pending.jsonl"
        pending.write_text(
            "\n".join('{"messages":[{"role":"user","content":"new%d"},'
                      '{"role":"assistant","content":"y"}]}' % i for i in range(4)),
            encoding="utf-8")
        monkeypatch.setattr(OL, "DATASET_DIR", ds)
        info = OL.build_online_dataset("omnipilot16", pending, replay_ratio=1.0)
        assert info["new_samples"] == 4
        assert info["replay_samples"] == 4
        assert info["train"] + info["val"] == 8

    def test_dedups_identical_samples(self, tmp_path, monkeypatch):
        ds = tmp_path / "datasets"
        ds.mkdir()
        pending = tmp_path / "p.jsonl"
        row = '{"messages":[{"role":"user","content":"same"},{"role":"assistant","content":"z"}]}'
        pending.write_text("\n".join([row, row]), encoding="utf-8")
        monkeypatch.setattr(OL, "DATASET_DIR", ds)
        info = OL.build_online_dataset("omnipilot16", pending)
        assert info["new_samples"] == 1


class TestRollback:
    def test_rollback_removes_latest_version(self, tmp_path, monkeypatch):
        # 底线 3：版本可回退。
        monkeypatch.setattr(OL, "ADAPTER_DIR", tmp_path)
        (tmp_path / "m_v1").mkdir()
        (tmp_path / "m_v2").mkdir()
        assert OL.rollback("m")["ok"] is True
        assert not (tmp_path / "m_v2").exists()
        assert (tmp_path / "m_v1").exists()

    def test_rollback_reports_when_no_versions(self, tmp_path, monkeypatch):
        monkeypatch.setattr(OL, "ADAPTER_DIR", tmp_path)
        assert OL.rollback("nope")["ok"] is False


class TestActiveDecider:
    def _questions(self):
        return {"q": {"type": "choice", "instructions": "选", "criteria": {"a": "", "b": ""}}}

    def test_accepts_confident_first_pass_without_any_lookup(self, tmp_path):
        eng = _ScriptedEngine([(True, 0.95)])
        store = KnowledgeStore(tmp_path / "k.jsonl")
        r = ActiveDecider(eng, store).decide("状态", self._questions())
        assert r["reliable"] is True and r["acquired"] is False
        assert [a["stage"] for a in r["attempts"]] == ["direct"]

    def test_uses_memory_before_going_online(self, tmp_path):
        store = KnowledgeStore(tmp_path / "k.jsonl")
        store.add(KnowledgeItem(kind=FACT, topic="状态", content="相关的既有知识，可直接复用",
                                source="manual", evidence="手工整理"))
        eng = _ScriptedEngine([(True, 0.20), (True, 0.90)])
        r = ActiveDecider(eng, store).decide("状态", self._questions())
        stages = [a["stage"] for a in r["attempts"]]
        assert "memory" in stages and r["reliable"] is True
        assert r["acquired"] is False          # 记忆够用就不必联网

    def test_acquires_online_when_memory_is_not_enough(self, tmp_path):
        store = KnowledgeStore(tmp_path / "k.jsonl")
        # 脚本按调用次序生效：第 1 次(直接决策)不可信，第 2 次(求知后)变可信。
        # 注意这里只有两次探测——记忆库初始为空，不会触发 memory 阶段。
        eng = _ScriptedEngine([(True, 0.10), (True, 0.88)])
        r = ActiveDecider(
            eng, store,
            fetch=lambda q: "查到的可靠外部信息，长度足够通过入库闸门，可被后续复用。",
        ).decide("状态", self._questions(), topic="状态")
        stages = [a["stage"] for a in r["attempts"]]
        assert "acquire" in stages and "after_acquire" in stages
        assert r["acquired"] is True and r["reliable"] is True
        assert store.size() == 1               # 新知识确实被记住了

    def test_reports_honestly_when_fetch_fails(self, tmp_path):
        store = KnowledgeStore(tmp_path / "k.jsonl")
        eng = _ScriptedEngine([(True, 0.10)])

        def boom(_q):
            raise RuntimeError("断网")

        r = ActiveDecider(eng, store, fetch=boom).decide("状态", self._questions())
        acc = [a for a in r["attempts"] if a["stage"] == "acquire"][0]
        assert acc["ok"] is False and r["acquired"] is False
        assert store.size() == 0

    def test_stays_unreliable_when_model_is_not_letter_faithful(self, tmp_path):
        store = KnowledgeStore(tmp_path / "k.jsonl")
        eng = _ScriptedEngine([(False, 0.99)])
        r = ActiveDecider(eng, store).decide("状态", self._questions())
        assert r["reliable"] is False          # 高置信但非字母作答 → 依然不可信

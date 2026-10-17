"""知识层（外部记忆）回归测试 —— 不依赖网络与模型。

锁定本轮设计的三个关键约束：
1. **抗幻觉闸门**：无出处、无证据的数值断言一律不得入库。
   外部记忆若不加闸门，会变成"幻觉放大器"——把猜测存下来、下次当真读回。
2. **复用计数**：命中要累加 hits，这是"哪些知识值得内化进权重"的依据。
3. **可撤销**：forget 之后检索不到，且不破坏已有历史。
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from omnipilot.knowledge import FACT, RULE, SKILL  # noqa: E402
from omnipilot.knowledge.store import KnowledgeItem, KnowledgeStore, verify_item  # noqa: E402


@pytest.fixture()
def store(tmp_path) -> KnowledgeStore:
    return KnowledgeStore(tmp_path / "k.jsonl")


def item(**kw) -> KnowledgeItem:
    base = dict(kind=FACT, topic="Ollama 的 logprobs 用法", content="用 logprobs=true 取候选概率",
                source="https://example.com/doc", evidence="原始文档片段")
    base.update(kw)
    return KnowledgeItem(**base)


class TestVerifyGate:
    def test_accepts_a_well_sourced_item(self):
        ok, why = verify_item(item())
        assert ok is True and why == "通过"

    @pytest.mark.parametrize("kw,why", [
        ({"kind": "guess"}, "未知类型"),
        ({"topic": "  "}, "空 topic"),
        ({"content": ""}, "空 content"),
        ({"source": ""}, "无出处"),
    ])
    def test_rejects_unusable_items(self, kw, why):
        ok, _ = verify_item(item(**kw))
        assert ok is False

    def test_rejects_numeric_claim_without_evidence(self):
        # 这条最要紧：编造的数字只要没有 evidence，就绝不能进记忆。
        ok, why = verify_item(item(content="2026 年营收是 1234567 元", evidence=""))
        assert ok is False and "疑似编造" in why

    def test_allows_numeric_claim_with_evidence(self):
        ok, _ = verify_item(item(content="2026 年营收是 1234567 元", evidence="财报第 3 页"))
        assert ok is True


class TestStore:
    def test_add_then_search_finds_it(self, store):
        assert store.add(item())[0] is True
        hits = store.search("logprobs 候选概率")
        assert len(hits) == 1 and hits[0].topic.startswith("Ollama")

    def test_gate_blocks_rejected_items_from_storage(self, store):
        ok, why = store.add(item(source=""))
        assert ok is False and store.size() == 0

    def test_force_bypasses_gate_for_tests_only(self, store):
        assert store.add(item(source=""), force=True)[0] is True
        assert store.size() == 1

    def test_duplicate_add_counts_hits_instead_of_duplicating(self, store):
        store.add(item())
        store.add(item())
        assert store.size() == 1
        assert store.all()[0].hits == 1

    def test_persists_across_instances(self, tmp_path):
        p = tmp_path / "k.jsonl"
        KnowledgeStore(p).add(item())
        again = KnowledgeStore(p)
        assert again.size() == 1

    def test_search_returns_empty_for_unrelated_query(self, store):
        store.add(item())
        assert store.search("完全无关的足球比赛结果") == []

    def test_forget_removes_from_search(self, store):
        store.add(item())
        key = store.all()[0].key
        assert store.forget(key) is True
        assert store.search("logprobs") == []

    def test_forget_unknown_key_returns_false(self, store):
        assert store.forget("nope") is False

    def test_skill_and_rule_are_valid_kinds(self, store):
        assert store.add(item(kind=SKILL, topic="写要点文件", content="用 file_write 逐条写入"))[0]
        assert store.add(item(kind=RULE, topic="路径唯一", content="任务给定路径时必须逐字一致"))[0]
        assert store.size() == 2


class TestAcquisitionAndPrompt:
    def test_acquire_stores_verified_web_content(self, store):
        ok, why = store.acquire("Python 版本", "Python 当前稳定版本是多少？",
                                lambda q: "Python 3.14 是当前稳定版本，发布于 2025 年。",
                                source="https://python.org")
        assert ok is True and store.size() == 1

    def test_acquire_rejects_too_short_content(self, store):
        ok, why = store.acquire("x", "q", lambda q: "短")
        assert ok is False and store.size() == 0

    def test_acquire_swallows_fetch_errors_instead_of_crashing(self, store):
        def boom(_q):
            raise RuntimeError("网络断了")

        ok, why = store.acquire("x", "q", boom)
        assert ok is False and "取数失败" in why

    def test_prompt_block_is_empty_when_nothing_matches(self, store):
        assert store.prompt_block("无关查询") == ""

    def test_prompt_block_renders_and_counts_usage(self, store):
        store.add(item())
        block = store.prompt_block("logprobs 怎么用")
        assert "已知知识" in block and "logprobs" in block
        assert store.all()[0].hits == 1

    def test_prompt_block_respects_char_budget(self, store):
        for i in range(20):
            store.add(item(topic=f"主题{i}", content="内容" * 100))
        block = store.prompt_block("主题", k=20, max_chars=300)
        assert len(block) <= 300 + 60          # 允许标题行略超，但不得整段无节制拼接

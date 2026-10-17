"""知识层：经校验的外部记忆（对应大脑"海马体"式的快速学习）。

为什么必须存在（回答"人能上网搜、记下来、以后再用，为什么模型不行"）：
纯权重式学习有两个硬伤——
1. **灾难性遗忘**：新样本覆盖旧能力，学得越多旧技能掉得越快；
2. **昂贵**：每学一条新事实都要一次训练。
人类恰好相反：先把经验记成"笔记"（快、便宜、不遗忘、可回查），
再靠反复使用把它慢慢内化。本模块实现的就是这层"笔记"。

设计上刻意做成零算力、可审计、可回滚：
- 零算力：纯文本检索，CPU 微秒级，不碰显存；
- 可审计：每条都留 source/evidence，谁写进去的、依据是什么一目了然；
- 可回滚：append-only JSONL，删掉某行即可撤销一条被污染的知识。

与 train/online.py 的分工（互补学习系统）：
    快层 知识库（本模块，秒级生效、永不遗忘）
      ↓ 反复复用、命中足够多次
    慢层 LoRA 小步增量更新（分钟级，内化进权重）
这样既拿到"学了就会"的即时性，又保留"内化"的长期收益，
而且外部记忆天然规避了灾难性遗忘。
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from ..config import DATA_DIR

STORE_PATH = DATA_DIR / "knowledge.jsonl"

# 类型：fact=事实断言，skill=可复用做法，rule=约束/偏好
FACT, SKILL, RULE = "fact", "skill", "rule"
_KINDS = (FACT, SKILL, RULE)

# 需要来源背书的数值形态（与 eval/hallucination.py 的口径保持一致）。
_NUM_RE = re.compile(r"\d{3,}")


@dataclass
class KnowledgeItem:
    """一条知识。`verified` 为真表示它通过了抗幻觉闸门。"""

    kind: str
    topic: str                     # 检索用的主题键（一句话）
    content: str                   # 知识正文
    source: str = ""               # 出处：URL / 工具名 / 老师模型名
    evidence: str = ""             # 原始依据（片段），便于复核与审计
    confidence: float = 0.5
    verified: bool = False
    hits: int = 0                  # 被复用次数（用于决定是否值得内化进权重）
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    @property
    def key(self) -> str:
        """去重键：归一化 topic + content 的哈希。"""
        blob = f"{_norm(self.topic)}|{_norm(self.content)}"
        return hashlib.md5(blob.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict:
        d = asdict(self)
        d["key"] = self.key
        return d


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def verify_item(item: KnowledgeItem) -> tuple[bool, str]:
    """抗幻觉闸门：不合格的知识一律不许入库。

    这是本模块最重要的函数。外部记忆若不加闸门，就会变成"幻觉放大器"——
    模型把自己的猜测写进库、下次再读回来当真，越用越错且无法察觉。

    判据（与 train/distill.py 的 verify_teacher_action 同一思路）：
    1. topic/content 非空（不许存空壳）；
    2. 必须有 source（无出处的知识不可信）；
    3. 含多位数值的断言必须有 evidence（数值幻觉的最大来源）。
    """
    if item.kind not in _KINDS:
        return False, f"未知类型：{item.kind}"
    if not _norm(item.topic) or not _norm(item.content):
        return False, "topic/content 不能为空"
    if not _norm(item.source):
        return False, "缺少出处（source），无出处的知识不可信"
    if _NUM_RE.search(item.content) and not _norm(item.evidence):
        return False, "断言含具体数值但无 evidence，疑似编造"
    return True, "通过"


class KnowledgeStore:
    """append-only 知识库：写入即生效，检索零算力。"""

    def __init__(self, path: Path | str = STORE_PATH) -> None:
        self.path = Path(path)
        self._items: dict[str, KnowledgeItem] = {}
        self._loaded = False

    # ------------------------------------------------------------------ 读写
    def load(self) -> None:
        self._items = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                d.pop("key", None)
                try:
                    it = KnowledgeItem(**d)
                except TypeError:
                    continue          # 结构不符的行跳过，不让坏数据阻断整个库
                self._items[it.key] = it
        self._loaded = True

    def _ensure(self) -> None:
        if not self._loaded:
            self.load()

    def add(self, item: KnowledgeItem, *, force: bool = False) -> tuple[bool, str]:
        """入库。默认必须过闸门；已存在的同键条目只累加 hits 与更新 confidence。"""
        self._ensure()
        if not force:
            ok, why = verify_item(item)
            if not ok:
                return False, why
            item.verified = True
        old = self._items.get(item.key)
        if old is not None:
            old.hits += 1
            old.confidence = max(old.confidence, item.confidence)
            old.updated = time.time()
            self._append(old)
            return True, "已存在，累加复用计数"
        self._items[item.key] = item
        self._append(item)
        return True, "已入库"

    def _append(self, item: KnowledgeItem) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item.as_dict(), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ 检索
    def all(self) -> list[KnowledgeItem]:
        self._ensure()
        return list(self._items.values())

    def size(self) -> int:
        self._ensure()
        return len(self._items)

    def search(self, query: str, k: int = 5) -> list[KnowledgeItem]:
        """关键词检索。零算力、确定性、可解释（命中了哪些词是可以复现的）。"""
        self._ensure()
        keys = [t for t in re.split(r"[\s,，。;；、]+", query or "") if len(t) >= 2]
        if not keys:
            return self.all()[-k:]
        scored: list[tuple[float, KnowledgeItem]] = []
        for it in self._items.values():
            text = _norm(f"{it.topic} {it.content}")
            hit = sum(text.count(_norm(t)) for t in keys)
            if hit:
                # 置信度与复用次数作为次级权重：既相关又更可信的排前面
                scored.append((hit * (0.5 + it.confidence) + 0.05 * it.hits, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:k]]

    def mark_used(self, items: Iterable[KnowledgeItem]) -> None:
        """记录一次复用（用于评估"哪些知识值得内化进权重"）。"""
        self._ensure()
        for it in items:
            cur = self._items.get(it.key)
            if cur is not None:
                cur.hits += 1
                cur.updated = time.time()
                self._append(cur)

    def forget(self, key: str) -> bool:
        """撤销一条知识（append-only 文件里追加一条删除记录，不破坏历史）。"""
        self._ensure()
        if key not in self._items:
            return False
        del self._items[key]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "_deleted": True}, ensure_ascii=False) + "\n")
        return True

    # ------------------------------------------------------------------ 主动求知
    def acquire(self, topic: str, question: str, fetch: Callable[[str], str],
                *, kind: str = FACT, source: str = "web", min_chars: int = 20) -> tuple[bool, str]:
        """主动求知：查一次 → 过闸门 → 入库。对应人的"不知道就去查，记下来"。

        `fetch` 是注入的取数函数（真实使用时可传浏览器/搜索工具），
        注入式设计让它可在离线环境下用替身测试，不把网络依赖焊死在逻辑里。
        """
        try:
            raw = fetch(question) or ""
        except Exception as e:  # 取数失败不应让整个求知流程崩掉
            return False, f"取数失败：{type(e).__name__}: {e}"
        if len(raw.strip()) < min_chars:
            return False, "取回内容过短，不足以形成可信知识"
        item = KnowledgeItem(kind=kind, topic=topic, content=raw.strip()[:2000],
                             source=source, evidence=raw.strip()[:400],
                             confidence=0.6)
        return self.add(item)

    def prompt_block(self, query: str, k: int = 5, *, max_chars: int = 1200) -> str:
        """把命中的知识渲染成可注入决策提示的一段文本（空则返回空串）。"""
        hits = self.search(query, k)
        if not hits:
            return ""
        lines = []
        used = 0
        for it in hits:
            line = f"- [{it.kind}] {it.topic}：{it.content[:200]}（来源：{it.source}）"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        if not lines:
            return ""
        self.mark_used(hits)
        return "已知知识（来自历史积累，可直接复用）：\n" + "\n".join(lines)

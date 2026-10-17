"""记忆层。

三层、零外部依赖、全部本地落盘：
- 工作记忆(working)：当前任务的执行轨迹，随任务结束清空。
- 情景记忆(episodic)：历史任务与结果，JSONL 追加，支持关键词检索复用经验。
- 画像记忆(profile)：用户偏好/长期约束，跨任务常驻。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import DATA_DIR

EPISODIC_PATH = DATA_DIR / "episodic.jsonl"
PROFILE_PATH = DATA_DIR / "profile.json"


@dataclass
class Step:
    n: int
    tool: str
    args: dict
    ok: bool
    output: str
    elapsed: float = 0.0


@dataclass
class Episode:
    task: str
    domains: list[str]
    steps: list[Step] = field(default_factory=list)
    result: str = ""
    ok: bool = False
    ts: float = field(default_factory=time.time)


class Memory:
    def __init__(self, episodic_path: Path = EPISODIC_PATH, profile_path: Path = PROFILE_PATH) -> None:
        self.episodic_path = Path(episodic_path)
        self.profile_path = Path(profile_path)

    # -------------------------------------------------------------- 工作记忆
    # (由 Agent 循环在内存中维护，这里不持久化)

    # -------------------------------------------------------------- 情景记忆
    def record(self, episode: Episode) -> None:
        self.episodic_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(episode)
        with self.episodic_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _iter_episodes(self):
        if not self.episodic_path.exists():
            return
        with self.episodic_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def recent(self, n: int = 5) -> list[dict]:
        items = list(self._iter_episodes())
        return items[-n:]

    def recall(self, query: str, n: int = 3) -> list[dict]:
        """关键词检索历史经验（轻量召回，无需向量库）。"""
        keys = [k for k in query.replace("，", " ").replace(",", " ").split() if len(k) >= 2]
        if not keys:
            return self.recent(n)
        scored: list[tuple[int, dict]] = []
        for ep in self._iter_episodes():
            text = ep.get("task", "") + " " + ep.get("result", "")
            score = sum(text.count(k) for k in keys)
            if score:
                scored.append((score, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:n]]

    # -------------------------------------------------------------- 画像记忆
    def load_profile(self) -> dict:
        if self.profile_path.exists():
            try:
                return json.loads(self.profile_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def set_profile(self, **kv) -> None:
        prof = self.load_profile()
        prof.update(kv)
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile_path.write_text(json.dumps(prof, ensure_ascii=False, indent=2), encoding="utf-8")

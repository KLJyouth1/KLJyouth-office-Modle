"""定向训练 · 示范采集。

两种数据来源（对应"公开数据 + 你的示范"结合策略里"你的示范"这一半）：
1. 自演采集(Self-Play)：智能体每成功跑通一步，就自动把
   「当时给模型的完整上下文 prompt」→「模型选择的动作 JSON」
   沉淀为一条 SFT 样本。跑得越多，专属模型越贴合这台机器与你的习惯。
2. 人工示范(Manual)：用 record() 手动登记一条 观察→动作。

输出为标准 chat 格式 JSONL，可直接被 dataset.py / PEFT / LLaMA-Factory 消费。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ..config import DEMO_DIR

DEFAULT_PATH = DEMO_DIR / "sft.jsonl"


class SFTCollector:
    def __init__(self, path: Path | str = DEFAULT_PATH, *, system: str | None = None,
                 only_success: bool = True, dedupe: bool = True) -> None:
        self.path = Path(path)
        self.system = system or "你是一个运行在 Windows 上的桌面智能体。"
        self.only_success = only_success
        self.dedupe = dedupe
        self._seen: set[str] = set()
        if self.dedupe and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._seen.add(hashlib.md5(line.encode("utf-8")).hexdigest())

    # ------------------------------------------------------------------ 内部
    def _append(self, record: dict) -> bool:
        blob = json.dumps(record, ensure_ascii=False)
        h = hashlib.md5(blob.encode("utf-8")).hexdigest()
        if self.dedupe and h in self._seen:
            return False
        self._seen.add(h)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(blob + "\n")
        return True

    # ------------------------------------------------------------------ 写入
    def add_step(self, decision: dict | None, step=None) -> bool:
        """由 Agent 循环回调：记录一条 (prompt -> action) 样本。"""
        if not decision:
            return False
        prompt = decision.get("prompt")
        action = decision.get("action")
        if not prompt or not action:
            return False
        is_final = "final" in action
        if step is not None and not step.ok and self.only_success:
            return False
        record = {
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(action, ensure_ascii=False)},
            ],
            "meta": {
                "tool": action.get("tool", "(final)" if is_final else ""),
                "domain": getattr(step, "domain", ""),
                "ok": True if (step is None or step.ok) else False,
                "source": "self-play",
                "ts": time.time(),
            },
        }
        return self._append(record)

    def record(self, observation: str, action: dict, *, source: str = "manual") -> bool:
        """人工登记一条样本：observation 为给模型的上下文，action 为期望动作。"""
        record = {
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": observation},
                {"role": "assistant", "content": json.dumps(action, ensure_ascii=False)},
            ],
            "meta": {"source": source, "ts": time.time()},
        }
        return self._append(record)

    # ------------------------------------------------------------------ 查询
    def count(self) -> int:
        if not self.path.exists():
            return 0
        return sum(1 for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip())

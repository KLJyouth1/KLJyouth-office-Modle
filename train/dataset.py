"""定向训练 · 数据集构建。

把多来源的 chat 样本（自演采集 / 人工示范 / 公开数据集转换产物）合并、
校验、去重、切分为 train/val，产出可直接训练的 JSONL。
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from ..config import DATASET_DIR


def load_jsonl(path: Path | str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def valid_sample(rec: dict) -> bool:
    """校验是否为合法 chat 样本：至少要有 user + assistant 两条。

    system 是可选的——普通对话样本可能不带 system（由模型自带默认模板补齐），
    早期版本强制要求 system 会把这类样本误丢（实测踩到）。
    """
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return False
    roles = [m.get("role") for m in msgs if isinstance(m, dict)]
    if not {"user", "assistant"}.issubset(set(roles)):
        return False
    return all(isinstance(m.get("content"), str) and m["content"].strip() for m in msgs)


def build_dataset(
    sources: list[Path | str],
    *,
    out_dir: Path | str = DATASET_DIR,
    val_ratio: float = 0.1,
    seed: int = 42,
    name: str = "omnipilot_sft",
) -> dict:
    """合并多来源样本，去重切分，写出 {train,val}.jsonl 与 dataset_info.json。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw: list[dict] = []
    for src in sources:
        raw.extend(load_jsonl(src))

    seen: set[str] = set()
    kept: list[dict] = []
    dropped = 0
    for rec in raw:
        if not valid_sample(rec):
            dropped += 1
            continue
        key = hashlib.md5(
            json.dumps(rec["messages"], ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(rec)

    rnd = random.Random(seed)
    rnd.shuffle(kept)
    n_val = max(1, int(len(kept) * val_ratio)) if len(kept) > 5 else 0
    val = kept[:n_val]
    train = kept[n_val:]

    train_path = out / f"{name}_train.jsonl"
    val_path = out / f"{name}_val.jsonl"
    _write(train_path, train)
    _write(val_path, val)

    info = {
        "name": name,
        "total": len(kept),
        "train": len(train),
        "val": len(val),
        "dropped": dropped,
        "sources": [str(s) for s in sources],
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    (out / f"{name}_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

"""在线增量学习（互补学习系统的"慢层"）：把反复用到的经验内化进权重。

为什么需要它（配合 knowledge/store.py 的"快层"）：
快层（知识库）秒级生效、永不遗忘，但它只是"笔记"，不改变模型本身。
人不是只记笔记——反复用到的东西会逐渐变成本能。慢层做的就是这个内化：
把**已成功、已校验**的经验攒成小批，做几步 LoRA 增量更新，写回适配器。

三个必须守住的工程底线（否则"持续学习"会变成"持续退化"）：
1. **只学成功且可核验的经验**：失败轨迹会把错误行为内化，绝不可入库。
2. **必须混入回放样本**：只用新样本微调会灾难性遗忘，把之前学的任务全冲掉。
   因此每次更新都从原始训练集抽一批作为 replay 一起训。
3. **小步、可回滚**：每次只走极少步数并另存版本目录（不是原地覆盖），
   发现退化可立刻回退到上一版。
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from ..config import ADAPTER_DIR, DATASET_DIR

# 待学习经验的暂存区（攒够再训，避免每条经验都触发一次训练）
PENDING_PATH = DATASET_DIR / "online_pending.jsonl"


def episode_to_samples(episode: dict, *, max_samples: int = 3) -> list[dict]:
    """把一次成功轨迹转成"决策格式"的训练样本（任务 -> 领域 / 工具）。

    为什么复用决策格式：在线学习要提升的正是决策质量（选对域、选对工具），
    而不是让模型重新学聊天。这样每条样本都直接对齐 decision 层的能力目标。

    只接受 ok=True 的轨迹；失败轨迹一律不产出样本（见模块 docstring 第 1 条）。
    """
    if not episode.get("ok"):
        return []
    task = str(episode.get("task") or "").strip()
    if not task:
        return []
    domains = [str(d) for d in (episode.get("domains") or []) if d]
    steps = episode.get("steps") or []
    out: list[dict] = []
    if domains:
        out.append({"messages": [
            {"role": "user", "content": f"任务：{task}\n这条任务属于哪些领域？只回答领域名。"},
            {"role": "assistant", "content": ",".join(domains)},
        ], "meta": {"source": "online", "kind": "routing", "task": task}})
    # 成功用过的工具也是"对的答案"
    for st in steps:
        if not st.get("ok"):
            continue
        tool = str(st.get("tool") or "").strip()
        if not tool:
            continue
        out.append({"messages": [
            {"role": "user", "content": f"任务：{task}\n下一步应该调用哪个工具？只回答工具名。"},
            {"role": "assistant", "content": tool},
        ], "meta": {"source": "online", "kind": "tool_choice", "tool": tool}})
        if len(out) >= max_samples:
            break
    return out[:max_samples]


class ExperienceBuffer:
    """待学习经验缓冲：追加即落盘，训练时消费。"""

    def __init__(self, path: Path | str = PENDING_PATH) -> None:
        self.path = Path(path)

    def add(self, samples: list[dict]) -> int:
        if not samples:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        return len(samples)

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def count(self) -> int:
        return len(self.load())

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


def _dedup(samples: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for s in samples:
        key = json.dumps(s.get("messages"), ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def build_online_dataset(base_name: str = "omnipilot16",
                         pending: Path | str = PENDING_PATH,
                         *, replay_ratio: float = 1.0, seed: int = 7,
                         out_name: str = "omnipilot16_online") -> dict:
    """把"新经验 + 回放样本"混合成一个可训练的迷你数据集。

    replay 来自原始训练集。**没有回放就会灾难性遗忘**，这是本模块最关键的一步，
    因此回放比例默认 1.0（新旧各半），而不是象征性地放几条。
    """
    buf = ExperienceBuffer(pending)
    new = _dedup(buf.load())
    old = []
    src = Path(DATASET_DIR) / f"{base_name}_train.jsonl"
    if src.exists():
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    old.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    rnd = random.Random(seed)
    rnd.shuffle(old)
    n_replay = min(len(old), max(0, int(len(new) * replay_ratio)))
    mixed = _dedup(new + old[:n_replay])
    rnd.shuffle(mixed)

    out_dir = Path(DATASET_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_val = max(1, int(len(mixed) * 0.1)) if len(mixed) > 5 else 0
    val, train = mixed[:n_val], mixed[n_val:]
    for name, rows in ((f"{out_name}_train.jsonl", train), (f"{out_name}_val.jsonl", val)):
        with (out_dir / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    info = {"new_samples": len(new), "replay_samples": n_replay,
            "train": len(train), "val": len(val), "out_name": out_name}
    (out_dir / f"{out_name}_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def _step_dir(adapter: Path, version: int) -> Path:
    return adapter.parent / f"{adapter.name}_v{version}"


def online_update(base_model: str, prev_adapter: Path | str, *, steps: int = 4,
                  lr: float = 1e-4, name: str = "omnipilot16",
                  use_4bit: bool = True, max_len: int = 384, seed: int = 7,
                  consume: bool = True, verbose: bool = True) -> dict:
    """从上一版适配器继续训练极少步数，产出**新版本目录**（不原地覆盖）。

    为什么从 prev_adapter 继续而不是从基座重训：这样才能"在现有能力上增量学习"；
    从基座重训等于全新训练，既慢又会丢掉之前所有积累。

    返回统计（含版本目录），失败时 ok=False 并给出原因，不假装成功。
    """
    info = build_online_dataset(name, out_name=f"{name}_online")
    if info["new_samples"] == 0:
        return {"ok": False, "message": "没有待学习的新经验（online_pending.jsonl 为空）",
                "info": info}

    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    from .qlora import _encode, _read_jsonl

    ds_dir = Path(DATASET_DIR)
    train_rows = _read_jsonl(ds_dir / f"{name}_online_train.jsonl")
    prev = Path(prev_adapter)
    if not prev.exists():
        return {"ok": False, "message": f"上一版适配器不存在：{prev}"}

    # 版本目录：v1, v2, ... 递增，绝不覆盖历史（可回滚是底线之一）
    version = 1
    while _step_dir(Path(ADAPTER_DIR) / name, version).exists():
        version += 1
    out = _step_dir(Path(ADAPTER_DIR) / name, version)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(prev), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {"trust_remote_code": True}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        load_kwargs["device_map"] = {"": 0}
    try:
        base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    except Exception as e:
        return {"ok": False, "message": f"加载基座失败：{type(e).__name__}: {str(e)[:200]}"}

    model = PeftModel.from_pretrained(base, str(prev), is_trainable=True)
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    class _DS(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.items = []
            for r in rows:
                try:
                    self.items.append(_encode(tokenizer, r["messages"], max_len))
                except Exception:
                    continue

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    def _collate(batch):
        mx = max(len(b["input_ids"]) for b in batch)
        pad = tokenizer.pad_token_id
        ids, att, lab = [], [], []
        for b in batch:
            n = mx - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad] * n)
            att.append(b["attention_mask"] + [0] * n)
            lab.append(b["labels"] + [-100] * n)
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(att, dtype=torch.long),
                "labels": torch.tensor(lab, dtype=torch.long)}

    args = TrainingArguments(
        output_dir=str(out / "ckpt"),
        per_device_train_batch_size=1, gradient_accumulation_steps=1,
        max_steps=max(1, steps), learning_rate=lr, logging_steps=1,
        save_strategy="no", report_to=[], remove_unused_columns=False,
        dataloader_num_workers=0, optim="adamw_torch",
        fp16=torch.cuda.is_available(), seed=seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = Trainer(model=model, args=args, train_dataset=_DS(train_rows),
                      data_collator=_collate)
    try:
        result = trainer.train()
    except RuntimeError as e:
        return {"ok": False, "message": f"增量训练失败（显存不足？）：{str(e)[:200]}",
                "hint": "先 `ollama stop <模型>` 释放显存后重试"}

    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    stats = {"ok": True, "prev_adapter": str(prev), "new_adapter": str(out),
             "version": version, "steps": max(1, steps),
             "train_loss": float(result.training_loss) if result and result.training_loss else None,
             "new_samples": info["new_samples"], "replay_samples": info["replay_samples"]}
    (out / "online_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    if consume:
        ExperienceBuffer().clear()      # 已内化进权重，清空待学缓冲
    if verbose:
        print(f"[online] 已产出新版本适配器：{out}（步数={max(1, steps)}）")
    return stats


def rollback(name: str, version: int | None = None) -> dict:
    """回滚：删除指定版本目录（若不给版本则删除最新一版）。用于发现退化时撤销。"""
    root = Path(ADAPTER_DIR)
    versions = sorted(
        [d for d in root.glob(f"{name}_v*") if d.is_dir()],
        key=lambda p: int(p.name.rsplit("_v", 1)[-1]) if p.name.rsplit("_v", 1)[-1].isdigit() else 0,
    )
    if not versions:
        return {"ok": False, "message": f"没有找到 {name} 的增量版本"}
    target = root / f"{name}_v{version}" if version else versions[-1]
    if not target.exists():
        return {"ok": False, "message": f"版本不存在：{target.name}"}
    shutil.rmtree(target)
    return {"ok": True, "removed": str(target)}

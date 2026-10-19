"""定向训练 · LoRA / QLoRA 微调。

在低配机器上的取舍：
- 本机 PyTorch 为 CPU 版、且未装 bitsandbytes，因此默认走 **普通 LoRA（非量化）**；
  若要 4-bit QLoRA，请安装 CUDA 版 torch 与 bitsandbytes，并把 use_4bit=True。
- 只训练极小的低秩适配器参数（默认 r=16），显存/内存占用远小于全量微调。
产物：adapter 目录（含 adapter_config.json / adapter_model.safetensors），
交给 train/export.py 即可注册为 Ollama 模型热插拔使用。
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import ADAPTER_DIR, DATASET_DIR


def _render(tokenizer, messages: list[dict], *, generate_prompt: bool = False) -> str:
    """用模型自带的 chat 模板渲染对话（无模板则退化为简单拼接）。

    `generate_prompt=True` 渲染"待续写"的提示（末尾保留助手头 `<|im_start|>assistant\\n`），
    供 `_encode` 计算标签掩码边界用；`False` 渲染完整对话（含助手回复）。
    """
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=generate_prompt
            )
        except Exception:
            pass
    text = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)
    return text + ("<|assistant|>\n" if generate_prompt else "")


def _encode(tokenizer, messages: list[dict], max_len: int) -> dict:
    """编码样本，并把 prompt 部分的 label 置 -100（只学助手回复）。

    前缀必须用 `generate_prompt=True` 渲染（**这是一个已实测的真实缺陷的修复**）：
    此前用 `add_generation_prompt=False`，前缀止于用户轮的 `<|im_end|>`，于是标签区间是
    `'<|im_start|>assistant\\nA<|im_end|>\\n'` —— **第一个被学习的 token 是助手头而不是字母**。
    而推理端（Ollama 的 Go 模板）已经把 `<|im_start|>assistant\\n` 预填在提示尾部，
    只读首个生成 token 的 logprob 当字母概率。训练/推理的"回答起点"因此错开一个助手头，
    字母概率随之失真 —— 这正是"letter_ok=1.0 但多选任务普跌甚至低于随机"的根因。
    """
    full = _render(tokenizer, messages)
    prefix = _render(tokenizer, messages[:-1], generate_prompt=True) if len(messages) > 1 else ""
    ids = tokenizer(full, truncation=True, max_length=max_len, add_special_tokens=False)["input_ids"]
    pids = tokenizer(prefix, truncation=True, max_length=max_len, add_special_tokens=False)["input_ids"]
    labels = list(ids)
    for i in range(min(len(pids), len(ids))):
        labels[i] = -100  # 前缀 prompt 不参与损失
    if all(l == -100 for l in labels) and labels:  # 兜底：避免整条被掩掉
        labels[-1] = ids[-1]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def train(
    base_model: str,
    dataset_dir: str | Path = DATASET_DIR,
    *,
    name: str = "omnipilot_sft",
    out_dir: str | Path | None = None,
    epochs: int = 3,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_len: int = 1536,
    batch_size: int = 1,
    grad_accum: int = 8,
    use_4bit: bool = False,
    gradient_checkpointing: bool = False,
    limit: int | None = None,
    resume_from: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """执行 LoRA 微调，返回训练统计。"""
    import os

    # 显存分配策略：
    #  - 默认用系统默认分配器。实测：
    #      expandable_segments:True → 几十步后触发 CUDACachingAllocator.cpp 的
    #        `!handles_.at(i)` 内部断言（碎片相关，且为已知 expandable 特性缺陷）；
    #      backend:cudaMallocAsync   → 不崩，但慢约 2.3 倍（7.7 s/it vs 3.1 s/it）；
    #      默认分配器                → 3.1 s/it 且稳定，故采用默认。
    #  - 可用 OMNIPILOT_CUDA_ALLOC_CONF 覆盖（如回退 expandable_segments）。
    _alloc = os.environ.get("OMNIPILOT_CUDA_ALLOC_CONF")
    if _alloc:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _alloc)

    import torch
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    ds_dir = Path(dataset_dir)
    train_rows = _read_jsonl(ds_dir / f"{name}_train.jsonl")
    val_rows = _read_jsonl(ds_dir / f"{name}_val.jsonl")
    if limit:
        train_rows, val_rows = train_rows[:limit], val_rows[: limit // 10 or 1]
    if not train_rows:
        return {"ok": False, "message": f"找不到训练样本：{ds_dir / f'{name}_train.jsonl'}"}

    out = Path(out_dir) if out_dir else (ADAPTER_DIR / name)
    out.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[train] 基座={base_model} 训练样本={len(train_rows)} 验证={len(val_rows)}")
        print(f"[train] 4bit={use_4bit} | CUDA={torch.cuda.is_available()} | 输出={out}")
    # 训练前显存预检：实测过"Ollama 老师模型仍驻留显存 → 训练 CUDA OOM"的事故。
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            free_b, total_b = torch.cuda.mem_get_info()
            free_gb, total_gb = free_b / 1e9, total_b / 1e9
            if verbose:
                print(f"[train] 显存 空闲={free_gb:.2f}GB / 总={total_gb:.2f}GB")
            if free_gb < total_gb * 0.55:
                print(f"[train] 警告：可用显存偏低（{free_gb:.2f}GB）。若 OOM，请先 `ollama stop <模型>` 释放显存。")
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"trust_remote_code": True}
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
            )
            # 关键：不要用 device_map="auto"。实测在 4GB 卡上 auto 会把大部分层卸载到 CPU，
            # 每步都要走 PCIe，1.5B 直接掉到 20~47 s/step（慢 10 倍以上）。
            # 1.5B 4-bit 权重仅约 1.3GB，可以整块放进 GPU（实测加载后仍剩 ~2.2GB）。
            # 真放不下时在下面回退到 "auto"。
            load_kwargs["device_map"] = {"": 0}
        except Exception as e:
            if verbose:
                print(f"[train] 4bit 不可用({e})，回退普通 LoRA")
            use_4bit = False

    # 兼容 transformers 4.x(torch_dtype) 与 5.x(dtype) 的命名差异
    if not use_4bit:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype, **load_kwargs)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, **load_kwargs)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
        except Exception as e:
            if verbose:
                print(f"[train] 整块放 GPU 失败（{str(e)[:120]}），改用 device_map='auto' 卸载")
            load_kwargs["device_map"] = "auto"
            model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    if gradient_checkpointing:
        # 4GB 显存的关键开关：不缓存中间激活，用重算换显存，避免显存溢出到内存(PCIe 拖慢 10x+)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if verbose:
        model.print_trainable_parameters()
        print(f"[train] gradient_checkpointing={gradient_checkpointing}")

    class _DS(Dataset):
        """预编码数据集。

        性能要点：若在 __getitem__ 里逐条 tokenize，则每个 epoch 都重复分词，
        在有 GPU 时 CPU 分词会成为瓶颈（实测吞吐被拖到 ~2s/样本）。
        这里在构造时一次性编码并缓存，之后每个 step 都是纯张量计算。
        """

        def __init__(self, rows):
            if verbose:
                print(f"[train] 预编码 {len(rows)} 条样本 ...")
            self.items = []
            for r in rows:
                try:
                    self.items.append(_encode(tokenizer, r["messages"], max_len))
                except Exception:
                    continue
            if verbose:
                print(f"[train] 预编码完成：{len(self.items)} 条")

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    def _collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = tokenizer.pad_token_id
        ids, att, lab = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad] * n)
            att.append(b["attention_mask"] + [0] * n)
            lab.append(b["labels"] + [-100] * n)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(att, dtype=torch.long),
            "labels": torch.tensor(lab, dtype=torch.long),
        }

    def _build_trainer(bs: int) -> Trainer:
        a = TrainingArguments(
            output_dir=str(out / "ckpt"),
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=epochs,
            learning_rate=lr,
            logging_steps=5,
            # 长跑保险：1.5B 在本机约 10-30s/step，一轮十几个小时；定期存 LoRA（约十几 MB）
            # 这样中途崩溃/断电/暂停不至于全部白跑。
            # 间隔从 400 降到 100：实测"训练到第 100 步后暂停"会因首个检查点在 400 步
            # 而**丢失全部进度**（无任何检查点可续）。降到 100 后单次暂停最多丢 100 步，
            # 代价只是多几次约 50MB 的落盘（save_total_limit=2 控制占用）。
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            eval_strategy="no",
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
            report_to=[],
            remove_unused_columns=False,
            dataloader_num_workers=0,
            optim="adamw_torch",
        )
        return Trainer(model=model, args=a, train_dataset=_DS(train_rows), data_collator=_collate)

    bs = batch_size
    result = None
    trainer = None
    # 断点续训：checkpoint-XXXX 一直有落盘，但此前没有任何入口能读回它，
    # 等于"崩溃安全"只是注释里的承诺。resume_from 让检查点真正可用；
    # 一旦因显存故障改了 batch，就不能再沿用旧检查点（优化器步数对不上）。
    resumed = str(resume_from) if resume_from else None
    while True:
        try:
            trainer = _build_trainer(bs)
            result = trainer.train(resume_from_checkpoint=resumed)
            break
        except RuntimeError as e:
            msg = str(e).lower()
            # 两类可恢复的显存故障：容量不足(OOM) 与 分配器内部断言(碎片)
            recoverable = "out of memory" in msg or "internal assert failed" in msg
            if not recoverable or bs <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            bs = max(1, bs // 2)
            resumed = None
            if verbose:
                print(f"[train] 显存故障（{e}），自动把 batch 降到 {bs} 重试")
    trainer.save_model(str(out))       # 只保存 LoRA 适配器
    tokenizer.save_pretrained(str(out))

    stats = {
        "ok": True,
        "base_model": base_model,
        "adapter_dir": str(out),
        "train_samples": len(train_rows),
        "epochs": epochs,
        "train_loss": float(result.training_loss) if result and result.training_loss else None,
        "resumed_from": str(resume_from) if resume_from else None,
    }
    (out / "train_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"[train] 完成，适配器已保存: {out}")
    return stats

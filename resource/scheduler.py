"""块调度与异构卸载规划（对应可行性报告 P1 + P2）。

把报告里的架构落成**可计算的规划器**：
- P1 块调度：按 KV 分页思路算清"上下文能吃多少"，避免碎片与 OOM；
- P2 异构卸载：按显存预算算出"多少层驻 GPU、多少层留 CPU"（对应 Ollama 的 num_gpu）。

关键设计原则（来自本机实测）：
**先保证不溢出，再谈吞吐。** 溢出到系统内存不会报错，却会让速度下降 7–10 倍。
所以本模块宁可保守（留安全边界），也不把显存用满。
"""
from __future__ import annotations

from dataclasses import dataclass

from .governor import Budget, read_budget

# 上下文下限：ctx=0 会让推理直接失效（实测踩到过，故设兜底）
MIN_CTX = 512

# 常见模型的推理结构参数（用于估算 KV cache，可按需扩展）
# n_layers / n_kv_heads / head_dim 决定 KV 每 token 的字节数
MODEL_SPECS: dict[str, dict] = {
    "qwen2.5-0.5b": {"layers": 24, "kv_heads": 2, "head_dim": 64, "hidden": 896, "params_b": 0.49},
    "qwen2.5-1.5b": {"layers": 28, "kv_heads": 2, "head_dim": 128, "hidden": 1536, "params_b": 1.54},
    "qwen2.5-3b": {"layers": 36, "kv_heads": 2, "head_dim": 128, "hidden": 2048, "params_b": 3.09},
    "qwen3-4b": {"layers": 36, "kv_heads": 8, "head_dim": 128, "hidden": 2560, "params_b": 4.0},
    "qwen2.5-7b": {"layers": 28, "kv_heads": 4, "head_dim": 128, "hidden": 3584, "params_b": 7.6},
}


def kv_bytes_per_token(spec: dict, *, kv_bits: int = 16) -> int:
    """每个 token 的 KV cache 字节数。

    公式：2（K 与 V）× 层数 × KV头数 × head_dim × 每元素字节数。
    这就是 PagedAttention 要"分页"管理的对象——它随上下文线性增长，
    是长上下文在 4GB 卡上最先爆掉的部分。
    """
    elem = kv_bits // 8
    return 2 * int(spec["layers"]) * int(spec["kv_heads"]) * int(spec["head_dim"]) * elem


def weight_bytes(params_b: float, bits: int = 4) -> float:
    """权重的理论字节数（GB）。分块/卸载都不能改变它，只有量化能。"""
    return params_b * 1e9 * (bits / 8) / (1024 ** 3)


@dataclass
class Plan:
    model: str
    quant_bits: int
    weights_gb: float
    ctx: int
    kv_gb: float
    gpu_layers: int
    total_layers: int
    offload_ratio: float
    mode: str
    notes: list[str]

    def as_dict(self) -> dict:
        return {
            "model": self.model, "quant_bits": self.quant_bits,
            "weights_gb": round(self.weights_gb, 2),
            "ctx": self.ctx, "kv_gb": round(self.kv_gb, 3),
            "gpu_layers": self.gpu_layers, "total_layers": self.total_layers,
            "offload_ratio": round(self.offload_ratio, 2),
            "mode": self.mode, "notes": self.notes,
        }


def max_ctx_that_fits(spec: dict, free_vram_gb: float, *, reserve_gb: float = 0.35,
                      kv_bits: int = 16, steps=(512, 1024, 2048, 4096, 8192, 16384, 32768)) -> int:
    """在给定空闲显存下，KV cache 最多能撑多长上下文（离散步长向上取）。"""
    per_tok = kv_bytes_per_token(spec, kv_bits=kv_bits)
    usable = max(0.0, free_vram_gb - reserve_gb) * (1024 ** 3)
    best = 0
    for c in steps:
        if c * per_tok <= usable:
            best = c
        else:
            break
    return best


def plan_run(model_key: str, *, quant_bits: int = 4, ctx: int | None = None,
             budget: Budget | None = None) -> Plan:
    """为一次推理生成块调度 + 卸载方案（P1/P2 的联合决策）。"""
    b = budget or read_budget()
    spec = MODEL_SPECS.get(model_key)
    if spec is None:
        raise KeyError(f"未收录模型 {model_key}；可选 {list(MODEL_SPECS)}")

    notes: list[str] = []
    w_gb = weight_bytes(spec["params_b"], quant_bits)

    # ---- P1：上下文与 KV 预算 ----
    if ctx is None:
        ctx = max_ctx_that_fits(spec, b.usable_vram_gb)
        if ctx < MIN_CTX:
            ctx = MIN_CTX          # 兜底：ctx=0 会让推理直接失效
            notes.append(f"可用显存紧张，上下文回落到最小可用值 {MIN_CTX}")
        else:
            notes.append(f"上下文按显存自动定为 {ctx}（KV 分页口径）")
    kv_gb = ctx * kv_bytes_per_token(spec, kv_bits=16) / (1024 ** 3)
    if b.contended:
        notes.append(f"⚠ 当前显存有竞争（空闲仅 {b.vram_free_gb:.2f} GB），"
                     f"以下按硬件规划容量 {b.planning_vram_gb:.2f} GB 计算；实际运行前建议先释放显存")

    # ---- P2：层卸载比例 ----
    total_layers = int(spec["layers"])
    need_gb = w_gb + kv_gb
    if not b.has_gpu:
        gpu_layers, mode = 0, "cpu_only"
        notes.append("未检测到可用 GPU，全部层走 CPU")
    elif need_gb <= b.usable_vram_gb:
        gpu_layers, mode = total_layers, "full_gpu"
        notes.append(f"权重 {w_gb:.2f} + KV {kv_gb:.2f} ≤ 可用 {b.usable_vram_gb:.2f} GB，可全层驻 GPU")
    else:
        # 只把"能装下的那部分权重"放 GPU：留出 KV 所需显存
        room_for_weights = max(0.0, b.usable_vram_gb - kv_gb)
        ratio = max(0.0, min(1.0, room_for_weights / w_gb)) if w_gb else 0.0
        gpu_layers = int(total_layers * ratio)
        mode = "partial_offload" if gpu_layers > 0 else "cpu_only"
        notes.append(
            f"权重 {w_gb:.2f} GB 超出可用显存，按 {ratio:.0%} 卸载 → {gpu_layers}/{total_layers} 层驻 GPU，"
            f"其余 {total_layers - gpu_layers} 层留内存（需双队列流水掩盖 PCIe）"
        )

    if kv_gb > b.usable_vram_gb * 0.6:
        notes.append(f"⚠ KV cache {kv_gb:.2f} GB 已占可用显存 >60%，建议降 ctx 或启用 KV 量化")
    notes.append("硬约束：显存留 15% 余量，避免溢出（溢出会慢 7–10 倍）")

    return Plan(
        model=model_key, quant_bits=quant_bits, weights_gb=w_gb, ctx=ctx, kv_gb=kv_gb,
        gpu_layers=gpu_layers, total_layers=total_layers,
        offload_ratio=1 - (gpu_layers / total_layers) if total_layers else 0.0,
        mode=mode, notes=notes,
    )


def ollama_options(plan: Plan) -> dict:
    """把方案翻译成 Ollama 的 options（可直接用于 API 调用）。"""
    opts: dict = {"num_ctx": int(plan.ctx)}
    if plan.gpu_layers >= plan.total_layers:
        opts["num_gpu"] = 999          # 全部层上 GPU
    else:
        opts["num_gpu"] = int(plan.gpu_layers)
    return opts


def describe(model_key: str, quant_bits: int = 4, ctx: int | None = None) -> str:
    p = plan_run(model_key, quant_bits=quant_bits, ctx=ctx)
    lines = [
        f"模型 {p.model}｜量化 {p.quant_bits}-bit｜权重 {p.weights_gb:.2f} GB",
        f"上下文 {p.ctx} tokens｜KV cache {p.kv_gb:.3f} GB",
        f"驻 GPU 层 {p.gpu_layers}/{p.total_layers}（卸载 {p.offload_ratio:.0%}）｜模式 {p.mode}",
        "Ollama options: " + str(ollama_options(p)),
        "说明：",
    ] + ["  - " + n for n in p.notes]
    return "\n".join(lines)

"""异构并行计算层：算力盘点与阶段分派（让闲置的 GPU/CPU 同时干活）。"""
from __future__ import annotations

from .devices import (
    CPU,
    CUDA,
    IGPU,
    NPU,
    ComputeInventory,
    Device,
    Stage,
    discover,
    llama_gpu_layers,
    plan_parallel,
    suggest_ollama_num_gpu,
)

__all__ = [
    "CUDA", "IGPU", "CPU", "NPU",
    "Device", "ComputeInventory", "Stage",
    "discover", "plan_parallel", "llama_gpu_layers", "suggest_ollama_num_gpu",
]

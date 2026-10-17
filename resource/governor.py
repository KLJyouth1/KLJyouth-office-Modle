"""资源治理层（对应可行性报告中的 P0：把"不溢出"变成硬约束）。

本模块来自本机实测得到的核心结论：
**"显存溢出到系统内存"是低配设备上最大的隐形性能杀手**——它不报错，
却会让速度下降 7–10 倍（实测：全参微调峰值 10.26GB / 卡 4GB → 单步 11.93s；
消除溢出后 0.30s/样本）。因此调度前必须先算清"装得下吗"，再决定卸载比例。
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

# 安全边界：显存不要用到 100%，桌面/驱动本身要占用
VRAM_SAFETY = 0.85
RAM_SAFETY = 0.80
# 桌面常驻保留（浏览器/桌面合成器等）。实测本机桌面约占用 1.2GB。
DESKTOP_RESERVE_GB = 1.5


@dataclass
class Budget:
    vram_total_gb: float
    vram_free_gb: float
    ram_total_gb: float
    ram_free_gb: float

    @property
    def planning_vram_gb(self) -> float:
        """**规划用**显存容量。

        实测教训：不能拿"当前空闲显存"当容量。若此刻有别的任务占着显存
        （本机训练时空闲一度只剩 0.11GB），规划器会算出 ctx=0 这种荒谬结果。
        正确口径是：以**硬件容量**扣掉桌面常驻保留，得到本机稳定可用的规划容量。
        """
        return round(max(0.0, self.vram_total_gb - DESKTOP_RESERVE_GB), 2)

    @property
    def usable_vram_gb(self) -> float:
        """可用于模型的显存（规划口径 × 安全系数）。"""
        return round(self.planning_vram_gb * VRAM_SAFETY, 2)

    @property
    def contended(self) -> bool:
        """是否存在显存竞争（有其他任务占着显存）。"""
        return self.has_gpu and self.vram_free_gb < self.usable_vram_gb * 0.5

    @property
    def usable_ram_gb(self) -> float:
        return round(self.ram_free_gb * RAM_SAFETY, 2)

    @property
    def has_gpu(self) -> bool:
        return self.vram_total_gb > 0

    def as_dict(self) -> dict:
        return {
            "vram_total_gb": self.vram_total_gb, "vram_free_gb": self.vram_free_gb,
            "planning_vram_gb": self.planning_vram_gb,
            "usable_vram_gb": self.usable_vram_gb, "contended": self.contended,
            "ram_total_gb": self.ram_total_gb, "ram_free_gb": self.ram_free_gb,
            "usable_ram_gb": self.usable_ram_gb,
        }


def _nvidia_vram() -> tuple[float, float]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return 0.0, 0.0
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, encoding="utf-8", errors="replace",
        )
        if out.returncode != 0 or not out.stdout.strip():
            return 0.0, 0.0
        first = out.stdout.strip().splitlines()[0]
        total, free = [float(x.strip()) for x in first.split(",")[:2]]
        return round(total / 1024, 2), round(free / 1024, 2)
    except Exception:
        return 0.0, 0.0


def read_budget() -> Budget:
    """读取当前 GPU/RAM 预算（真实值，非估算）。"""
    import psutil

    vm = psutil.virtual_memory()
    v_total, v_free = _nvidia_vram()
    return Budget(
        vram_total_gb=v_total, vram_free_gb=v_free,
        ram_total_gb=round(vm.total / 2**30, 2), ram_free_gb=round(vm.available / 2**30, 2),
    )


def decide(model_gb: float, budget: Budget | None = None) -> dict:
    """给定模型体量，决定执行模式与卸载比例（P0 的核心决策）。

    返回 mode ∈ {full_gpu, partial_offload, cpu_only, refuse}。
    """
    b = budget or read_budget()
    if not b.has_gpu:
        fits_cpu = model_gb <= b.usable_ram_gb
        return {"mode": "cpu_only" if fits_cpu else "refuse",
                "gpu_ratio": 0.0, "note": "无可用 GPU；依据内存判断" if fits_cpu else "内存也不足"}
    if model_gb <= b.usable_vram_gb:
        return {"mode": "full_gpu", "gpu_ratio": 1.0,
                "note": f"全部驻留显存（{model_gb:.2f} ≤ 可用 {b.usable_vram_gb:.2f} GB）"}
    total_pool = b.usable_vram_gb + b.usable_ram_gb
    if model_gb <= total_pool:
        ratio = round(b.usable_vram_gb / model_gb, 2)
        return {"mode": "partial_offload", "gpu_ratio": ratio,
                "note": f"按 {ratio:.0%} 驻留 GPU、其余驻内存（需双队列流水掩盖 PCIe）"}
    return {"mode": "refuse", "gpu_ratio": 0.0,
            "note": f"显存+内存合计 {total_pool:.1f} GB 仍装不下 {model_gb:.2f} GB，需量化或换 MoE"}


def check_spill(peak_alloc_gb: float, budget: Budget | None = None) -> dict:
    """溢出检测：判断一次运行是否已"溢出到内存"（这是隐形性能杀手）。"""
    b = budget or read_budget()
    if not b.has_gpu:
        return {"spilled": False, "severity": "n/a", "note": "无 GPU"}
    capacity = b.vram_total_gb
    if peak_alloc_gb <= capacity:
        util = peak_alloc_gb / capacity if capacity else 0
        sev = "high" if util > 0.95 else ("mid" if util > 0.85 else "ok")
        return {"spilled": False, "severity": sev,
                "note": f"峰值 {peak_alloc_gb:.2f} GB / 上限 {capacity:.2f} GB（占用 {util:.0%}）"}
    over = peak_alloc_gb - capacity
    return {"spilled": True, "severity": "critical",
            "note": f"已溢出 {over:.2f} GB 到系统内存 → 走 PCIe，实测会拖慢 7–10 倍。"
                    f"请启用梯度检查点 / 降低批或序列长度 / 降低卸载比例。"}


def advise(model_gb: float = 0.0) -> str:
    """生成一份可执行资源建议（CLI 展示用）。"""
    b = read_budget()
    lines = [
        f"显存：总 {b.vram_total_gb} GB / 空闲 {b.vram_free_gb} GB（安全可用 {b.usable_vram_gb} GB）"
        if b.has_gpu else "显存：未检测到可用 NVIDIA GPU → 只能 CPU 推理",
        f"内存：总 {b.ram_total_gb} GB / 空闲 {b.ram_free_gb} GB（安全可用 {b.usable_ram_gb} GB）",
    ]
    if model_gb > 0:
        d = decide(model_gb, b)
        lines.append(f"模型 {model_gb:.2f} GB → 模式 {d['mode']}（GPU 驻留 {d['gpu_ratio']:.0%}）：{d['note']}")
    lines.append("硬约束：先保证「不溢出」，再谈吞吐（实测溢出会让速度下降 7–10 倍）")
    return "\n".join(lines)

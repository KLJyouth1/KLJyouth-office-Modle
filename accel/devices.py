"""异构并行计算：把 dGPU / iGPU / CPU 同时用起来（回答"为什么不能并行"）。

先说清事实边界，避免把可能性说成能力：
- **可以真并行的**：多个独立子任务分派到不同设备同时执行（如 dGPU 跑主生成、
  CPU 跑工具执行与检索、iGPU 跑图像预处理/小模型），以及单模型把一部分层放
  GPU、一部分层放 CPU（llama.cpp/Ollama 的 n-gpu-layers 就是在做这件事）。
- **不能凭空变出来的**：把 4GB 显存变成 16GB。异构并行的收益是**吞吐与利用率的
  提升**，不是总容量与单次大模型推理能力的提升。任何声称能让 4GB 跑 70B 全速
  的说法都不成立（除非做极重度的量化+大量 CPU 参与，速度会掉到不可用）。
- 因此本模块的目标是：**让闲置的 iGPU 与 CPU 分担本可并行的活儿**，
  并给出可实测的分层/分派方案，而不是承诺突破物理容量上限。

设计原则：探测必须基于实测（nvidia-smi / WMI / ORT provider），
拿不到证据的就标为 unavailable，绝不假设某设备"应该有"。
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field

# 设备类别
CUDA, IGPU, CPU, NPU, UNKNOWN = "cuda", "igpu", "cpu", "npu", "unknown"

# 核显参与计算所需的运行时（缺少时只能报告"可用但需装运行时"，不谎称可用）
_IGPU_RUNTIMES = ("openvino", "onnxruntime-directml", "torch_directml")


@dataclass
class Device:
    """一个计算单元。"""

    name: str
    kind: str
    memory_mb: int = 0
    backend: str = ""              # 实际可用的后端：cuda / openvino / directml / cpu
    available: bool = False        # 是否**当前就能拿来算**
    note: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "memory_mb": self.memory_mb,
                "backend": self.backend, "available": self.available, "note": self.note}


@dataclass
class ComputeInventory:
    """本机算力盘点结果。"""

    devices: list[Device] = field(default_factory=list)
    cpu_cores: int = 0
    cpu_logical: int = 0
    ram_gb: float = 0.0

    def usable(self) -> list[Device]:
        return [d for d in self.devices if d.available]

    def find(self, kind: str) -> Device | None:
        return next((d for d in self.devices if d.kind == kind), None)

    def as_dict(self) -> dict:
        return {"cpu_cores": self.cpu_cores, "cpu_logical": self.cpu_logical,
                "ram_gb": self.ram_gb, "devices": [d.as_dict() for d in self.devices]}

    def summary(self) -> str:
        lines = [f"CPU {self.cpu_cores}核/{self.cpu_logical}线程，内存 {self.ram_gb:.1f}GB"]
        for d in self.devices:
            flag = "可用" if d.available else "不可用"
            extra = f"（{d.note}）" if d.note else ""
            lines.append(f"{d.kind:6} {d.name} [{flag}/{d.backend or '-'}]"
                         f"{f' {d.memory_mb}MB' if d.memory_mb else ''}{extra}")
        return "\n".join(lines)


# ---------------------------------------------------------------------- 探测
def _nvidia() -> list[Device]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
    except Exception:
        return []
    out = []
    for line in (r.stdout or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                mb = int(float(parts[1]))
            except ValueError:
                mb = 0
            out.append(Device(name=parts[0], kind=CUDA, memory_mb=mb, backend="cuda",
                              available=True))
    return out


def _intel_igpu() -> list[Device]:
    """探测核显，并如实标注它当前能否参与计算。"""
    out: list[Device] = []
    try:
        import psutil  # noqa: F401  仅用于确认环境，不参与判断
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | "
             "Select-Object Name,AdapterRAM,Status | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=30)
        raw = (r.stdout or "").strip()
        if raw:
            data = json.loads(raw)
            if isinstance(data, dict):
                data = [data]
            for v in data:
                name = str(v.get("Name") or "")
                if not name:
                    continue
                low = name.lower()
                # 核显识别：Intel(R) UHD/Iris/Arc 是 Intel 的集显/独显；
                # Virtual Display / GameViewer 这类虚拟适配器不是计算设备，排除。
                if "intel" not in low and "amd radeon graphics" not in low:
                    continue
                if "virtual" in low:
                    continue
                mb = int(v.get("AdapterRAM") or 0) // (1024 * 1024) if v.get("AdapterRAM") else 0
                rt = _igpu_runtime()
                out.append(Device(
                    name=name, kind=IGPU, memory_mb=mb,
                    backend=rt or "none",
                    available=bool(rt),
                    note="" if rt else f"需安装 {'/'.join(_IGPU_RUNTIMES)} 才能参与计算",
                ))
    except Exception:
        pass
    return out


def _igpu_runtime() -> str:
    """核显可用后端。ORT 的 provider 是权威判据（CPU EP 不算核显可用）。"""
    import importlib.util as u

    if u.find_spec("openvino"):
        return "openvino"
    try:
        import onnxruntime as ort

        provs = ort.get_available_providers()
        for p, tag in (("DmlExecutionProvider", "directml"),
                       ("OpenVINOExecutionProvider", "openvino")):
            if p in provs:
                return tag
    except Exception:
        pass
    return ""


def _openvino_devices() -> list[Device]:
    """用 OpenVINO 枚举真实可计算的设备（它的枚举是"能算"的权威证据）。

    OpenVINO 会同时看到 Intel 核显与 NVIDIA 独显，因此这里只负责产出
    "经 OpenVINO 确认可用"的设备清单，由 merge 阶段与 nvidia-smi/WMI 的结果合并，
    避免同一块卡被记成两个设备。
    """
    try:
        import openvino as ov
    except Exception:
        return []
    out: list[Device] = []
    try:
        core = ov.Core()
        for d in core.available_devices:
            try:
                full = str(core.get_property(d, "FULL_DEVICE_NAME"))
            except Exception:
                full = d
            low = full.lower()
            if d.startswith("GPU") and "intel" in low:
                kind = IGPU
            elif d.startswith("GPU"):
                kind = CUDA
            elif d.startswith("NPU"):
                kind = NPU
            else:
                kind = CPU
            out.append(Device(name=full, kind=kind, backend="openvino", available=True,
                              note=f"OpenVINO 设备 {d}"))
    except Exception:
        return []
    return out


def _canon(name: str) -> str:
    """把设备名归一化成可比较的键，用于识别"两路探测看到的同一块硬件"。

    实测踩到的坑：nvidia-smi 报 "NVIDIA GeForce RTX 3050 Ti Laptop GPU"，
    而 OpenVINO 报 "... GPU (dGPU)"。若只去掉厂商词而保留括号内容，
    两边会得到 "3050 ti laptop gpu" 与 "3050 ti laptop gpu dgpu" → 判成两台设备，
    规划器于是把两个重算力阶段并行派给同一块卡，实测反而更慢。
    故必须先整段剥掉括号及其内容，再去掉泛化词，只留下真正区分的型号/系列词。
    """
    s = re.sub(r"\([^)]*\)", " ", (name or "").lower())     # 先剥掉括号与其中内容
    for tag in ("intel", "nvidia", "geforce", "quadro", "graphics", "laptop",
                "notebook", "gpu", "corporation", "amd", "advanced micro devices"):
        s = s.replace(tag, " ")
    return " ".join(s.split())


def _same_device(a: str, b: str) -> bool:
    """两个名字是否指同一块硬件。允许包含关系（如 "iris xe" 与 "iris xe max"）。"""
    ka, kb = _canon(a), _canon(b)
    if not ka or not kb:
        return ka == kb
    return ka == kb or ka in kb or kb in ka


def _merge(primary: list[Device], extra: list[Device]) -> list[Device]:
    """合并两路探测结果：同名视为同一设备，取"更可用"的那份信息。

    为什么必须合并：nvidia-smi 与 OpenVINO 会分别看到同一块 RTX 3050 Ti。
    直接拼接会让规划器以为有两块卡，从而把两个重算力阶段并行派给同一块卡 → 反而变慢。
    """
    remaining = list(extra)
    out: list[Device] = []
    for d in primary:
        hit = next((e for e in remaining if _same_device(d.name, e.name)), None)
        if hit is not None:
            remaining.remove(hit)
            # 同一设备：以 primary 的名字/显存为准，但采用 extra 的可用性与后端
            out.append(Device(name=d.name, kind=d.kind, memory_mb=d.memory_mb or hit.memory_mb,
                              backend=hit.backend or d.backend, available=True,
                              note=hit.note or d.note))
        else:
            out.append(d)
    out.extend(remaining)
    return out


def _cpu() -> tuple[Device, int, int, float]:
    cores = logical = 0
    ram = 0.0
    try:
        import psutil

        cores = psutil.cpu_count(logical=False) or 0
        logical = psutil.cpu_count() or 0
        ram = psutil.virtual_memory().total / 2**30
    except Exception:
        pass
    dev = Device(name="CPU", kind=CPU, memory_mb=int(ram * 1024), backend="cpu",
                 available=cores > 0)
    return dev, cores, logical, ram


def discover() -> ComputeInventory:
    """盘点本机算力：逐个实测，拿不到证据的设备一律标为不可用。"""
    cpu, cores, logical, ram = _cpu()
    primary = _merge([*_nvidia(), *_intel_igpu()], _openvino_devices())
    return ComputeInventory(devices=[*primary, cpu],
                            cpu_cores=cores, cpu_logical=logical, ram_gb=ram)


# ---------------------------------------------------------------------- 调度
@dataclass
class Stage:
    """一个可独立执行的工作阶段。"""

    name: str
    kind: str            # compute（吃算力，如 LLM 生成）/ io（等网络磁盘）/ light（轻量计算）
    needs_mb: int = 0    # 显存/内存需求估计


# 各类阶段优先落在哪个设备：这里体现"并行"的核心判断——
# 主生成占 dGPU，而 IO/轻量计算根本不该占 GPU，它们可以同时跑在 CPU/iGPU 上。
_PREFERENCE: dict[str, tuple[str, ...]] = {
    "compute": (CUDA, CPU),          # 重算力优先独显，其次 CPU
    "light": (IGPU, CPU, CUDA),      # 轻量计算交给闲置的核显/CPU，不抢独显
    "io": (CPU,),                    # 网络/磁盘等待本就该用 CPU 线程，不占算力
}


def plan_parallel(inv: ComputeInventory, stages: list[Stage]) -> dict:
    """把阶段分派到设备。返回分配表 + 可并行的批次划分。

    批次划分规则：落到**不同设备**的阶段互不争抢，可同批并行；
    落到**同一设备**的阶段串行（同一算力单元同一时刻只做一件事）。
    """
    assign: dict[str, dict] = {}
    load: dict[str, int] = {}
    for st in stages:
        prefs = _PREFERENCE.get(st.kind, (CPU,))
        chosen = None
        for pref in prefs:
            d = inv.find(pref)
            if d and d.available and (st.needs_mb <= 0 or d.memory_mb >= st.needs_mb):
                chosen = d
                break
        if chosen is None:
            chosen = inv.find(CPU) or Device("CPU", CPU, backend="cpu", available=True)
        assign[st.name] = {"device": chosen.name, "kind": chosen.kind}
        load[chosen.kind] = load.get(chosen.kind, 0) + 1

    # 按设备分组即为"可并行批"；组内按顺序串行
    batches: dict[str, list[str]] = {}
    for name, a in assign.items():
        batches.setdefault(a["kind"], []).append(name)
    parallelizable = sum(1 for group in batches.values() if len(group) >= 1)
    return {"assign": assign, "batches": batches,
            "parallel_width": parallelizable,
            "device_load": load,
            "note": "不同设备的阶段可同批并行；同设备内串行（单卡同一时刻只做一件事）"}


def llama_gpu_layers(total_layers: int, vram_mb: int, *, per_layer_mb: int = 0,
                     reserve_mb: int = 512) -> int:
    """估算应放多少层到 GPU（llama.cpp/Ollama 的 n-gpu-layers）。

    这是"GPU+CPU 并行"在单模型上的体现：GPU 层与 CPU 层**同时**参与前向，
    不是二选一。层数估算刻意保守（留 reserve 给 KV cache 与激活），
    因为高估会导致溢出到内存、速度反而比纯 CPU 更慢。
    """
    if per_layer_mb <= 0:
        return 0
    usable = max(0, vram_mb - reserve_mb)
    return max(0, min(total_layers, usable // per_layer_mb))


def suggest_ollama_num_gpu(vram_mb: int, model_layers: int, *, per_layer_mb: int = 0) -> dict:
    """给 Ollama 的 `num_gpu` 参数一个实测型建议值。"""
    n = llama_gpu_layers(model_layers, vram_mb, per_layer_mb=per_layer_mb)
    return {"num_gpu_layers": n, "total_layers": model_layers,
            "mode": "gpu+cpu 混合" if 0 < n < model_layers else
                    ("全 GPU" if n >= model_layers else "全 CPU"),
            "note": "GPU 层与 CPU 层并行前向；层数高估会溢出到内存反而变慢"}

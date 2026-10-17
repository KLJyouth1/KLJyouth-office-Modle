"""异构并行计算层回归测试 —— 不依赖真实 GPU（用替身设备）。

锁定本轮的关键设计约束：
1. **合并同名设备**：nvidia-smi 与 OpenVINO 会看到同一块卡，若不合并，
   规划器会以为有两块卡，把两个重算力阶段并行派给同一块卡 → 反而变慢。
2. **不可用设备不得被分派**：拿不到运行时的设备（如未装 OpenVINO 的核显）
   必须标 unavailable 且不参与分派，绝不假装可用。
3. **不同设备才可并行**：同设备内的阶段必须串行（单卡同一时刻只做一件事）。
4. **层数估算保守**：高估 n-gpu-layers 会导致溢出到内存，比纯 CPU 更慢。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omnipilot.accel import devices as D  # noqa: E402
from omnipilot.accel.devices import (  # noqa: E402
    CPU,
    CUDA,
    IGPU,
    ComputeInventory,
    Device,
    Stage,
    llama_gpu_layers,
    plan_parallel,
    suggest_ollama_num_gpu,
)


def inv_with(*devs: Device, cores: int = 12, ram: float = 31.7) -> ComputeInventory:
    return ComputeInventory(devices=list(devs), cpu_cores=cores, cpu_logical=16, ram_gb=ram)


class TestMerge:
    def test_merges_same_physical_device_seen_by_two_probes(self):
        # nvidia-smi 报 "NVIDIA GeForce RTX 3050 Ti Laptop GPU"，
        # OpenVINO 报 "NVIDIA GeForce RTX 3050 Ti Laptop GPU (dGPU)" → 必须视为一台。
        primary = [Device("NVIDIA GeForce RTX 3050 Ti Laptop GPU", CUDA, 4096, "cuda", True)]
        extra = [Device("NVIDIA GeForce RTX 3050 Ti Laptop GPU (dGPU)", CUDA, 0, "openvino", True)]
        merged = D._merge(primary, extra)
        assert len(merged) == 1
        assert merged[0].memory_mb == 4096            # 保留 primary 的显存信息
        assert merged[0].available is True

    def test_keeps_devices_only_one_probe_saw(self):
        primary = [Device("NVIDIA GeForce RTX 3050 Ti Laptop GPU", CUDA, 4096, "cuda", True)]
        extra = [Device("Intel(R) Iris(R) Xe Graphics (iGPU)", IGPU, 0, "openvino", True)]
        merged = D._merge(primary, extra)
        assert {d.kind for d in merged} == {CUDA, IGPU}

    def test_marks_device_available_when_openvino_confirms_it(self):
        # 未装 OpenVINO 时核显是不可用的；装了之后必须被翻成可用。
        primary = [Device("Intel(R) Iris(R) Xe Graphics", IGPU, 2047, "none", False)]
        extra = [Device("Intel(R) Iris(R) Xe Graphics (iGPU)", IGPU, 0, "openvino", True)]
        merged = D._merge(primary, extra)
        assert merged[0].available is True
        assert merged[0].backend == "openvino"


class TestInventory:
    def test_usable_excludes_unavailable_devices(self):
        inv = inv_with(Device("gpu", CUDA, 4096, "cuda", True),
                       Device("igpu", IGPU, 2047, "none", False),
                       Device("CPU", CPU, 32000, "cpu", True))
        assert {d.kind for d in inv.usable()} == {CUDA, CPU}

    def test_find_returns_first_of_kind(self):
        inv = inv_with(Device("gpu", CUDA, 4096, "cuda", True))
        assert inv.find(CUDA).name == "gpu"
        assert inv.find(IGPU) is None


class TestPlanParallel:
    def test_heavy_compute_goes_to_cuda_and_io_to_cpu(self):
        inv = inv_with(Device("gpu", CUDA, 4096, "cuda", True), Device("CPU", CPU, 32000, "cpu", True))
        plan = plan_parallel(inv, [Stage("gen", "compute", 1200), Stage("exec", "io")])
        assert plan["assign"]["gen"]["kind"] == CUDA
        assert plan["assign"]["exec"]["kind"] == CPU

    def test_unavailable_gpu_is_never_assigned(self):
        inv = inv_with(Device("gpu", CUDA, 0, "cuda", False), Device("CPU", CPU, 32000, "cpu", True))
        plan = plan_parallel(inv, [Stage("gen", "compute", 1200)])
        assert plan["assign"]["gen"]["kind"] == CPU

    def test_light_work_prefers_igpu_when_available(self):
        inv = inv_with(Device("igpu", IGPU, 2047, "openvino", True),
                       Device("gpu", CUDA, 4096, "cuda", True),
                       Device("CPU", CPU, 32000, "cpu", True))
        plan = plan_parallel(inv, [Stage("prep", "light")])
        # 轻量计算不该去抢独显的显存
        assert plan["assign"]["prep"]["kind"] == IGPU

    def test_light_work_falls_back_to_cpu_without_igpu(self):
        inv = inv_with(Device("gpu", CUDA, 4096, "cuda", True), Device("CPU", CPU, 32000, "cpu", True))
        plan = plan_parallel(inv, [Stage("prep", "light")])
        assert plan["assign"]["prep"]["kind"] == CPU

    def test_same_device_stages_are_grouped_for_serial_execution(self):
        inv = inv_with(Device("gpu", CUDA, 4096, "cuda", True), Device("CPU", CPU, 32000, "cpu", True))
        plan = plan_parallel(inv, [Stage("a", "compute", 100), Stage("b", "compute", 100)])
        assert len(plan["batches"][CUDA]) == 2
        assert plan["parallel_width"] >= 1

    def test_stage_needing_more_memory_than_device_falls_back(self):
        inv = inv_with(Device("igpu", IGPU, 512, "openvino", True), Device("CPU", CPU, 32000, "cpu", True))
        plan = plan_parallel(inv, [Stage("big", "light", needs_mb=99999)])
        assert plan["assign"]["big"]["kind"] == CPU


class TestLayerSplit:
    def test_returns_zero_without_per_layer_estimate(self):
        # 没有每层显存数据时不做猜测——猜错会让模型溢出到内存，比纯 CPU 更慢。
        assert llama_gpu_layers(28, 4096, per_layer_mb=0) == 0

    def test_reserves_headroom_for_kv_cache(self):
        # 每层 200MB、共 28 层需 5600MB；4GB 卡装不下 → 必须只放一部分层。
        n = llama_gpu_layers(28, 4096, per_layer_mb=200, reserve_mb=512)
        assert 0 <= n < 28

    def test_never_exceeds_total_layers(self):
        assert llama_gpu_layers(28, 100000, per_layer_mb=10) == 28

    def test_suggest_reports_hybrid_mode(self):
        r = suggest_ollama_num_gpu(4096, 28, per_layer_mb=200)
        assert r["mode"] == "gpu+cpu 混合"
        assert r["num_gpu_layers"] < r["total_layers"]

    def test_suggest_reports_full_gpu_when_it_fits(self):
        r = suggest_ollama_num_gpu(4096, 28, per_layer_mb=60)
        assert r["mode"] == "全 GPU"

"""异构算力实测台：同一计算在不同单元上跑，并测"并行"是否真带来吞吐提升。

为什么必须实测而不是查规格书：核显标称 TFLOPS 与"实际能否加速这个负载"是两回事。
Iris Xe 跑小矩阵乘可能因 kernel 启动开销而**慢于 CPU**；只有实测才知道该把活派给谁。
本模块给出可复现的数字，用来支撑 plan_parallel 的分派决策，而不是凭直觉。

依赖 OpenVINO（可选）：装了就测，没装就如实报告不可测。
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field

from .devices import CPU, CUDA, IGPU, discover


def _ov():
    try:
        import openvino as ov

        return ov
    except Exception:
        return None


def build_mlp(ov, *, in_dim: int = 256, hidden: int = 1024, batch: int = 32):
    """构造一个可复现的小 MLP 图（MatMul -> Add -> ReLU），代表轻量计算负载。

    刻意不依赖外部 onnx 文件：让基准在任何机器上都能立刻跑，不因缺模型而搁置。
    """
    import numpy as np

    rng = np.random.default_rng(0)
    w1 = rng.standard_normal((in_dim, hidden)).astype(np.float32) * 0.02
    b1 = np.zeros((hidden,), dtype=np.float32)
    w2 = rng.standard_normal((hidden, in_dim)).astype(np.float32) * 0.02
    b2 = np.zeros((in_dim,), dtype=np.float32)

    p = ov.opset13.parameter([batch, in_dim], ov.Type.f32, name="x")
    c1 = ov.opset13.constant(w1)
    c2 = ov.opset13.constant(b1)
    h = ov.opset13.matmul(p, c1, False, False)
    h = ov.opset13.add(h, c2)
    h = ov.opset13.relu(h)
    o = ov.opset13.matmul(h, ov.opset13.constant(w2), False, False)
    o = ov.opset13.add(o, ov.opset13.constant(b2))
    return ov.Model([o], [p], "mlp")


@dataclass
class BenchResult:
    device: str
    ok: bool
    ms: float = 0.0
    iters: int = 0
    error: str = ""

    def as_dict(self) -> dict:
        return {"device": self.device, "ok": self.ok, "ms": round(self.ms, 3),
                "iters": self.iters, "error": self.error}


def bench_device(ov_device: str, *, iters: int = 30, in_dim: int = 256,
                 hidden: int = 1024, batch: int = 32) -> BenchResult:
    """在指定 OpenVINO 设备上跑固定次数的推理，返回平均单次耗时（ms）。"""
    ov = _ov()
    if ov is None:
        return BenchResult(ov_device, False, error="未安装 openvino")
    import numpy as np

    try:
        core = ov.Core()
        model = build_mlp(ov, in_dim=in_dim, hidden=hidden, batch=batch)
        compiled = core.compile_model(model, ov_device)
        x = np.zeros((batch, in_dim), dtype=np.float32)
        infer = compiled.create_infer_request()
        infer.infer({0: x})                      # 预热：排除首次编译/上传开销
        t0 = time.perf_counter()
        for _ in range(iters):
            infer.infer({0: x})
        ms = (time.perf_counter() - t0) * 1000 / iters
        return BenchResult(ov_device, True, ms=ms, iters=iters)
    except Exception as e:  # 设备不可用/编译失败都要如实返回，不伪造数字
        return BenchResult(ov_device, False, error=f"{type(e).__name__}: {str(e)[:160]}")


def compare_all(*, iters: int = 30) -> dict:
    """在 OpenVINO 暴露的每台设备上各测一次。"""
    ov = _ov()
    if ov is None:
        return {"ok": False, "message": "未安装 openvino，无法实测异构算力"}
    core = ov.Core()
    devices = list(core.available_devices)
    results = [bench_device(d, iters=iters) for d in devices]
    ok = [r for r in results if r.ok]
    best = min(ok, key=lambda r: r.ms) if ok else None
    return {"ok": True, "devices": [r.as_dict() for r in results],
            "fastest": best.device if best else None,
            "fastest_ms": round(best.ms, 3) if best else None}


def parallel_vs_sequential(devices: list[str], rounds: int = 3, *, iters: int = 20) -> dict:
    """对比"逐个跑"与"同时跑"的总耗时——这是"并行"是否真有效的直接证据。

    注意：只有当负载互不争抢资源时才可能加速。同一块物理 GPU 的两个 OpenVINO 设备
    （如 GPU.0 核显与 GPU.1 独显）是**不同硬件**，可以真并行；
    若两个任务落在同一块卡上，则只能串行，此处也会如实体现为"无加速"。
    """
    ov = _ov()
    if ov is None:
        return {"ok": False, "message": "未安装 openvino"}
    import numpy as np

    def make(d):
        core = ov.Core()
        model = build_mlp(ov)
        compiled = core.compile_model(model, d)
        x = np.zeros((32, 256), dtype=np.float32)
        req = compiled.create_infer_request()
        req.infer({0: x})
        return req, x

    try:
        prepared = [(d, *make(d)) for d in devices]
    except Exception as e:
        return {"ok": False, "message": f"编译失败：{type(e).__name__}: {str(e)[:160]}"}

    # 串行
    t0 = time.perf_counter()
    for _ in range(rounds):
        for _d, req, x in prepared:
            for _ in range(iters):
                req.infer({0: x})
    seq = time.perf_counter() - t0

    # 并行（每设备一个线程）
    def worker(req, x, n):
        for _ in range(n):
            req.infer({0: x})

    threads = [threading.Thread(target=worker, args=(req, x, rounds * iters))
               for _d, req, x in prepared]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    par = time.perf_counter() - t0

    return {"ok": True, "devices": devices, "sequential_s": round(seq, 4),
            "parallel_s": round(par, 4),
            "speedup": round(seq / par, 3) if par > 0 else None,
            "note": "加速比 >1 表示并行确实生效；≈1 说明负载受限或落在同一硬件上"}


def recommend_placement(*, iters: int = 30) -> dict:
    """综合实测 + 盘点，给出"什么活交给谁"的建议。"""
    inv = discover()
    bench = compare_all(iters=iters)
    if not bench.get("ok"):
        return {"inventory": inv.as_dict(), "bench": bench}
    by_dev: dict[str, float] = {d["device"]: d["ms"] for d in bench["devices"] if d["ok"]}
    # CPU 是保底；若核显比 CPU 快，轻量计算就应该改派核显，否则留在 CPU 更稳
    cpu_ms = by_dev.get("CPU")
    igpu_ms = next((ms for k, ms in by_dev.items() if k.startswith("GPU.0")), None)
    advice = []
    if cpu_ms and igpu_ms:
        if igpu_ms < cpu_ms:
            advice.append(f"轻量计算改派核显：实测核显 {igpu_ms:.2f}ms < CPU {cpu_ms:.2f}ms")
        else:
            advice.append(f"轻量计算留在 CPU：实测核显 {igpu_ms:.2f}ms 不优于 CPU {cpu_ms:.2f}ms，"
                          "核显仍有价值——它能与 dGPU 并行而不抢显存")
    return {"inventory": inv.as_dict(), "bench": bench, "advice": advice,
            "kinds": {CPU: "通用/IO", IGPU: "轻量并行计算（不占 dGPU 显存）", CUDA: "主生成"}}


if __name__ == "__main__":
    print(json.dumps(recommend_placement(), ensure_ascii=False, indent=2))

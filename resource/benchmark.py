"""推理性能实测（为报告提供**真实**指标，而非估算）。

通过 Ollama 的 /api/generate 流式接口测量：
- 首 token 延迟（TTFT）：反映预填充与权重加载成本
- 解码速度（tok/s）：反映稳态带宽效率，是低配设备的核心指标
- 显存占用：结合 governor.check_spill 判断是否溢出

设计意图：报告里的每一个性能数字都应能被本模块复现。
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass

from ..config import OLLAMA_HOST


@dataclass
class BenchResult:
    model: str
    prompt: str
    ttft_s: float
    decode_tok_s: float
    total_s: float
    n_tokens: int
    load_s: float
    eval_count: int

    def as_dict(self) -> dict:
        return {
            "model": self.model, "ttft_s": round(self.ttft_s, 3),
            "decode_tok_s": round(self.decode_tok_s, 2),
            "total_s": round(self.total_s, 3), "n_tokens": self.n_tokens,
            "load_s": round(self.load_s, 3), "eval_count": self.eval_count,
        }


def _post_stream(path: str, payload: dict, timeout: float = 600):
    req = urllib.request.Request(
        f"{OLLAMA_HOST.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def bench_generate(model: str, prompt: str = "用一句话说明你能做什么。",
                   *, num_predict: int = 64, num_ctx: int = 2048,
                   num_gpu: int | None = None) -> BenchResult:
    """跑一次生成并测量 TTFT 与解码速度。"""
    options: dict = {"num_predict": int(num_predict), "num_ctx": int(num_ctx),
                     "temperature": 0.0}
    if num_gpu is not None:
        options["num_gpu"] = int(num_gpu)
    payload = {"model": model, "prompt": prompt, "stream": True,
               "options": options, "keep_alive": "2m"}

    t0 = time.time()
    first_at = None
    n = 0
    load_s = 0.0
    eval_count = 0
    resp = _post_stream("/api/generate", payload)
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("response"):
                if first_at is None:
                    first_at = time.time()
                n += 1
            if obj.get("done"):
                load_s = (obj.get("load_duration") or 0) / 1e9
                eval_count = int(obj.get("eval_count") or 0)
                break
    t1 = time.time()

    ttft = (first_at - t0) if first_at else (t1 - t0)
    decode_window = max(1e-6, t1 - (first_at or t0))
    # 用 Ollama 自报的 eval_count 更准；回退到本地计数
    tokens = eval_count or n
    speed = tokens / decode_window if decode_window > 0 else 0.0
    return BenchResult(model=model, prompt=prompt[:40], ttft_s=ttft, decode_tok_s=speed,
                       total_s=t1 - t0, n_tokens=tokens, load_s=load_s, eval_count=eval_count)


def bench_suite(models: list[str], *, num_predict: int = 64) -> list[dict]:
    """对多个模型跑同一基准，返回可对比表。"""
    out = []
    for m in models:
        try:
            r = bench_generate(m, num_predict=num_predict)
            out.append(r.as_dict())
        except Exception as e:
            out.append({"model": m, "error": f"{type(e).__name__}: {e}"})
    return out


def format_table(rows: list[dict]) -> str:
    if not rows:
        return "（无数据）"
    head = f"{'模型':<26}{'TTFT(s)':>9}{'tok/s':>9}{'总(s)':>8}{'加载(s)':>9}"
    lines = [head, "-" * len(head)]
    for r in rows:
        if "error" in r:
            lines.append(f"{r['model']:<26}{'FAILED':>9}  {r['error'][:40]}")
            continue
        lines.append(
            f"{r['model']:<26}{r['ttft_s']:>9.3f}{r['decode_tok_s']:>9.2f}"
            f"{r['total_s']:>8.2f}{r['load_s']:>9.3f}"
        )
    return "\n".join(lines)

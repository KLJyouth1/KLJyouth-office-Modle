"""KV cache 量化与块调度的回归测试。

锁定本轮实测得到的硬约束：
1. KV 量化档位必须**同时**影响 kv_gb 估算与自动 ctx 上限（旧实现把 kv_bits
   写死成 16，导致规划器永远按 f16 算，q8_0/q4_0 的收益完全看不见）。
2. f16 → q8_0 → q4_0 的 KV 占用必须是 1 : 1/2 : 1/4。
3. KV 量化是服务端环境变量，必须由 ollama_env() 暴露，而不是混进请求 options。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omnipilot.resource.governor import Budget  # noqa: E402
from omnipilot.resource.scheduler import (  # noqa: E402
    DECISION_KV_TYPE,
    KV_QUANT_BITS,
    kv_bytes_per_token,
    MODEL_SPECS,
    ollama_env,
    ollama_options,
    plan_run,
)

BUDGET = Budget(vram_total_gb=4.0, vram_free_gb=3.0,
                ram_total_gb=32.0, ram_free_gb=20.0)


def test_kv_quant_ratio_is_one_half_quarter():
    f16 = plan_run("qwen2.5-1.5b", ctx=8192, budget=BUDGET, kv_type="f16")
    q8 = plan_run("qwen2.5-1.5b", ctx=8192, budget=BUDGET, kv_type="q8_0")
    q4 = plan_run("qwen2.5-1.5b", ctx=8192, budget=BUDGET, kv_type="q4_0")

    assert q8.kv_gb == f16.kv_gb / 2
    assert q4.kv_gb == f16.kv_gb / 4
    # 估算口径必须与实测一致：1.5B/28 层/2 KV 头/128 维，f16 = 28 KB/token。
    assert kv_bytes_per_token(MODEL_SPECS["qwen2.5-1.5b"], kv_bits=16) == 28672


def test_kv_quant_raises_auto_ctx_ceiling():
    f16 = plan_run("qwen2.5-1.5b", budget=BUDGET, kv_type="f16")
    q8 = plan_run("qwen2.5-1.5b", budget=BUDGET, kv_type="q8_0")
    assert q8.ctx >= f16.ctx


def test_unknown_kv_type_is_rejected():
    try:
        plan_run("qwen2.5-1.5b", ctx=2048, budget=BUDGET, kv_type="int4")
    except ValueError as e:
        assert "int4" in str(e)
    else:
        raise AssertionError("未知 KV 档位必须报错，不能静默按 f16 处理")


def test_ollama_env_carries_kv_type():
    plan = plan_run("qwen2.5-1.5b", ctx=4096, budget=BUDGET, kv_type="q8_0")
    env = ollama_env(plan)
    assert env == {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0"}
    assert set(KV_QUANT_BITS) == {"f16", "q8_0", "q4_0"}
    # 请求 options 里不应出现 KV 类型（改不了显存格式，只会让人误以为生效）。
    assert "kv_cache_type" not in ollama_options(plan)


def test_decision_default_stays_f16():
    # 实测 q8_0 会让决策模型掉 6.3 个点、q4_0 直接打崩字母输出，
    # 因此"默认值"必须锁死在 f16，不能因为省显存而悄悄换档。
    assert DECISION_KV_TYPE == "f16"
    plan = plan_run("qwen2.5-1.5b", ctx=2048, budget=BUDGET)
    assert plan.kv_type == "f16"

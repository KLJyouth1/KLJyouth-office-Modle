"""资源治理层：预算、溢出检测、块调度、异构卸载、性能实测。"""
from .benchmark import BenchResult, bench_generate, bench_suite, format_table
from .governor import Budget, advise, check_spill, decide, read_budget
from .scheduler import (
    DECISION_KV_TYPE,
    KV_QUANT_BITS,
    MODEL_SPECS,
    Plan,
    describe as describe_plan,
    kv_bytes_per_token,
    max_ctx_that_fits,
    ollama_env,
    ollama_options,
    plan_run,
    weight_bytes,
)

__all__ = [
    "Budget", "read_budget", "decide", "check_spill", "advise",
    "MODEL_SPECS", "KV_QUANT_BITS", "DECISION_KV_TYPE", "Plan", "plan_run", "describe_plan",
    "ollama_options", "ollama_env", "max_ctx_that_fits", "kv_bytes_per_token", "weight_bytes",
    "BenchResult", "bench_generate", "bench_suite", "format_table",
]

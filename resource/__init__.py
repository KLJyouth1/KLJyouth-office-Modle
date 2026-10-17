"""资源治理层：预算、溢出检测、块调度、异构卸载、性能实测。"""
from .benchmark import BenchResult, bench_generate, bench_suite, format_table
from .governor import Budget, advise, check_spill, decide, read_budget
from .scheduler import (
    MODEL_SPECS,
    Plan,
    describe as describe_plan,
    kv_bytes_per_token,
    max_ctx_that_fits,
    ollama_options,
    plan_run,
    weight_bytes,
)

__all__ = [
    "Budget", "read_budget", "decide", "check_spill", "advise",
    "MODEL_SPECS", "Plan", "plan_run", "describe_plan", "ollama_options",
    "max_ctx_that_fits", "kv_bytes_per_token", "weight_bytes",
    "BenchResult", "bench_generate", "bench_suite", "format_table",
]

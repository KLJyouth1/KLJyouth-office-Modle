"""评测层：幻觉与可靠性度量（YOLO 式严谨指标）。"""
from .hallucination import Case, Report, default_cases, evaluate

__all__ = ["Case", "Report", "evaluate", "default_cases"]

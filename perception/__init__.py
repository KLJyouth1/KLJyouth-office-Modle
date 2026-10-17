"""感知层：屏幕截取与 Windows UI 元素树。

低运行环境设计：
- 截屏用 mss（无 GUI 依赖、极轻量）。
- 元素定位优先走系统原生 UI 自动化树（零模型开销），仅在其不足时才回退到 VLM。
- 全程 DPI 感知，保证"看到的坐标"与"点的坐标"一致。
"""
from .screen import make_dpi_aware, capture, capture_region, list_monitors, capture_window
from .ui_tree import UIElement, dump_elements, find_element, describe_for_model

__all__ = [
    "make_dpi_aware",
    "capture",
    "capture_region",
    "list_monitors",
    "capture_window",
    "UIElement",
    "dump_elements",
    "find_element",
    "describe_for_model",
]

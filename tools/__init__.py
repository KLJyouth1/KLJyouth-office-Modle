"""工具层聚合入口：按领域装配所有工具到注册表。"""
from __future__ import annotations

from .base import FuncTool, Tool, ToolRegistry, ToolResult
from . import blender, browser, code, media, office, system

__all__ = ["Tool", "ToolResult", "ToolRegistry", "FuncTool", "build_registry"]


def build_registry(headless: bool = True, domains: tuple[str, ...] | None = None) -> ToolRegistry:
    """装配工具注册表。

    domains=None 表示装配全部领域；否则只装指定领域（减少给模型的干扰项）。
    """
    reg = ToolRegistry()

    def want(d: str) -> bool:
        return domains is None or d in domains

    if want("system"):
        reg.extend(system.build_tools())
    if want("browser"):
        reg.extend(browser.build_tools(headless=headless))
    if want("office"):
        reg.extend(office.build_tools())
    if want("code"):
        reg.extend(code.build_tools())
    if want("media"):
        reg.extend(media.build_tools())
    if want("blender"):
        reg.extend(blender.build_tools())
    return reg

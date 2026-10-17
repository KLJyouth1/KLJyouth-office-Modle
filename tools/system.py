"""系统级操作工具：键鼠、剪贴板、截屏。

中文输入关键点：pyautogui 无法键入中文。这里对非 ASCII 文本自动走
"写剪贴板 + Ctrl+V"，保证中文/符号输入可靠。
"""
from __future__ import annotations

from ..perception.screen import capture
from ..runtime.llm import encode_image  # noqa: F401  (保留依赖关系清晰)
from .base import FuncTool, ToolResult, Tool
import pyautogui

pyautogui.FAILSAFE = True  # 鼠标挪到屏幕左上角可紧急中止
pyautogui.PAUSE = 0.05


def _set_clipboard(text: str) -> None:
    import win32clipboard

    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_UNICODETEXT, text)
    finally:
        win32clipboard.CloseClipboard()


def _is_ascii(text: str) -> bool:
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def sys_click(x: int, y: int, button: str = "left", clicks: int = 1) -> ToolResult:
    pyautogui.click(x=int(x), y=int(y), button=button, clicks=int(clicks))
    return ToolResult(True, f"已在 ({x},{y}) 点击 {clicks} 次", {"x": x, "y": y})


def sys_move(x: int, y: int) -> ToolResult:
    pyautogui.moveTo(int(x), int(y))
    return ToolResult(True, f"鼠标移动到 ({x},{y})")


def sys_type(text: str, interval: float = 0.02) -> ToolResult:
    text = str(text)
    if _is_ascii(text):
        pyautogui.typewrite(text, interval=interval)
    else:  # 中文/emoji 等走剪贴板粘贴
        _set_clipboard(text)
        pyautogui.hotkey("ctrl", "v")
    return ToolResult(True, f"已输入 {len(text)} 个字符")


def sys_hotkey(keys: list[str] | str) -> ToolResult:
    combo = keys if isinstance(keys, list) else [k.strip() for k in str(keys).split("+")]
    pyautogui.hotkey(*combo)
    return ToolResult(True, f"已按下组合键 {'+'.join(combo)}")


def sys_press(key: str, presses: int = 1) -> ToolResult:
    pyautogui.press(key, presses=int(presses))
    return ToolResult(True, f"已按 {key} ×{presses}")


def sys_scroll(amount: int, x: int | None = None, y: int | None = None) -> ToolResult:
    if x is not None and y is not None:
        pyautogui.moveTo(int(x), int(y))
    pyautogui.scroll(int(amount))
    return ToolResult(True, f"已滚动 {amount}")


def sys_screenshot(monitor: int = 1) -> ToolResult:
    path = capture(int(monitor))
    return ToolResult(True, f"截图已保存: {path}", {"path": path})


def build_tools() -> list[Tool]:
    d = "system"
    return [
        FuncTool("sys_click", "在屏幕坐标处点击鼠标", sys_click,
                 {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"},
                  "button": {"type": "string"}, "clicks": {"type": "integer"}}, "required": ["x", "y"]}, d),
        FuncTool("sys_move", "移动鼠标到屏幕坐标", sys_move,
                 {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                  "required": ["x", "y"]}, d),
        FuncTool("sys_type", "在当前焦点处输入文本(自动支持中文)", sys_type,
                 {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}, d),
        FuncTool("sys_hotkey", "按下组合键，如 ['ctrl','s']", sys_hotkey,
                 {"type": "object", "properties": {"keys": {"type": "array", "items": {"type": "string"}}},
                  "required": ["keys"]}, d),
        FuncTool("sys_press", "按下单个按键，如 enter/esc/tab", sys_press,
                 {"type": "object", "properties": {"key": {"type": "string"}, "presses": {"type": "integer"}},
                  "required": ["key"]}, d),
        FuncTool("sys_scroll", "滚动屏幕(正数向上/负数向下)", sys_scroll,
                 {"type": "object", "properties": {"amount": {"type": "integer"}, "x": {"type": "integer"},
                  "y": {"type": "integer"}}, "required": ["amount"]}, d),
        FuncTool("sys_screenshot", "截取屏幕并保存，返回图片路径", sys_screenshot,
                 {"type": "object", "properties": {"monitor": {"type": "integer"}}}, d),
    ]

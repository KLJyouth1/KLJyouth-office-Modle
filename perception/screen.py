"""屏幕截取模块。

DPI 关键点：Windows 缩放(如 125%/150%)会导致"逻辑坐标"与"物理像素"不一致。
本模块在进程启动时声明 DPI 感知，使 mss 抓取的物理像素坐标与 pyautogui 的
点击坐标处于同一坐标系，避免"看到的位置"和"点到的地方"错位。
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import mss
from PIL import Image

from ..config import RUN_DIR

_DPI_SET = False


def make_dpi_aware() -> None:
    """声明进程 DPI 感知（幂等）。应在任何截屏/点击前调用一次。"""
    global _DPI_SET
    if _DPI_SET:
        return
    try:
        # 2 = PROCESS_PER_MONITOR_DPI_AWARE
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    _DPI_SET = True


def list_monitors() -> list[dict]:
    """列出显示器。索引 0=虚拟全屏，1..=各物理显示器。"""
    make_dpi_aware()
    with mss.mss() as sct:
        return list(sct.monitors)


def _grab(region: dict, path: str | Path | None = None) -> str:
    make_dpi_aware()
    with mss.mss() as sct:
        shot = sct.grab(region)
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    if path is None:
        path = RUN_DIR / "last_shot.png"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return str(path)


def capture(monitor: int = 0, path: str | Path | None = None) -> str:
    """抓取整块显示器。monitor=0 为所有显示器拼接。"""
    make_dpi_aware()
    with mss.mss() as sct:
        region = sct.monitors[monitor]
    return _grab(region, path)


def capture_region(left: int, top: int, width: int, height: int, path: str | Path | None = None) -> str:
    """抓取屏幕指定矩形区域。"""
    return _grab({"left": left, "top": top, "width": width, "height": height}, path)


def capture_window(title_substr: str, path: str | Path | None = None) -> str:
    """按窗口标题关键字截取该窗口。返回图片路径。"""
    import win32gui

    matches: list[tuple[int, str]] = []

    def _cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if title and title_substr.lower() in title.lower():
            matches.append((hwnd, title))

    win32gui.EnumWindows(_cb, None)
    if not matches:
        raise LookupError(f"未找到标题包含 {title_substr!r} 的可见窗口")

    hwnd, title = matches[0]
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    return _grab({"left": left, "top": top, "width": max(1, right - left), "height": max(1, bottom - top)}, path)

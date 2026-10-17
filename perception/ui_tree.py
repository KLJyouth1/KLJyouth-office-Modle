"""Windows UI 元素树感知。

思路：屏幕上绝大多数可交互元素都在系统原生 UI 自动化(UIA)树里有名字与坐标。
直接读树做定位，零模型算力开销、零幻觉，是低配机器上最划算的"接地(grounding)"手段。
只有当 UIA 树缺失(游戏、自绘界面、网页画布)时才回退到视觉 VLM。
"""
from __future__ import annotations

from dataclasses import dataclass

from .screen import make_dpi_aware


@dataclass
class UIElement:
    name: str
    control_type: str
    left: int
    top: int
    right: int
    bottom: int
    window: str = ""

    @property
    def center(self) -> tuple[int, int]:
        return ((self.left + self.right) // 2, (self.top + self.bottom) // 2)

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_dict(self) -> dict:
        cx, cy = self.center
        return {
            "name": self.name,
            "type": self.control_type,
            "center": [cx, cy],
            "rect": [self.left, self.top, self.right, self.bottom],
            "window": self.window,
        }


def _from_info(info, window_title: str) -> UIElement | None:
    try:
        r = info.rectangle
    except Exception:
        return None
    return UIElement(
        name=(info.name or "").strip(),
        control_type=str(info.control_type or ""),
        left=int(r.left),
        top=int(r.top),
        right=int(r.right),
        bottom=int(r.bottom),
        window=window_title,
    )


def dump_elements(
    window_title_substr: str | None = None,
    *,
    max_depth: int = 5,
    max_windows: int = 6,
    limit: int = 600,
) -> list[UIElement]:
    """遍历可见窗口，导出 UI 元素(含坐标)。

    window_title_substr: 只看标题包含该关键字的窗口；None=前 max_windows 个窗口。
    max_depth: 树的遍历深度(越大越全但越慢)。
    """
    from pywinauto import Desktop

    make_dpi_aware()
    desktop = Desktop(backend="uia")

    try:
        wins = desktop.windows()
    except Exception:
        return []

    if window_title_substr:
        key = window_title_substr.lower()
        wins = [w for w in wins if key in (w.window_text() or "").lower()]

    out: list[UIElement] = []
    for w in wins[:max_windows]:
        try:
            title = w.window_text() or ""
        except Exception:
            continue
        top = _from_info(w.element_info, title)
        if top:
            out.append(top)
        try:
            for d in w.descendants(depth=max_depth):
                el = _from_info(d.element_info, title)
                if el:
                    out.append(el)
                if len(out) >= limit:
                    break
        except Exception:
            pass
        if len(out) >= limit:
            break
    return out


def find_element(
    text: str,
    elements: list[UIElement] | None = None,
    *,
    control_type: str | None = None,
    prefer_small: bool = True,
) -> UIElement | None:
    """按文本查找最可能的可点击元素。

    打分规则：精确匹配 > 包含匹配；同等条件下取面积较小者(更可能是按钮本身而非容器)。
    """
    els = elements if elements is not None else dump_elements()
    key = text.strip().lower()
    if not key:
        return None

    best: UIElement | None = None
    best_score = -1.0
    for el in els:
        if not el.name:
            continue
        if control_type and control_type.lower() not in el.control_type.lower():
            continue
        name = el.name.lower()
        if key == name:
            score = 1000.0
        elif key in name:
            score = 500.0 + len(key) / max(1, len(name))
        elif name in key:
            score = 300.0
        else:
            continue
        if el.area == 0:
            score -= 100.0
        if prefer_small and el.area > 0:
            # 面积越大扣分越多，避免选中整块容器
            score -= el.area / 1_000_000.0
        if score > best_score:
            best_score = score
            best = el
    return best


def describe_for_model(elements: list[UIElement], limit: int = 60) -> str:
    """把 UI 元素压缩成给模型看的紧凑文本清单。"""
    lines: list[str] = []
    for i, el in enumerate(elements[:limit]):
        if el.area == 0:
            continue
        cx, cy = el.center
        label = el.name[:40] if el.name else "(无标题)"
        lines.append(f"[{i}] {el.control_type or '?'} \"{label}\" @({cx},{cy}) in \"{el.window[:30]}\"")
    return "\n".join(lines)

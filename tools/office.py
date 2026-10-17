"""办公软件工具（Windows COM 自动化）。

依托本机已安装的 Microsoft Office，通过 COM 直接操纵 Word/Excel/PowerPoint。
全部原地操作文件，无界面依赖（Visible=False），符合低运行环境。
"""
from __future__ import annotations

from pathlib import Path

from .base import FuncTool, Tool, ToolResult


def _co_init() -> None:
    try:
        import pythoncom

        pythoncom.CoInitialize()
    except Exception:
        pass


def _safe_quit(app) -> None:
    """容错退出 Office 应用。

    实测教训：Word 关闭时可能抛 RPC 失败、PowerPoint 的 Application 未必有 Quit()。
    这些"收尾动作"失败不应该让一个已经成功写盘的操作被判定为失败。
    """
    try:
        app.Quit()
    except Exception:
        pass


def _safe_close(obj, save_changes: int = 0) -> None:
    try:
        obj.Close(save_changes)
    except Exception:
        try:
            obj.Close()
        except Exception:
            pass


def _col_letter(idx: int) -> str:
    """1 -> A, 27 -> AA"""
    s = ""
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        s = chr(65 + rem) + s
    return s


# ---------------------------------------------------------------------- Excel
def excel_write(path: str, rows: list[list], sheet: int = 1, start_cell: str = "A1") -> ToolResult:
    _co_init()
    import win32com.client as win32

    p = Path(path).resolve()
    app = win32.DispatchEx("Excel.Application")
    app.Visible = False
    app.DisplayAlerts = False
    try:
        if p.exists():
            wb = app.Workbooks.Open(str(p))
        else:
            wb = app.Workbooks.Add()
            p.parent.mkdir(parents=True, exist_ok=True)
            wb.SaveAs(str(p))
        ws = wb.Worksheets(sheet)
        anchor = ws.Range(start_cell)
        r0, c0 = anchor.Row, anchor.Column
        for ri, row in enumerate(rows):
            for ci, val in enumerate(row):
                ws.Cells(r0 + ri, c0 + ci).Value = val
        wb.Save()
        _safe_close(wb, 0)
        return ToolResult(True, f"已写入 {len(rows)} 行到 {p.name}:{start_cell}", {"path": str(p)})
    finally:
        _safe_quit(app)


def excel_read(path: str, sheet: int = 1, max_rows: int = 50) -> ToolResult:
    _co_init()
    import win32com.client as win32

    p = Path(path).resolve()
    app = win32.DispatchEx("Excel.Application")
    app.Visible = False
    app.DisplayAlerts = False
    try:
        wb = app.Workbooks.Open(str(p))
        ws = wb.Worksheets(sheet)
        used = ws.UsedRange
        nrow = min(int(used.Rows.Count), int(max_rows))
        ncol = int(used.Columns.Count)
        out: list[list] = []
        for r in range(1, nrow + 1):
            out.append([ws.Cells(r, c).Value for c in range(1, ncol + 1)])
        wb.Close(SaveChanges=False)
        return ToolResult(True, f"读取 {nrow}×{ncol}", {"rows": out})
    finally:
        _safe_quit(app)


# ---------------------------------------------------------------------- Word
def word_create(path: str, paragraphs: list[str], title: str | None = None) -> ToolResult:
    _co_init()
    import win32com.client as win32

    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    app = win32.DispatchEx("Word.Application")
    app.Visible = False
    try:
        doc = app.Documents.Add()
        if title:
            doc.Content.InsertAfter(title + "\n")
        for para in paragraphs:
            doc.Content.InsertAfter(para + "\n")
        doc.SaveAs(str(p))
        _safe_close(doc, 0)
        return ToolResult(True, f"已生成 Word 文档: {p}", {"path": str(p)})
    finally:
        _safe_quit(app)


def word_read(path: str) -> ToolResult:
    """读回 Word 文档正文（用于校验写入结果）。"""
    _co_init()
    import win32com.client as win32

    p = Path(path).resolve()
    app = win32.DispatchEx("Word.Application")
    app.Visible = False
    app.DisplayAlerts = 0
    try:
        doc = app.Documents.Open(str(p))
        text = doc.Content.Text
        _safe_close(doc, 0)
        return ToolResult(True, text[:4000], {"length": len(text), "path": str(p)})
    finally:
        _safe_quit(app)


# ---------------------------------------------------------------------- PowerPoint
def ppt_create(path: str, slides: list[dict]) -> ToolResult:
    """slides: [{"title": "...", "body": "..."}]"""
    _co_init()
    import win32com.client as win32

    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    app = win32.DispatchEx("PowerPoint.Application")
    try:
        pres = app.Presentations.Add()
        for s in slides:
            slide = pres.Slides.Add(pres.Slides.Count + 1, 2)  # 2 = ppLayoutText
            if s.get("title"):
                slide.Shapes.Title.TextFrame.TextRange.Text = str(s["title"])
            if s.get("body"):
                slide.Shapes.Placeholders(2).TextFrame.TextRange.Text = str(s["body"])
        pres.SaveAs(str(p))
        _safe_close(pres)
        return ToolResult(True, f"已生成 PPT({len(slides)}页): {p}", {"path": str(p), "slides": len(slides)})
    finally:
        _safe_quit(app)


def build_tools() -> list[Tool]:
    d = "office"
    obj = lambda props, req=None: {"type": "object", "properties": props, "required": req or []}  # noqa: E731
    return [
        FuncTool("excel_write", "向 Excel 写入二维数据(rows=[[..],[..]])", excel_write,
                 obj({"path": {"type": "string"}, "rows": {"type": "array"}, "sheet": {"type": "integer"},
                      "start_cell": {"type": "string"}}, ["path", "rows"]), d),
        FuncTool("excel_read", "读取 Excel 前若干行数据", excel_read,
                 obj({"path": {"type": "string"}, "sheet": {"type": "integer"}, "max_rows": {"type": "integer"}},
                     ["path"]), d),
        FuncTool("word_create", "创建 Word 文档并写入多段文本", word_create,
                 obj({"path": {"type": "string"}, "paragraphs": {"type": "array", "items": {"type": "string"}},
                      "title": {"type": "string"}}, ["path", "paragraphs"]), d),
        FuncTool("word_read", "读取 Word 文档正文(用于校验)", word_read,
                 obj({"path": {"type": "string"}}, ["path"]), d),
        FuncTool("ppt_create", "创建 PPT(每页 title/body)", ppt_create,
                 obj({"path": {"type": "string"}, "slides": {"type": "array"}}, ["path", "slides"]), d),
    ]

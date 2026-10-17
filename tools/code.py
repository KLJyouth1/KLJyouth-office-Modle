"""代码工程工具：执行命令、跑脚本、git 查询。

用于"代码工程"领域的自动化：构建、测试、运行、查看版本状态。
命令在子进程中执行并捕获输出，超时可控，不会挂死决策循环。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ..config import WORKSPACE
from .base import FuncTool, Tool, ToolResult


def _run(cmd: list[str], cwd: str | None, timeout: int) -> ToolResult:
    try:
        proc = subprocess.run(
            cmd, cwd=cwd or str(WORKSPACE), capture_output=True, text=True,
            timeout=int(timeout), encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return ToolResult(False, f"命令超时({timeout}s): {' '.join(cmd)}")
    except FileNotFoundError as e:
        return ToolResult(False, f"命令不存在: {e}")
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    tail = out[-4000:]
    if err:
        tail += "\n[stderr]\n" + err[-1500:]
    return ToolResult(proc.returncode == 0, f"exit={proc.returncode}\n{tail}",
                      {"returncode": proc.returncode, "stdout": out, "stderr": err})


def code_run(command: str, cwd: str | None = None, timeout: int = 120) -> ToolResult:
    return _run(["powershell", "-NoProfile", "-Command", command], cwd, timeout)


# 让 code_python 具备 REPL 语义：若最后一句是表达式，自动求值并打印。
# 修复的真实缺陷：用户/模型写 `12345*67` 时没有任何 print，旧实现只会返回 "exit=0"
# （空输出），模型因此只能把 "exit=0" 当作答案复述。现在会返回真实计算结果。
_REPL_WRAPPER = """
import ast, sys
_code = open(sys.argv[1], encoding='utf-8').read()
_tree = ast.parse(_code)
_body = list(_tree.body)
_last = None
if _body and isinstance(_body[-1], ast.Expr):
    _last = _body.pop()
if _body:
    exec(compile(ast.Module(_body, []), '<omni>', 'exec'), globals())
if _last is not None:
    _v = eval(compile(ast.Expression(_last.value), '<omni>', 'eval'), globals())
    if _v is not None:
        print(_v if isinstance(_v, str) else repr(_v))
"""


def code_python(code: str, cwd: str | None = None, timeout: int = 120) -> ToolResult:
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.gettempdir()) / "omni_snippet.py"
    tmp.write_text(code, encoding="utf-8")
    return _run([sys.executable, "-c", _REPL_WRAPPER, str(tmp)], cwd, timeout)


def git_status(cwd: str | None = None) -> ToolResult:
    return _run(["git", "status", "--short"], cwd, 30)


def git_log(cwd: str | None = None, n: int = 10) -> ToolResult:
    return _run(["git", "log", f"-{n}", "--oneline"], cwd, 30)


def file_write(path: str, content: str, append: bool = False) -> ToolResult:
    """把文本写入文件（自动建目录）。

    为什么单列一个工具：实测把"写文件"交给 code_python 时，小模型生成的代码经常被
    截断/写错（`ast.parse` 直接失败），多步任务因此卡死。让模型只填 path + content，
    成功率天差地别。
    """
    p = Path(path)
    if not p.is_absolute():
        p = WORKSPACE / p
    p.parent.mkdir(parents=True, exist_ok=True)
    if append:
        with p.open("a", encoding="utf-8") as f:
            f.write(content)
    else:
        p.write_text(content, encoding="utf-8")
    return ToolResult(True, f"已{'追加' if append else '写入'} {len(content)} 字到 {p}",
                      {"path": str(p), "chars": len(content), "bytes": len(content.encode('utf-8'))})


def file_read(path: str, limit: int = 4000) -> ToolResult:
    """读取文本文件内容（用于校验/后续处理）。"""
    p = Path(path)
    if not p.is_absolute():
        p = WORKSPACE / p
    if not p.exists():
        return ToolResult(False, f"文件不存在：{p}")
    txt = p.read_text(encoding="utf-8", errors="replace")[: int(limit or 4000)]
    return ToolResult(True, txt, {"path": str(p), "length": len(txt)})


def build_tools() -> list[Tool]:
    d = "code"
    obj = lambda props, req=None: {"type": "object", "properties": props, "required": req or []}  # noqa: E731
    return [
        FuncTool("file_write", "把文本写入文件(自动建目录；append=true 追加)", file_write,
                 obj({"path": {"type": "string"}, "content": {"type": "string"}, "append": {"type": "boolean"}},
                     ["path", "content"]), d),
        FuncTool("file_read", "读取文本文件内容", file_read,
                 obj({"path": {"type": "string"}, "limit": {"type": "integer"}}, ["path"]), d),
        FuncTool("code_run", "在 PowerShell 中执行命令并返回输出", code_run,
                 obj({"command": {"type": "string"}, "cwd": {"type": "string"}, "timeout": {"type": "integer"}},
                     ["command"]), d),
        FuncTool("code_python", "执行 Python 代码片段并返回输出", code_python,
                 obj({"code": {"type": "string"}, "cwd": {"type": "string"}, "timeout": {"type": "integer"}},
                     ["code"]), d),
        FuncTool("git_status", "查看 git 工作区变更", git_status, obj({"cwd": {"type": "string"}}), d),
        FuncTool("git_log", "查看最近提交记录", git_log,
                 obj({"cwd": {"type": "string"}, "n": {"type": "integer"}}), d),
    ]

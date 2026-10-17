"""3D 建模工具（Blender 无头自动化）。

本机当前未安装 Blender；本模块会自动探测常见安装路径，
未找到时返回明确的安装指引而不是崩溃，保证决策循环继续。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .base import FuncTool, Tool, ToolResult


def find_blender() -> str | None:
    env = os.environ.get("BLENDER_PATH")
    if env and Path(env).exists():
        return env
    which = shutil.which("blender")
    if which:
        return which
    for base in (Path(r"C:\Program Files\Blender Foundation"), Path(r"D:\Blender Foundation")):
        if base.exists():
            for exe in sorted(base.glob("Blender */blender.exe"), reverse=True):
                return str(exe)
    return None


def blender_run_script(script: str, blend_file: str | None = None, timeout: int = 600) -> ToolResult:
    """在 Blender 后台模式执行 Python 脚本(bpy API)。"""
    exe = find_blender()
    if not exe:
        return ToolResult(False, "未找到 Blender。请安装 Blender 并设置环境变量 BLENDER_PATH，或加入 PATH。")

    tmp = Path(tempfile.gettempdir()) / "omni_blender_script.py"
    tmp.write_text(script, encoding="utf-8")
    cmd = [exe, "-b"]
    if blend_file:
        cmd.append(blend_file)
    cmd += ["-P", str(tmp)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return ToolResult(False, f"Blender 执行超时({timeout}s)")
    tail = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()[-3000:]
    return ToolResult(proc.returncode == 0, f"blender exit={proc.returncode}\n{tail}")


def blender_render(blend_file: str, output: str, frame: int = 1, timeout: int = 900) -> ToolResult:
    """渲染指定 .blend 场景的某一帧到图片。"""
    exe = find_blender()
    if not exe:
        return ToolResult(False, "未找到 Blender。请安装后设置 BLENDER_PATH。")
    out = str(Path(output).resolve())
    tmpl = out
    cmd = [exe, "-b", blend_file, "-o", tmpl, "-f", str(int(frame))]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return ToolResult(False, "Blender 渲染超时")
    tail = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()[-2000:]
    return ToolResult(proc.returncode == 0, f"渲染 exit={proc.returncode}\n{tail}", {"output": out})


def blender_available() -> ToolResult:
    exe = find_blender()
    if exe:
        return ToolResult(True, f"Blender 可用: {exe}", {"path": exe})
    return ToolResult(False, "Blender 未安装（3D 领域需先补齐该依赖）")


def build_tools() -> list[Tool]:
    d = "blender"
    obj = lambda props, req=None: {"type": "object", "properties": props, "required": req or []}  # noqa: E731
    return [
        FuncTool("blender_run_script", "在 Blender 无头模式执行 bpy Python 脚本", blender_run_script,
                 obj({"script": {"type": "string"}, "blend_file": {"type": "string"}, "timeout": {"type": "integer"}},
                     ["script"]), d),
        FuncTool("blender_render", "渲染 .blend 场景某帧到图片", blender_render,
                 obj({"blend_file": {"type": "string"}, "output": {"type": "string"}, "frame": {"type": "integer"}},
                     ["blend_file", "output"]), d),
        FuncTool("blender_available", "检测 Blender 是否可用", blender_available, obj({}), d),
    ]

"""视频与图片编辑工具。

- 视频：调用 ffmpeg（本机 PATH 已有）做转码/裁剪/抽音/抽帧。
- 图片：Pillow 做缩放/裁剪/格式转换。
均为命令行/库调用，无 GUI，低开销。
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .base import FuncTool, Tool, ToolResult

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


def _ff(args: list[str], timeout: int = 600) -> ToolResult:
    try:
        proc = subprocess.run([FFMPEG, "-y", *args], capture_output=True, text=True,
                              timeout=timeout, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ToolResult(False, "未找到 ffmpeg，请先安装并加入 PATH")
    except subprocess.TimeoutExpired:
        return ToolResult(False, f"ffmpeg 超时({timeout}s)")
    ok = proc.returncode == 0
    tail = (proc.stderr or "").strip()[-1200:]
    return ToolResult(ok, f"ffmpeg exit={proc.returncode}\n{tail}")


def media_convert(input: str, output: str, extra_args: list[str] | None = None) -> ToolResult:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    return _ff(["-i", input, *(extra_args or []), output])


def media_trim(input: str, output: str, start: float, duration: float) -> ToolResult:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    return _ff(["-ss", str(start), "-i", input, "-t", str(duration), "-c", "copy", output])


def media_extract_audio(input: str, output: str) -> ToolResult:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    return _ff(["-i", input, "-vn", "-acodec", "libmp3lame", "-q:a", "2", output])


def media_extract_frames(input: str, output_dir: str, fps: float = 1.0) -> ToolResult:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = _ff(["-i", input, "-vf", f"fps={fps}", str(out / "frame_%03d.png")])
    if res.ok:
        n = len(list(out.glob("frame_*.png")))
        return ToolResult(True, f"已抽取 {n} 帧到 {out}", {"dir": str(out), "count": n})
    return res


def media_probe(input: str) -> ToolResult:
    try:
        proc = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams",
                               "-of", "json", input], capture_output=True, text=True,
                              timeout=60, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ToolResult(False, "未找到 ffprobe")
    if proc.returncode != 0:
        return ToolResult(False, f"ffprobe 失败: {proc.stderr[:400]}")
    import json as _json

    info = _json.loads(proc.stdout)
    fmt = info.get("format", {})
    return ToolResult(True, f"时长 {fmt.get('duration')}s, 格式 {fmt.get('format_name')}",
                      {"format": fmt, "streams": info.get("streams", [])})


# ---------------------------------------------------------------------- 图片
def image_resize(input: str, output: str, width: int, height: int) -> ToolResult:
    from PIL import Image

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Image.open(input) as im:
        im.resize((int(width), int(height))).save(output)
    return ToolResult(True, f"已缩放到 {width}x{height}: {output}", {"path": output})


def image_crop(input: str, output: str, left: int, top: int, right: int, bottom: int) -> ToolResult:
    from PIL import Image

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Image.open(input) as im:
        im.crop((int(left), int(top), int(right), int(bottom))).save(output)
    return ToolResult(True, f"已裁剪: {output}", {"path": output})


def image_info(input: str) -> ToolResult:
    from PIL import Image

    with Image.open(input) as im:
        return ToolResult(True, f"{im.size[0]}x{im.size[1]} {im.mode} {im.format}",
                          {"width": im.size[0], "height": im.size[1], "mode": im.mode, "format": im.format})


def build_tools() -> list[Tool]:
    d = "media"
    obj = lambda props, req=None: {"type": "object", "properties": props, "required": req or []}  # noqa: E731
    return [
        FuncTool("video_convert", "用 ffmpeg 转码视频(可附额外参数)", media_convert,
                 obj({"input": {"type": "string"}, "output": {"type": "string"},
                      "extra_args": {"type": "array", "items": {"type": "string"}}}, ["input", "output"]), d),
        FuncTool("video_trim", "按起点与时长裁剪视频", media_trim,
                 obj({"input": {"type": "string"}, "output": {"type": "string"}, "start": {"type": "number"},
                      "duration": {"type": "number"}}, ["input", "output", "start", "duration"]), d),
        FuncTool("video_extract_audio", "从视频抽取音频(mp3)", media_extract_audio,
                 obj({"input": {"type": "string"}, "output": {"type": "string"}}, ["input", "output"]), d),
        FuncTool("video_extract_frames", "按帧率抽取视频帧为图片", media_extract_frames,
                 obj({"input": {"type": "string"}, "output_dir": {"type": "string"}, "fps": {"type": "number"}},
                     ["input", "output_dir"]), d),
        FuncTool("video_probe", "读取视频/音频元信息", media_probe, obj({"input": {"type": "string"}}, ["input"]), d),
        FuncTool("image_resize", "缩放图片", image_resize,
                 obj({"input": {"type": "string"}, "output": {"type": "string"}, "width": {"type": "integer"},
                      "height": {"type": "integer"}}, ["input", "output", "width", "height"]), d),
        FuncTool("image_crop", "裁剪图片", image_crop,
                 obj({"input": {"type": "string"}, "output": {"type": "string"}, "left": {"type": "integer"},
                      "top": {"type": "integer"}, "right": {"type": "integer"}, "bottom": {"type": "integer"}},
                     ["input", "output", "left", "top", "right", "bottom"]), d),
        FuncTool("image_info", "读取图片尺寸/格式", image_info, obj({"input": {"type": "string"}}, ["input"]), d),
    ]

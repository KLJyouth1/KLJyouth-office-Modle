"""把 LoRA 适配器导出为 Ollama 可热插拔模型。

这是"定向训练"闭环的最后一环：训练得到的专属适配器，无需改任何推理代码，
直接作为新的 Ollama 模型名被 OmniPilot 调用（按领域热切换）。

两条路径（按可用性自动选择）：
A. Ollama 直接加载 Safetensors LoRA 适配器目录（新版 Ollama 支持）。
B. 若本机有 llama.cpp，则先把 LoRA 转成 GGUF，再交给 Ollama 加载（兼容性最好）。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from ..config import WORKSPACE
from ..runtime.llm import OllamaClient


def find_llama_cpp() -> Path | None:
    """查找 llama.cpp 的 LoRA->GGUF 转换脚本。"""
    candidates = [
        WORKSPACE / "llama.cpp" / "convert_lora_to_gguf.py",
        Path(r"C:\llama.cpp\convert_lora_to_gguf.py"),
        Path(r"D:\llama.cpp\convert_lora_to_gguf.py"),
    ]
    for c in candidates:
        if c.exists():
            return c
    which = shutil.which("convert_lora_to_gguf.py")
    return Path(which) if which else None


def to_ollama(
    name: str,
    base_model: str,
    adapter_dir: str | Path,
    *,
    system: str | None = None,
    llama_cpp_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """注册专属模型到 Ollama。返回 {'ok':bool,'method':str,'message':str}。"""
    adapter = Path(adapter_dir)
    if not adapter.exists():
        return {"ok": False, "method": "none", "message": f"适配器目录不存在: {adapter}"}

    client = OllamaClient()
    if not client.ping():
        return {"ok": False, "method": "none", "message": "Ollama 未运行，请先启动"}

    # 路径 B：先转 GGUF（若 llama.cpp 可用），兼容性最好
    conv = None
    if llama_cpp_dir:
        p = Path(llama_cpp_dir) / "convert_lora_to_gguf.py"
        conv = p if p.exists() else None
    if conv is None:
        conv = find_llama_cpp()

    if conv is not None:
        gguf = adapter / "adapter.gguf"
        cmd = [sys.executable, str(conv), "--base-model-id", base_model.split(":")[0],
               "--outfile", str(gguf), "--outtype", "f16", str(adapter)]
        if verbose:
            print(f"[export] 用 llama.cpp 转换 LoRA -> GGUF: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode == 0 and gguf.exists():
            client.create_from_adapter(name, base_model, gguf, system=system)
            return {"ok": True, "method": "llama.cpp->gguf", "message": f"已创建模型 {name}（GGUF 适配器）"}
        if verbose:
            print("[export] llama.cpp 转换失败，回退 Safetensors 路径：", (proc.stderr or "")[:300])

    # 路径 A：Ollama 直接吃 Safetensors 适配器目录
    try:
        client.create_from_adapter(name, base_model, adapter, system=system)
        return {"ok": True, "method": "ollama-safetensors", "message": f"已创建模型 {name}（Safetensors 适配器）"}
    except Exception as e:
        return {
            "ok": False, "method": "none",
            "message": f"导出失败：{e}。可安装 llama.cpp 后重试（路径 B 兼容性最好）。",
        }

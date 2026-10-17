"""定向训练流水线（惰性导入）。

闭环：示范采集(self-play/人工) → 合成专家轨迹 → 开源真实数据 → 数据集构建
      → LoRA 微调 → LoRA 合并 → GGUF 转换 → 注册为 Ollama 模型（热插拔）。

注意：这里刻意使用**惰性导入**（PEP 562 模块级 __getattr__）。
原因：训练常常跑在一个"只有 torch/transformers/peft"的精简环境里，
而 synth 依赖 agent/tools（会连带引入 mss/pyautogui/pywinauto 等 GUI 依赖）。
若在包初始化时就全部导入，精简环境将无法训练。惰性导入让"用什么才加载什么"。
"""
from __future__ import annotations

import importlib

_LAZY = {
    "SFTCollector": (".collect", "SFTCollector"),
    "build_dataset": (".dataset", "build_dataset"),
    "train_lora": (".qlora", "train"),
    "generate_synth": (".synth", "generate"),
    "generate_multistep": (".synth", "generate_multistep"),
    "generate_chat": (".chat_data", "generate_chat"),
    "to_ollama": (".export", "to_ollama"),
    "find_llama_cpp": (".export", "find_llama_cpp"),
    "merge_lora": (".merge", "merge_lora"),
    "build_gguf": (".merge", "build_gguf"),
    "register_ollama": (".merge", "register_ollama"),
    "ensure_chat_template": (".merge", "ensure_chat_template"),
    "CHATML_GO_TEMPLATE": (".merge", "CHATML_GO_TEMPLATE"),
    "convert_open_data": (".open_data", "convert_file"),
    "load_open_data": (".open_data", "load_json_records"),
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        mod = importlib.import_module(mod_name, __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)

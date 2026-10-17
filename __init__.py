"""OmniPilot —— 低运行环境需求、纯离线、复合架构的多模态决策智能体。

设计目标（对应需求）：
- 低运行环境需求：所有模型走 Ollama(GGUF 量化)，CPU 可跑，显存不足自动回退。
- 高质量运行：感知(屏幕/UI树/视觉) + 决策(规划) + 领域工具 + 验证闭环。
- 复合：浏览器 / 办公 / 代码 / 视频图片 / 3D 各领域工具可组合打通。
- 定向训练：示范采集 -> 数据集 -> LoRA 适配器 -> Ollama 热插拔，形成"独一无二"的专属模型。
"""

import sys as _sys
from pathlib import Path as _Path

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# 环境防遮蔽（已复现的真实缺陷）：
# 工作区根目录下存在 datasets/ tests/ docs/ 等与第三方库同名的目录。以工作区为
# 工作目录运行时，"空目录"会被 Python 识别为"命名空间包"，从而遮蔽真实库：
# 例如 transformers 会因 import datasets 命中本工作区的 datasets/ 目录而误判
# "datasets 可用"，进而在 isinstance(dataset, datasets.Dataset) 处崩溃。
# 这里把工作区根目录从 sys.path 中移除，从根上杜绝此类遮蔽。
# 注意：不影响 omnipilot 子模块导入（子模块通过包 __path__ 定位，不依赖 sys.path）。
_WORKSPACE_ROOT = _Path(__file__).resolve().parent.parent


def _is_workspace(path: str) -> bool:
    try:
        return _Path(path or ".").resolve() == _WORKSPACE_ROOT
    except Exception:
        return False


_sys.path[:] = [p for p in _sys.path if not _is_workspace(p)]

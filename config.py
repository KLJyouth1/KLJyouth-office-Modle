"""OmniPilot 全局配置。

低运行环境优先：模型全部交给 Ollama(GGUF 量化)托管，CPU 可运行、可选 GPU 卸载。
所有路径均以本文件位置为基准，不依赖运行时的工作目录。
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------- 路径 ----------------
PKG_DIR = Path(__file__).resolve().parent          # .../omnipilot
WORKSPACE = PKG_DIR.parent                          # 工作区根目录

DATA_DIR = PKG_DIR / "data"                         # 运行期数据
DEMO_DIR = DATA_DIR / "demos"                       # 人工示范采集
DATASET_DIR = DATA_DIR / "datasets"                 # 训练数据集
ADAPTER_DIR = DATA_DIR / "adapters"                 # LoRA 适配器
RUN_DIR = PKG_DIR / "runs"                          # 运行产物(截图/日志)

for _d in (DATA_DIR, DEMO_DIR, DATASET_DIR, ADAPTER_DIR, RUN_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------- 推理运行时 ----------------
# Ollama 默认监听 127.0.0.1:11434
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
REQUEST_TIMEOUT = float(os.environ.get("OMNIPILOT_TIMEOUT", "300"))

# ---------------- 模型 ----------------
# 决策/规划大脑：qwen3:4b（本地已有，2.5GB，CPU 可跑）
PLANNER_MODEL = os.environ.get("OMNIPILOT_PLANNER", "qwen3:4b")
# 视觉感知：moondream（本地已有，1.7GB，轻量 VLM）
VISION_MODEL = os.environ.get("OMNIPILOT_VISION", "moondream:latest")

# 决策模型：Jev 式 Choice/Noul/Score 的本地后端。
# 必须是"按字母约定训练过"的模型，否则首 token 会是候选原文、概率不可信（实测坑）。
DECISION_MODEL = os.environ.get("OMNIPILOT_DECISION", "omnipilot16")
# 路由策略：heuristic（只走关键词快路）| decision（只走决策层）| hybrid（默认：关键词未命中才走决策层）
ROUTER_MODE = os.environ.get("OMNIPILOT_ROUTER", "hybrid").lower()

# ---------------- 智能体循环 ----------------
MAX_STEPS = int(os.environ.get("OMNIPILOT_MAX_STEPS", "25"))
STEP_TIMEOUT = float(os.environ.get("OMNIPILOT_STEP_TIMEOUT", "300"))

# ---------------- 领域 ----------------
DOMAINS = ("browser", "office", "code", "media", "blender", "system")

# 各领域的一句话说明（唯一权威来源）。
# 同时被两处消费：1) agent/router.py 决策式路由的 choice criteria；
# 2) train/decision_data.py 构造路由训练/评测样本。放这里是为了避免两处文案漂移。
DOMAIN_DESC: dict[str, str] = {
    "browser": "网页浏览、检索、读网页",
    "office": "Word/Excel/PPT 等文档表格",
    "code": "代码、脚本、命令行、读写文件",
    "media": "视频/音频/图像处理",
    "blender": "3D 建模与渲染",
    "system": "桌面窗口、鼠标键盘、屏幕截图",
}

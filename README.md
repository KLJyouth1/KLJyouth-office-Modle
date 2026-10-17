# KLJyouth-office-Modle

**Marguerite 工作台的配套本地智能体模型仓库**：权重、训练代码、训练数据、评测集与 LoRA 适配器。
模型统一命名格式 **`KLJyouth-office-V<版本>-<变体>`**，当前发布版本 **`KLJyouth-office-V0.16-C`**。

> 本仓库不含工作台源码。源码见 [Marguerite](https://github.com/KLJyouth1/Marguerite)，
> 安装包见 [Marguerite-Releases](https://github.com/KLJyouth1/Marguerite-Releases)。

---

## 当前版本：KLJyouth-office-V0.16-C

| 项 | 值（来自 `data/adapters/omnipilot16c/train_stats.json` 与 `data/datasets/omnipilot16_info.json`） |
|---|---|
| 基座 | `Qwen2.5-1.5B-Instruct`（Qwen2ForCausalLM，hidden 1536 / 28 层 / 12 头，vocab 151,936，上下文 32,768） |
| 微调方式 | LoRA（PEFT），`data/adapters/omnipilot16c/adapter_model.safetensors`（16.65 MB） |
| 训练样本 | 3,781（另有 198 验证样本） |
| 训练轮数 | 2 epochs，train_loss **0.1019** |
| 数据种子 | `20261016`（可复现） |
| 发布产物 | `KLJyouth-office-V0.16-C-adapter.safetensors`（在 Releases） |

### 训练数据覆盖的 11 类能力（共 5,781 条，评测 1,060 条）

| 任务 | 总数 | 训练 / 评测 |
|---|---|---|
| `routing`（模型/工具路由） | 800 | 640 / 160 |
| `tool_choice`（工具选择） | 600 | 480 / 120 |
| `intent_massive` | 600 | 480 / 120 |
| `intent_clinc` | 500 | 400 / 100 |
| `emotion6` / `emotion28` | 600 / 400 | 480/120、320/80 |
| `fact_vitaminc` | 600 | 480 / 120 |
| `nli_mnli` / `nli_score` | 600 / 400 | 480/120、320/80 |
| `abstain`（拒答） | 100 | 80 / 20 |
| `grounded`（有据回答） | 100 | 80 / 20 |

## 历史权重（legacy，命名保留可追溯）

早期 0.5B / 1.5B 线的 fp16 合并权重仍在 Releases（`KLJyouth-office-legacy-*`）。
它们的**具体适配器变体未在构建记录中标注**，因此不冠以 V0.16-C 名义——只按可核实的基座与体量标注。
如果你的记录能确认某个 legacy 文件就是 16c 合并版，告诉我，我立刻改名。

| 文件 | 体积 | 说明 |
|---|---|---|
| `KLJyouth-office-legacy-0.5B-f16.gguf` | 948 MB | 0.5B 级 fp16 GGUF |
| `KLJyouth-office-legacy-0.5B-f16.safetensors` | 942 MB | 同一模型的 safetensors |
| `KLJyouth-office-legacy-0.5B-v2.7z.001/.002` | 2950 MB（展开） | 0.5B 线第二版 GGUF |
| `KLJyouth-office-legacy-1.5B.7z.001/.002` | 2950 MB（展开） | 1.5B 级 GGUF |
| `KLJyouth-office-legacy-1.5B-f16.safetensors.7z.001/.002` | 2944 MB（展开） | 1.5B 级 safetensors |
| `KLJyouth-office-legacy-base-1.5B.7z.001/.002` | 2950 MB（展开） | 1.5B 基座 GGUF（对比/再微调用） |

> GitHub Release 单资产上限 **2 GiB**，所以 >2 GiB 的权重以 7-Zip 分卷发布。
> 下载同前缀全部卷后，对第一卷解压：`7z x KLJyouth-office-legacy-1.5B.7z.001`。
> 每个资产都在 `SHA256SUMS.txt` 里。

## 仓库结构

```
train/             训练代码（qlora.py、decision_data.py 等 28 个文件）
eval/              评测台（decision_eval.py 等）
decision/          决策引擎与 schema
agent/ tools/      路由与工具层
data/datasets/     训练/验证/评测数据（40.1 MB，28 个文件）
data/adapters/     LoRA 适配器：omnipilot15 / 15b / 16 / 16b / 16c / 16d / sft
  omnipilot16c/    ← 即 KLJyouth-office-V0.16-C（adapter_model + tokenizer + chat_template + train_stats）
tests/             单元测试
```

> 仓库**不含** `**/ckpt/**/optimizer.pt`（训练优化器状态，每份 33.4 MB，可从 `train_stats.json`
> 与固定种子重跑复现）与 `__pycache__`。适配器权重、分词器、chat template、训练统计与数据集都在。

## 快速使用

### 方式一：Marguerite（最省事）
装 [Marguerite 0.2.13](https://github.com/KLJyouth1/Marguerite-Releases/releases/tag/v0.2.13)，
「设置 → 模型」指向本地 GGUF 即可。

### 方式二：自己合并 LoRA 并转 GGUF

```powershell
# 1) 合并（peft + transformers）：base = Qwen2.5-1.5B-Instruct
#    adapter = data/adapters/omnipilot16c
# 2) 导出 fp16 GGUF，再按需量化
# 3) ollama create ...
```

> 经验结论：量化务必**先转 GGUF 再量化**；直接从 safetensors 目录量化（Ollama 0.15.2 实测）
> 会产出崩溃模型。

## 许可

基座 **Qwen2.5-1.5B-Instruct** 为 Apache-2.0；本仓库的训练代码、数据与适配器沿用同一许可。
分发或商用前请自行核对基座模型与数据集（MASSIVE / CLINC150 / GoEmotions / MNLI / VitaminC 等）
各自的原始许可。

# OmniPilot

**Marguerite 工作台的配套本地智能体模型。** 目标是「低运行环境需求 + 工具调用正确」：
普通笔记本 CPU 即可运行，可完全离线，也可配合
[Marguerite](https://github.com/KLJyouth1/Marguerite) 使用。

> 本仓库是**模型仓库**（权重、分词器、Modelfile、说明），不含工作台源码。
> 源码见 [Marguerite](https://github.com/KLJyouth1/Marguerite)，
> 安装包见 [Marguerite-Releases](https://github.com/KLJyouth1/Marguerite-Releases)。

## 模型

| 模型 | 架构（来自 `config.json`） | fp16 体积 | 权重 |
|---|---|---|---|
| **OmniPilot 0.5B** | `Qwen2ForCausalLM`，hidden 896 / 24 层 / 14 头 / kv 2，vocab 151,936，上下文 32,768 | 948.1 MB（GGUF）<br>942.3 MB（safetensors） | `omnipilot-merged-f16.gguf`<br>`omnipilot-merged/model.safetensors` |
| **OmniPilot 1.5B** | `Qwen2ForCausalLM`，hidden 1536 / 28 层 / 12 头 / kv 2，vocab 151,936，上下文 32,768 | 2950.4 MB（GGUF）<br>2944.4 MB（safetensors） | `omnipilot15-f16.gguf`<br>`omnipilot15-merged/model.safetensors` |

另有两份同代产物：`omnipilot-merged-v2-f16.gguf`（0.5B 线第二版）与
`qwen2.5-1.5b-f16.gguf`（1.5B 线的**基座** fp16，便于对比与再微调）。

本仓库的 `omnipilot-0.5b/` 与 `omnipilot-1.5b/` 目录放了对应的
`config.json`、`generation_config.json`、`tokenizer.json`、`tokenizer_config.json`
与 `chat_template.jinja`——权重本身在 Releases。

### 分卷说明

GitHub Release 单资产上限 **2 GiB**。上述 >2 GiB 的文件以 **7-Zip 分卷**发布
（`*.7z.001` / `*.7z.002`）。下载同名全部卷后，对**第一卷**解压即可：

```powershell
7z x omnipilot15-f16.7z.001     # 会自动合并 .002 等后续卷
```

每个卷都附在 `SHA256SUMS.txt` 中。

## 快速使用

### 方式一：Marguerite 安装包（最简单）

装 [Marguerite](https://github.com/KLJyouth1/Marguerite-Releases/releases/latest)，
在「设置 → 模型」里指向本地 GGUF 即可，无需 Ollama。

### 方式二：Ollama

```powershell
ollama create omnipilot -f Modelfile/Modelfile.omnipilot_latest
ollama run omnipilot "把当前目录的文件按修改时间排序"
```

`Modelfile/` 里的四份文件分别对应 0.5B / 0.5B-v2 / 1.5B / 1.5B 基座；它们内置了
Marguerite 的 tools 模板（`<tool_call>{"name":…,"arguments":…}</tool_call>`）与
temperature / num_ctx 参数。

> 量化路径的经验结论：**先转 GGUF 再量化**才可靠；直接从 safetensors 目录量化
> （Ollama 0.15.2 实测）会产出崩溃模型。

## 设计取向

- **工具调用正确性优先于闲聊质量**：训练与评测围绕标准 `tool_call` 协议展开，要求模型产出
  可解析的调用而非自然语言描述。评测台还强制 `letter_ok` 校验——未按字母选项训练的模型
  首 token 会返回候选原文，只按字母取概率会导致**静默错误**。
- **低环境需求**：0.5B / 1.5B 在 CPU 上即可交互，不依赖独显。
- **不夸大能力**：评测集覆盖工具选择、参数填充、多步编排与拒答，用同口径对比基座与微调模型；
  未测到的能力不写进说明。

## 许可

基座为 **Qwen2.5** 系列（Apache-2.0）；本仓库的微调产物与配置沿用同一许可。
分发或商用前请自行核对基座模型的原始许可条款。

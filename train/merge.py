"""定向训练 · LoRA 合并与导出为 Ollama 模型。

踩坑记录（已复现的真实缺陷，务必保留修复）：
1. transformers 5.x 会把聊天模板存到独立的 `chat_template.jinja`，而不再写进
   `tokenizer_config.json`。llama.cpp 的转换器只读 `tokenizer_config.json`，
   于是导出的 GGUF 里**没有聊天模板**，Ollama 回退到 `{{ .Prompt }}`——
   模型收到完全未格式化的输入，输出乱码。
   → 修复：保存/合并时把模板写回 `tokenizer_config.json`，并在 Modelfile 中
     显式提供 Go 模板作为双保险。
2. Ollama 0.15.x 不会读取 GGUF 内嵌的 Jinja 模板，只认 Modelfile 的 Go 模板。
   → 修复：导出时始终写入 ChatML Go 模板。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from ..config import ADAPTER_DIR
from ..runtime.llm import OllamaClient

# Qwen/ChatML 系列通用的 Ollama Go 模板（Ollama 不吃 Jinja，必须用 Go 模板）
CHATML_GO_TEMPLATE = (
    "{{- if .Messages -}}\n"
    "{{- range .Messages -}}\n"
    "{{- if eq .Role \"system\" -}}<|im_start|>system\n{{ .Content }}<|im_end|>\n"
    "{{ else if eq .Role \"assistant\" -}}<|im_start|>assistant\n{{ .Content }}<|im_end|>\n"
    "{{ else -}}<|im_start|>user\n{{ .Content }}<|im_end|>\n"
    "{{ end -}}\n"
    "{{- end -}}\n"
    "{{- else -}}\n"
    "{{- if .System -}}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end -}}"
    "<|im_start|>user\n{{ .Prompt }}<|im_end|>\n"
    "{{- end -}}\n"
    "<|im_start|>assistant\n"
)

# 带工具支持的 Qwen2.5 模板（逐字抄自官方 `ollama show --modelfile qwen2.5:0.5b`）。
#
# 为什么必须换掉上面那个裸 ChatML 模板：
#   实测 POST /v1/chat/completions 带 tools 时，Ollama 直接回
#   `registry.ollama.ai/library/omnipilot-1.5b:latest does not support tools`，
#   一个 tool_calls 都不产生 —— 根因就是这个模板没有声明工具支持。
# 官方模板做了三件关键事：
#   1. `.Tools` 非空时把函数签名渲染进 <tools></tools>，并规定模型用
#      <tool_call>{"name":…,"arguments":…}</tool_call> 回复；
#   2. assistant 的 `.ToolCalls` 回渲染成同样的 <tool_call> 块（多轮）；
#   3. 工具结果以 <tool_response> 回灌（Qwen 的多轮工具协议）。
QWEN_TOOLS_GO_TEMPLATE = """{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}"""


def ensure_chat_template(model_dir: str | Path) -> bool:
    """把聊天模板写回 tokenizer_config.json（修复上面的缺陷 1）。

    返回 True 表示确实写入了模板。
    """
    d = Path(model_dir)
    tc = d / "tokenizer_config.json"
    jinja = d / "chat_template.jinja"
    if not tc.exists():
        return False
    cfg = json.loads(tc.read_text(encoding="utf-8"))
    if cfg.get("chat_template"):
        return False
    tpl = None
    if jinja.exists():
        tpl = jinja.read_text(encoding="utf-8")
    else:
        # 退回到基座模型的模板
        try:
            from transformers import AutoTokenizer

            base = cfg.get("_name_or_path") or cfg.get("name_or_path")
            if base:
                # 这里的 base 通常是仓库 id：同样必须解析成本地路径，否则离线必失败。
                tpl = getattr(
                    AutoTokenizer.from_pretrained(resolve_base_model(base)),
                    "chat_template",
                    None,
                )
        except Exception:
            tpl = None
    if not tpl:
        return False
    cfg["chat_template"] = tpl
    tc.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def resolve_base_model(base_model: str | Path) -> str:
    """把 HF 仓库 id 解析成本地缓存快照路径；已是本地目录则原样返回。

    为什么必须这么做（本轮实测根因）：transformers 面对**仓库 id** 时会向 Hub 请求
    `api/models/<id>/tree/main/additional_chat_templates` 来枚举聊天模板。本机到
    huggingface.co 不可达（WinError 10060），于是即便权重与 tokenizer 全在本地缓存里，
    `AutoTokenizer.from_pretrained` 仍会在 5 次重试后抛 ConnectTimeout，整个 deploy 直接失败
    （日志里同时打印着 "Using cached commit hash"，极具迷惑性）。
    解析成本地快照路径后，from_pretrained 全程不再触网。

    离线且未缓存时返回原始 id：联网机器仍可自动下载，行为不变。
    """
    p = Path(base_model)
    if p.exists():
        return str(p)
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(str(base_model), local_files_only=True)
    except Exception:
        return str(base_model)


def merge_lora(base_model: str, adapter_dir: str | Path, out_dir: str | Path) -> dict:
    """把 LoRA 适配器合并进基座，产出可直接转换的完整 HF 模型目录。"""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 先把 id 解析成本地路径：否则离线环境连 tokenizer 都加载不了（见函数注释）。
    base = resolve_base_model(base_model)
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16, trust_remote_code=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))

    ensured = ensure_chat_template(out)
    return {"ok": True, "merged_dir": str(out), "chat_template_restored": ensured}


def find_llama_cpp_convert() -> Path | None:
    """定位 llama.cpp 的 convert_hf_to_gguf.py。"""
    from ..config import WORKSPACE

    cands = [
        WORKSPACE.parent / "llama_cpp" / "src" / "llama.cpp-master" / "convert_hf_to_gguf.py",
        Path(r"D:\llama_cpp\src\llama.cpp-master\convert_hf_to_gguf.py"),
        Path(r"D:\llama.cpp\convert_hf_to_gguf.py"),
        Path(r"C:\llama.cpp\convert_hf_to_gguf.py"),
    ]
    for c in cands:
        if c.exists():
            return c
    which = shutil.which("convert_hf_to_gguf.py")
    return Path(which) if which else None


def build_gguf(merged_dir: str | Path, out_gguf: str | Path, outtype: str = "f16") -> dict:
    """用 llama.cpp 官方转换器把 HF 模型转成 GGUF（会带上聊天模板）。"""
    conv = find_llama_cpp_convert()
    if conv is None:
        return {"ok": False, "message": "未找到 llama.cpp 的 convert_hf_to_gguf.py（需先下载 llama.cpp）"}
    ensure_chat_template(merged_dir)  # 双保险：转换前再确保模板存在
    cmd = [sys.executable, str(conv), str(merged_dir), "--outfile", str(out_gguf), "--outtype", outtype]
    proc = subprocess.run(cmd, cwd=str(conv.parent), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    ok = proc.returncode == 0 and Path(out_gguf).exists()
    return {"ok": ok, "gguf": str(out_gguf) if ok else None,
            "message": (proc.stdout or "")[-300:] if ok else (proc.stderr or "")[-500:]}


def register_ollama(
    name: str,
    gguf_path: str | Path,
    *,
    system: str | None = None,
    temperature: float = 0.2,
    num_ctx: int = 4096,
    tools: bool = True,
    quantize: str | None = "q4_K_M",
) -> dict:
    """把 GGUF 注册为 Ollama 模型，并写入 ChatML Go 模板（修复缺陷 2）。

    tools=True（默认）使用带工具支持的官方 Qwen2.5 模板：不这样做的话，
    Ollama 会对带 `tools` 的请求直接报 "does not support tools"（实测）。

    quantize 默认 q4_K_M —— 这是**低运行环境的硬需求**，不是可选优化：
    1.5B 的 f16 GGUF 实测 3.1GB，在 4GB 卡上 `api/ps` 显示只有 2720MB 进显存、
    其余留在内存，agent 的一次请求直接挂死；量化后 986MB 可 **100% 进显存**。

    为什么是"对 GGUF 量化"而不是"让 Ollama 从 safetensors 目录导入时量化"：
    实测 0.15.2 的后者会产出崩溃的模型（`Assertion failed: found, llama-sampling.cpp:660`，
    连不带模板的 /api/generate 都 500）。同一个模型的 f16、第三方模型、0.5B 均正常 ——
    崩的是那条导入路径。故固定走"先转 GGUF，再 -q 量化"这条成熟路径。
    """
    client = OllamaClient()
    if not client.ping():
        return {"ok": False, "message": "Ollama 未运行，请先启动"}

    mf = Path(gguf_path).parent / f"Modelfile.{name.replace(':', '_')}"
    parts = [f"FROM {Path(gguf_path).resolve()}"]
    if system:
        parts.append(f'SYSTEM """{system}"""')
    tpl = QWEN_TOOLS_GO_TEMPLATE if tools else CHATML_GO_TEMPLATE
    parts.append(f'TEMPLATE """{tpl}"""')
    parts.append(f"PARAMETER temperature {temperature}")
    parts.append(f"PARAMETER num_ctx {num_ctx}")
    mf.write_text("\n".join(parts) + "\n", encoding="utf-8")

    o = str(Path(sys.executable).with_name("Scripts") / "ollama.exe")
    exe = "ollama" if shutil.which("ollama") else o
    subprocess.run([exe, "rm", name], capture_output=True, text=True)
    cmd = [exe, "create", name, "-f", str(mf)]
    if quantize:
        cmd += ["-q", quantize]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    ok = proc.returncode == 0
    return {"ok": ok, "model": name if ok else None, "modelfile": str(mf),
            "quantize": quantize,
            "message": (proc.stdout or "")[-200:] if ok else (proc.stderr or "")[-400:]}

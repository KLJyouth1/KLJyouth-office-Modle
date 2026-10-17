"""Ollama 离线推理运行时。

设计要点（低运行环境）：
- 仅依赖标准库 urllib，不引入 requests 等额外依赖。
- 模型由 Ollama 以 GGUF 量化形式托管，CPU 即可运行；显存不足时 Ollama 自动回退到 CPU。
- 统一封装文本对话、视觉(图像)输入、JSON 结构化输出与流式输出。
"""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

from ..config import OLLAMA_HOST, REQUEST_TIMEOUT


class OllamaError(RuntimeError):
    """Ollama 调用失败（连接不上 / HTTP 错误 / 返回非法）。"""


def encode_image(path: str | Path) -> str:
    """把本地图片编码为 Ollama 需要的 base64 字符串。"""
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


class OllamaClient:
    """极简 Ollama HTTP 客户端。"""

    def __init__(self, host: str = OLLAMA_HOST, timeout: float = REQUEST_TIMEOUT) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------ 底层
    def _request(self, path: str, payload: dict | None = None, *, stream: bool = False):
        url = f"{self.host}{path}"
        if payload is None:
            req = urllib.request.Request(url, method="GET")
        else:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}, method="POST"
            )
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:  # 服务端返回非 2xx
            body = e.read().decode("utf-8", "ignore")
            raise OllamaError(f"{path} HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:  # 连不上服务
            raise OllamaError(f"无法连接 Ollama({self.host})：{e.reason}") from e

        if stream:
            return resp  # 交给调用方逐行读取
        with resp:
            return json.loads(resp.read().decode("utf-8"))

    # ------------------------------------------------------------------ 元信息
    def ping(self) -> bool:
        """服务是否可用。"""
        try:
            self._request("/api/tags")
            return True
        except OllamaError:
            return False

    def list_models(self) -> list[dict]:
        return self._request("/api/tags").get("models", [])

    def has_model(self, name: str) -> bool:
        """是否存在该模型（忽略 :tag 后缀做宽松匹配）。"""
        base = name.split(":")[0]
        for m in self.list_models():
            full = m.get("name", "")
            if full == name or full.split(":")[0] == base:
                return True
        return False

    # ------------------------------------------------------------------ 对话
    def chat_message(
        self,
        model: str,
        messages: list[dict],
        *,
        system: str | None = None,
        images: list[str] | None = None,
        tools: list[dict] | None = None,
        json_mode: bool = False,
        json_schema: dict | None = None,
        options: dict | None = None,
        keep_alive: str = "10m",
        think: bool | None = None,
    ) -> dict:
        """非流式对话，返回**完整的 assistant message**（含 `tool_calls`）。

        tools: OpenAI 风格的函数声明列表；传了才会让 Ollama 走原生工具调用
        （服务端会把 schema 渲染进模板，并把模型的 <tool_call> 解析成结构化
        `tool_calls`）。此前本客户端从不发送 tools，导致 Modelfile 里
        `{{ .Tools }}` 段永远为空、也永远拿不到 tool_calls——这是"模型 ↔ 工作台
        契约错位"的根因之一。
        """
        msgs = self._build_messages(messages, system, images)
        fmt: Any = json_schema if json_schema else ("json" if json_mode else None)
        payload: dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "stream": False,
            "keep_alive": keep_alive,
        }
        if fmt is not None:
            payload["format"] = fmt
        if tools:
            payload["tools"] = tools
        if options:
            payload["options"] = options
        if think is not None:
            payload["think"] = think
        data = self._request("/api/chat", payload)
        return data.get("message") or {}

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        system: str | None = None,
        images: list[str] | None = None,
        json_mode: bool = False,
        json_schema: dict | None = None,
        options: dict | None = None,
        keep_alive: str = "10m",
        think: bool | None = None,
    ) -> str:
        """非流式对话，返回助手文本内容。

        images: base64 字符串列表（仅最后一条 user 消息会附带）。
        json_mode: 要求模型输出严格 JSON；json_schema 进一步约束结构。
        think: 思维链开关。None=用模型默认；False=关闭思考（低配机器上显著提速）。
        """
        msg = self.chat_message(
            model, messages, system=system, images=images, json_mode=json_mode,
            json_schema=json_schema, options=options, keep_alive=keep_alive, think=think,
        )
        return msg.get("content", "") or ""

    def chat_logprobs(
        self,
        model: str,
        messages: list[dict],
        *,
        system: str | None = None,
        top_logprobs: int = 8,
        options: dict | None = None,
        keep_alive: str = "10m",
        think: bool | None = None,
    ) -> dict:
        """只生成 1 个 token 并返回该位置的 top_logprobs（返回**完整响应体**）。

        这是 prefill-only 决策的基础：NeoHorse-Jev 用一次前向计算直接给出候选
        分布；在 Ollama 上等价的做法是——把候选渲染成 A/B/C 选项，只解码 1 个
        token，再读 logprobs 里各字母的 logprob，即为各候选的概率（log 域）。

        返回 dict 含 `logprobs` 字段：
            [{"token": "A", "logprob": -0.69, "top_logprobs": [{"token":"A","logprob":-0.69}, ...]}]
        注意 `logprobs` 挂在响应顶层，不在 `message` 里，因此本方法返回完整响应。
        需要 Ollama >= 0.12；若版本过低会不带该字段，由调用方判定并报错。
        """
        msgs = self._build_messages(messages, system, None)
        payload: dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "stream": False,
            "logprobs": True,
            "top_logprobs": int(top_logprobs),
            "keep_alive": keep_alive,
        }
        if options:
            payload["options"] = options
        if think is not None:
            payload["think"] = think
        return self._request("/api/chat", payload)

    def chat_stream(
        self,
        model: str,
        messages: list[dict],
        *,
        system: str | None = None,
        images: list[str] | None = None,
        options: dict | None = None,
        keep_alive: str = "10m",
        think: bool | None = None,
    ) -> Iterator[str]:
        """流式对话，逐段 yield 文本增量。"""
        msgs = self._build_messages(messages, system, images)
        payload: dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "stream": True,
            "keep_alive": keep_alive,
        }
        if options:
            payload["options"] = options
        if think is not None:
            payload["think"] = think
        resp = self._request("/api/chat", payload, stream=True)
        with resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("done"):
                    break
                chunk = (obj.get("message") or {}).get("content", "")
                if chunk:
                    yield chunk

    # ------------------------------------------------------------------ 视觉
    def vision(
        self,
        model: str,
        prompt: str,
        image_paths: list[str | Path],
        *,
        system: str | None = None,
        json_mode: bool = False,
        options: dict | None = None,
        think: bool | None = None,
    ) -> str:
        """对一张或多张图片提问（图像理解/界面理解）。"""
        images = [encode_image(p) for p in image_paths]
        return self.chat(
            model,
            [{"role": "user", "content": prompt}],
            system=system,
            images=images,
            json_mode=json_mode,
            options=options,
            think=think,
        )

    # ------------------------------------------------------------------ 流式生成
    def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        options: dict | None = None,
        stream: bool = False,
        keep_alive: str = "10m",
        think: bool | None = None,
    ):
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "keep_alive": keep_alive,
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options
        if think is not None:
            payload["think"] = think
        if not stream:
            return self._request("/api/generate", payload).get("response", "")
        resp = self._request("/api/generate", payload, stream=True)

        def _iter() -> Iterator[str]:
            with resp:
                for raw in resp:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("done"):
                        break
                    if obj.get("response"):
                        yield obj["response"]

        return _iter()

    # ------------------------------------------------------------------ 模型管理
    def pull(self, name: str, on_progress=None) -> None:
        """拉取模型（首次使用需联网一次；之后完全离线）。"""
        resp = self._request("/api/pull", {"name": name, "stream": True}, stream=True)
        with resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if on_progress:
                    on_progress(obj)

    def create_from_adapter(
        self,
        new_name: str,
        base_model: str,
        adapter_path: str | Path,
        *,
        system: str | None = None,
    ) -> None:
        """把一个 LoRA 适配器(GGUF)挂到基础模型上，生成新的可调用模型。

        这是"定向训练"闭环的最后一环：训练出的专属适配器即插即用，
        无需改任何推理代码，直接作为 OLLAMA 模型名调用。
        """
        payload: dict[str, Any] = {
            "model": new_name,
            "from": base_model,
            "adapters": {Path(adapter_path).stem: str(adapter_path)},
        }
        if system:
            payload["system"] = system
        self._request("/api/create", payload)

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _build_messages(
        messages: list[dict], system: str | None, images: list[str] | None
    ) -> list[dict]:
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)
        if images:
            # 把图像挂到最后一条 user 消息上（Ollama 约定）
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i].get("role") == "user":
                    msgs[i] = {**msgs[i], "images": images}
                    break
        return msgs

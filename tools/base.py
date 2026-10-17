"""工具层基类与注册表。

统一约定：每个工具都有 名称/描述/参数schema/所属领域，模型据此选择工具并填参。
所有工具调用都返回 ToolResult，异常被捕获为失败结果，绝不把异常抛进决策循环。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolResult:
    ok: bool
    output: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.output or ("OK" if self.ok else "FAILED")

    def to_dict(self) -> dict:
        return {"ok": self.ok, "output": self.output, "data": self.data}


class Tool(ABC):
    name: str = ""
    description: str = ""
    domain: str = "system"
    parameters: dict[str, Any] = {}

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        ...

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


class FuncTool(Tool):
    """把一个普通函数包装成 Tool。"""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., ToolResult],
        parameters: dict[str, Any] | None = None,
        domain: str = "system",
    ) -> None:
        self.name = name
        self.description = description
        self._func = func
        self.parameters = parameters or {}
        self.domain = domain

    def run(self, **kwargs) -> ToolResult:
        try:
            return self._func(**kwargs)
        except Exception as e:  # 工具失败不应中断整个循环
            return ToolResult(False, f"{type(e).__name__}: {e}")


class ToolRegistry:
    """工具注册表：集中管理、按领域筛选、生成给模型看的清单。"""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def extend(self, tools: list[Tool]) -> None:
        for t in tools:
            self.add(t)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def by_domain(self, domains: tuple[str, ...] | None) -> list[Tool]:
        if not domains:
            return self.all()
        return [t for t in self._tools.values() if t.domain in domains]

    def describe(self, domains: tuple[str, ...] | None = None, names: list[str] | None = None) -> str:
        """生成紧凑工具清单文本，供提示词使用。"""
        tools = self.by_domain(domains)
        if names:
            tools = [t for t in tools if t.name in names]
        lines = []
        for t in tools:
            props = list((t.parameters.get("properties") or {}).keys())
            req = t.parameters.get("required") or []
            arg_str = ", ".join(f"{p}*" if p in req else p for p in props)
            lines.append(f"- {t.name}({arg_str}): {t.description}")
        return "\n".join(lines)

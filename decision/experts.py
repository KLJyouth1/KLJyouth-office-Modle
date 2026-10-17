"""LoRA 专家路由：稀疏激活的领域专家（"类脑稀疏"在本机的可行实现）。

为什么是这个形态（而不是同时加载多份权重）：
4GB 显存不可能同时驻留多个专家。稀疏激活的正确做法不是"把 N 个专家一起塞进显存"，
而是**每步只激活一个**：先用门控选出专家，再只把该专家挂到基座上推理。
专家各自注册为独立 Ollama 模型，Ollama 按 keep_alive 自动卸载 →
同一时刻只有一个专家进显存，这正是稀疏激活。

门控为什么复用 Jev 的 Choice 接口：门控的输入是"状态"，输出是"选哪个专家"，
与 Choice 完全同构。复用它可以白拿到概率分布、letter_ok 可信度与回退策略，
不必另造一套易漂移的判定逻辑。

为什么必须保留 `default` 回退：门控可能不可信（思考型模型首 token 不是字母、
或置信度过低）。此时宁可用默认专家，也不要把请求路由到一个"其实很犹豫"的专家上。
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import DECISION_MODEL
from ..runtime.llm import OllamaClient
from .engine import DecisionEngine

# 门控置信度下限：低于它就走默认专家（宁可保守，也不误路由）。
GATE_MIN_CONF = 0.35


@dataclass(frozen=True)
class Expert:
    """一个专家 = 基座 + 某领域 LoRA，注册为独立 Ollama 模型。"""

    name: str            # 专家标识，也是门控候选的 key
    model: str           # Ollama 模型名（基座已挂好该领域适配器）
    desc: str = ""       # 门控看到的候选说明
    domain: str = ""     # 归属领域（对应 omnipilot 六大域，便于观测/统计）


class ExpertRegistry:
    """专家注册表。`default` 是门控不可信时的兜底专家。"""

    def __init__(self, experts: list[Expert] | None = None, default: str | None = None) -> None:
        self._by_name: dict[str, Expert] = {}
        for e in experts or []:
            self.add(e)
        self.default = default or (next(iter(self._by_name)) if self._by_name else None)

    def add(self, expert: Expert) -> None:
        self._by_name[expert.name] = expert

    def get(self, name: str | None) -> Expert | None:
        return self._by_name.get(name) if name else None

    def names(self) -> list[str]:
        return list(self._by_name)

    def all(self) -> list[Expert]:
        return list(self._by_name.values())

    def prune_to_available(self, client: OllamaClient) -> list[str]:
        """剔除 Ollama 里不存在的专家模型，返回被剔除的名字。

        不做这一步的后果很具体：门控可能选中一个尚未导出的专家模型，
        然后整条请求在推理时报 "model not found" —— 比直接回退危险。
        """
        missing = []
        for name, e in list(self._by_name.items()):
            try:
                if not client.has_model(e.model):
                    missing.append(name)
                    del self._by_name[name]
            except Exception:
                missing.append(name)
                del self._by_name[name]
        if self.default not in self._by_name:
            self.default = next(iter(self._by_name), None)
        return missing


class ExpertRouter:
    """门控（Choice）→ 选中专家 → 由该专家作答。"""

    def __init__(self, registry: ExpertRegistry, *,
                 gate: DecisionEngine | None = None,
                 gate_model: str = DECISION_MODEL,
                 client: OllamaClient | None = None,
                 min_conf: float = GATE_MIN_CONF) -> None:
        self.registry = registry
        self.client = client or OllamaClient()
        self.gate = gate or DecisionEngine(client=self.client, model=gate_model)
        self.min_conf = min_conf

    # ------------------------------------------------------------------ 门控
    def route(self, state: str) -> tuple[Expert | None, dict]:
        """选专家。返回 (专家, 诊断)；实在没有可用专家时返回 (None, 诊断)。"""
        names = self.registry.names()
        diag: dict = {"candidates": names, "gate_model": getattr(self.gate, "model", "?")}
        if not names:
            return None, {**diag, "reason": "注册表为空", "reliable": False}

        if len(names) == 1:
            diag.update({"picked": names[0], "reliable": True, "reason": "唯一专家，无需门控"})
            return self.registry.get(names[0]), diag

        criteria = {e.name: e.desc for e in self.registry.all()}
        try:
            sc = self.gate.score_candidates(state, "这条请求应该交给哪个领域专家处理？",
                                            names, criteria)
        except Exception as ex:  # 门控失败 → 默认专家
            diag.update({"picked": self.registry.default, "reliable": False,
                         "reason": f"门控不可用：{type(ex).__name__}: {ex}"})
            return self.registry.get(self.registry.default), diag

        best = names[sc.best_index()]
        reliable = sc.letter_ok and sc.probs[sc.best_index()] >= self.min_conf
        if not reliable:
            diag.update({"picked": self.registry.default, "reliable": False,
                         "gate_probs": dict(zip(names, sc.probs)),
                         "reason": "门控不可信（未按字母作答或置信度不足），回退默认专家"})
            return self.registry.get(self.registry.default), diag

        diag.update({"picked": best, "reliable": True,
                     "gate_probs": dict(zip(names, sc.probs)),
                     "gate_conf": round(sc.probs[sc.best_index()], 4)})
        return self.registry.get(best), diag

    # ------------------------------------------------------------------ 作答
    def decide(self, state: str, questions: dict) -> dict:
        """路由到专家后，由该专家按 Jev 契约作答（响应附 chosen_expert 诊断）。"""
        expert, diag = self.route(state)
        if expert is None:
            return {"answers": {}, "diagnostics": {"router": diag},
                    "error": "没有可用专家（注册表为空或全部模型缺失）"}
        eng = DecisionEngine(client=self.client, model=expert.model)
        out = eng.decide(state, questions)
        out.setdefault("diagnostics", {})["router"] = diag
        out["expert"] = expert.name
        return out

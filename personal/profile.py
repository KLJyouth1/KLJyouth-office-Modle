"""Personal Agent 层：让智能体主动记住并适配"你"。

设计依据（均为已发表/可核验工作）：
- PersonaVLM（CVPR2026 方向）：长期个性化多模态智能体，采用**多库式主动记忆管理**与
  **动量式人格演化**，使助手随用户偏好演化而适配。
- EPIC（UNIST 2026.10）："只记住重要的"——通过记忆选择把长期记忆体量大幅压缩，
  同时把**偏好遵循准确率提升 18.79 个百分点**。结论：记忆的价值在"选择"，不在"堆量"。
- ScrapMem（arXiv 2605.03804）：端侧个性化记忆通过"遗忘"控制成本。

实现取舍（低资源优先）：
- 四类长期记忆库：偏好 preference / 事实 fact / 技能 skill（+ 情节 episode 复用现有 Memory）。
- **动量更新**：新证据以动量方式缓慢改写权重，避免被单次异常带偏。
- **主动遗忘**：低权重、长期未命中的条目自动剪枝，控制体积。
- 全部本地 JSON 落盘，零外部依赖。
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import DATA_DIR

BANKS = ("preference", "fact", "skill")

# 失败噪声标记：画像里**绝不能**承载这些字符串。
# 为什么必须有这道过滤：`summary()` 会把 fact 库原样注入"计划"字段。历史失败记录一旦
# 出现在那里，小模型会把它读成"我当前正在失败"，从而放弃正常执行路径（实测已复现：
# 模型不调 file_write，转而去"打开记事本"）。而且画像**是落盘的**——旧版本已经写进去的
# 毒数据会持续注入，所以只在写入端拦截不够，呈现端也必须过滤。
_FAILURE_NOISE = ("失败", "未完成", "达到最大步数", "error", "exit=", "exception",
                  "timeout", "com_error", "traceback")


def _is_failure_noise(value: str) -> bool:
    v = (value or "").lower()
    return any(m in v for m in _FAILURE_NOISE)


DEFAULT_PATH = DATA_DIR / "personal_profile.json"

MOMENTUM = 0.30          # 动量系数：越小越"稳"，越不容易被单次证据带偏
INIT_DAMP = 0.50         # 新条目初始阻尼：单次证据只给一半权重，反复出现才推高
FORGET_THRESHOLD = 0.18  # 低于此权重且长期未命中 → 剪枝
FORGET_MAX_AGE = 60 * 60 * 24 * 30  # 30 天未命中视为陈旧


@dataclass
class Entry:
    key: str
    value: str
    weight: float = 0.5
    hits: int = 0
    updated: float = field(default_factory=time.time)

    @property
    def stale(self) -> bool:
        return (time.time() - self.updated) > FORGET_MAX_AGE


@dataclass
class PersonalProfile:
    """个人画像与长期记忆（三类库）。"""

    path: Path = DEFAULT_PATH
    banks: dict[str, dict[str, Entry]] = field(default_factory=dict)

    # ------------------------------------------------------------------ 持久化
    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> "PersonalProfile":
        p = Path(path)
        prof = cls(path=p, banks={b: {} for b in BANKS})
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                for bank, items in (raw.get("banks") or {}).items():
                    prof.banks.setdefault(bank, {})
                    for k, v in items.items():
                        prof.banks[bank][k] = Entry(**v)
            except (json.JSONDecodeError, TypeError):
                pass
        return prof

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "banks": {
            b: {k: asdict(e) for k, e in items.items()} for b, items in self.banks.items()
        }}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ 写入
    def observe(self, bank: str, key: str, value: str, *, signal: float = 1.0) -> Entry:
        """写入一条长期记忆，并用**动量**更新其权重。

        signal ∈ [0,1]：本次证据的强度。权重按
            w ← (1-m)*w + m*signal
        演化。这样单次强证据不会立刻把偏好刷成极端值，
        反复出现的证据才会把权重推高——这正是"人格演化"该有的样子。
        """
        if bank not in BANKS:
            raise ValueError(f"未知记忆库: {bank}（可选 {BANKS}）")
        bank_items = self.banks.setdefault(bank, {})
        entry = bank_items.get(key)
        if entry is None:
            # 新条目也要走阻尼：单次证据不足以把它推成"强偏好"，
            # 避免一次偶发输入就覆盖长期人格（实测发现的自身缺陷）。
            entry = Entry(key=key, value=value, weight=max(0.05, min(1.0, signal * INIT_DAMP)))
        else:
            entry.value = value  # 取最新表述
            entry.weight = (1 - MOMENTUM) * entry.weight + MOMENTUM * max(0.0, min(1.0, signal))
            entry.updated = time.time()
        bank_items[key] = entry
        return entry

    # ------------------------------------------------------------------ 读取
    def top(self, bank: str, k: int = 5) -> list[Entry]:
        items = list(self.banks.get(bank, {}).values())
        items.sort(key=lambda e: (e.weight, e.hits), reverse=True)
        for e in items[:k]:
            e.hits += 1
            e.updated = time.time()
        return items[:k]

    def recall(self, query: str, k: int = 4) -> list[Entry]:
        """按关键词命中 + 权重打分，跨库召回最相关的长期记忆。"""
        keys = [w for w in query.replace("，", " ").replace(",", " ").split() if len(w) >= 2]
        scored: list[tuple[float, Entry]] = []
        for bank, items in self.banks.items():
            for e in items.values():
                hit = sum(1 for w in keys if w in e.key or w in e.value)
                if hit:
                    scored.append((hit * e.weight, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = [e for _, e in scored[:k]]
        for e in out:
            e.hits += 1
            e.updated = time.time()
        return out

    # ------------------------------------------------------------------ 主动遗忘
    def forget(self, *, threshold: float = FORGET_THRESHOLD) -> list[str]:
        """剪枝低价值记忆（借鉴 EPIC："只记住重要的"）。返回被删条目。"""
        removed: list[str] = []
        for bank, items in self.banks.items():
            for k in list(items.keys()):
                e = items[k]
                if e.weight < threshold and e.stale:
                    items.pop(k)
                    removed.append(f"{bank}:{k}")
        return removed

    # ------------------------------------------------------------------ 学习
    def learn_from_episode(self, task: str, result: str, ok: bool) -> None:
        """从一次任务结果中提炼长期记忆。

        只从**成功**的轮次学技能；失败轮次什么都不记。
        实测缺陷：旧实现把失败原文（`失败：达到最大步数(12)仍未完成`）当"事实"存进
        `fact` 库，`summary()` 下一轮又把它当"已知事实"注入计划字段 —— 模型据此判断
        "我正在失败"，规划直接跑偏。失败不是事实、是噪声：记不住失败不影响能力，
        记错了会持续自我投毒（而且落盘后越滚越脏）。
        """
        if not ok:
            return
        self.observe("skill", task[:40], f"成功：{result[:60]}", signal=0.85)

    # ------------------------------------------------------------------ 呈现
    def summary(self, k_pref: int = 4, k_fact: int = 3, k_skill: int = 3) -> str:
        """生成给模型看的"画像"文本（注入计划字段，不污染观察字段）。"""
        parts: list[str] = []
        prefs = self.top("preference", k_pref)
        if prefs:
            parts.append("用户偏好：" + "；".join(f"{e.value}(w={e.weight:.2f})" for e in prefs))
        facts = [e for e in self.top("fact", k_fact + 4)
                 if not _is_failure_noise(e.value)][:k_fact]
        if facts:
            parts.append("已知事实：" + "；".join(e.value[:40] for e in facts))
        skills = self.top("skill", k_skill)
        if skills:
            parts.append("擅长/常用：" + "；".join(e.key[:24] for e in skills))
        return "\n".join(parts)

    def stats(self) -> dict:
        return {
            "banks": {b: len(items) for b, items in self.banks.items()},
            "total": sum(len(i) for i in self.banks.values()),
        }


def infer_preferences_from_text(text: str) -> list[tuple[str, str, float]]:
    """从自然语言里抽取偏好信号的轻量规则（零算力）。

    真正的个性化应来自真实交互统计；这里提供一个可用的冷启动规则，
    把用户显式表达的"我喜欢/我要/以后都…"转成偏好条目。
    """
    rules = (
        ("喜欢", "偏好", 0.8), ("我要", "偏好", 0.7), ("以后都", "偏好", 0.9),
        ("不要", "禁忌", 0.9), ("别", "禁忌", 0.7), ("必须", "约束", 0.8),
        ("默认", "默认设置", 0.7), ("总是", "习惯", 0.7),
    )
    out: list[tuple[str, str, float]] = []
    for marker, kind, w in rules:
        if marker in text:
            idx = text.find(marker)
            tail = text[idx + len(marker):].strip(" ：:，,。.）)")[:30]
            if tail:
                out.append((f"{kind}:{tail}", text[idx: idx + 40], w))
    return out

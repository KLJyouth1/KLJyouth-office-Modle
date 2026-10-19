"""从评测报告生成能力图（可复现，无网络、无模型依赖）。

用法：
    python scripts/make_capability_charts.py

输入（仓库内，路径以本脚本位置为基准，不依赖运行时工作目录）：
    docs/eval/omnipilot17-eval-report.json   —— 由 eval/decision_eval.py 产出的同口径报告
    data/datasets/decision_eval/*.jsonl      —— 评测集原文（用于推算每任务的"随机基线"）

输出：
    docs/charts/capability_<model>.png       每个模型一张：逐任务准确率 + 随机基线标记
    docs/charts/capability_compare.png       跨模型分组对比（含宏观平均）
    docs/charts/routing_radar.png            六域路由 recall 雷达（真正的"能力图"）
    docs/charts/tool_choice_by_domain.png    域内工具选择的逐域准确率

**诚实性约束（本脚本的核心设计原则）**：
1. 每个任务都画**随机基线**（1/候选数）。低于随机的任务用红色标出，并写入脚注——
   因为"低于随机"意味着模型不是没学会，而是被教反了，这比"没学会"更值得暴露。
2. 不隐藏多数类基线：accuracy 会被"恒答多数类"抬高，故报告同时给出宏平均与均衡宏平均。
3. 有多数类/随机基线都无法解释的退化任务时，图上显式标注。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "docs" / "eval" / "omnipilot17-eval-report.json"
EVAL_DIR = REPO / "data" / "datasets" / "decision_eval"
OUT_DIR = REPO / "docs" / "charts"

# 任务名 → 图中显示名（中文，便于阅读；键名仍是报告里的原键）
TASK_LABEL = {
    "routing": "六域路由",
    "directed_routing": "定向六域路由",
    "tool_choice": "工具选择",
    "directed_tool_choice": "定向工具选择",
    "intent_massive": "意图(MASSIVE)",
    "intent_clinc": "意图(CLINC)",
    "emotion6": "情绪(6类)",
    "emotion28": "情绪(28类)",
    "fact_vitaminc": "事实核验",
    "fact_evidence_noul": "证据是否足够",
    "nli_mnli": "蕴涵关系",
    "nli_noul": "是否蕴含",
    "nli_score": "支持程度",
    "abstain": "无据拒答",
    "grounded": "有据作答",
}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _chance_by_task() -> dict[str, float]:
    """每任务的随机基线 = 1/候选数。按**记录加权平均**（同任务内候选数可能随样本变化）。"""
    out: dict[str, float] = {}
    for f in sorted(EVAL_DIR.glob("*.jsonl")):
        tot = 0.0
        n = 0
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for q in rec["questions"].values():
                if q["type"] == "choice":
                    k = len(q.get("criteria") or {})
                elif q["type"] == "score":
                    k = len(q.get("criteria") or [])
                else:
                    k = 2          # noul: false/true
                tot += 1.0 / max(1, k)
                n += 1
        if n:
            out[f.stem] = tot / n
    return out


def _domain_map() -> dict[str, str]:
    """工具名 → 域。取自定向工具选择评测集里的 meta.domain（造数据时就写进去了）。"""
    p = EVAL_DIR / "directed_tool_choice.jsonl"
    m: dict[str, str] = {}
    if not p.exists():
        return m
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        dom = (rec.get("meta") or {}).get("domain")
        gold = next(iter(rec["gold"].values()))["value"]
        if dom:
            m[gold] = dom
    return m


def _setup_fonts(plt) -> None:
    """中文字体：Windows 上用雅黑/黑体；找不到就退化为默认（图会显示方块，但仍可出图）。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def chart_per_model(plt, models: list[dict], chance: dict[str, float]) -> None:
    for m in models:
        per = m["per_task"]
        tasks = sorted(per, key=lambda t: per[t]["accuracy"])
        y = range(len(tasks))
        acc = [per[t]["accuracy"] for t in tasks]
        ch = [chance.get(t, 0.0) for t in tasks]
        colors = ["#d64545" if per[t]["accuracy"] < ch[i] - 0.02 else "#3b7dd8"
                  for i, t in enumerate(tasks)]

        fig, ax = plt.subplots(figsize=(9, 0.42 * len(tasks) + 1.6))
        ax.barh(list(y), acc, color=colors, height=0.62)
        for i, c in enumerate(ch):
            ax.plot([c], [i], marker="|", color="black", markersize=14, markeredgewidth=2)
        ax.set_yticks(list(y))
        ax.set_yticklabels([TASK_LABEL.get(t, t) for t in tasks])
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("准确率（竖线＝该任务随机基线；红色＝低于随机）")
        ax.set_title(
            f"{m['model']} 能力图\n"
            f"宏平均 {m['macro_accuracy']:.3f} ｜ 均衡宏平均 {m['macro_balanced']:.3f} "
            f"｜ 字母合规率 {m['letter_ok_rate']:.3f}"
        )
        for i, t in enumerate(tasks):
            ax.text(min(acc[i] + 0.012, 0.97), i, f"{acc[i]:.3f}", va="center", fontsize=8)
        worse = [TASK_LABEL.get(t, t) for t in tasks if per[t]["accuracy"] < chance.get(t, 0) - 0.02]
        if worse:
            fig.text(0.01, 0.01, "低于随机的任务（说明被教反，而非没学会）：" + "、".join(worse),
                     fontsize=8, color="#d64545")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(OUT_DIR / f"capability_{m['model']}.png", dpi=160)
        plt.close(fig)


def chart_compare(plt, models: list[dict]) -> None:
    tasks = sorted(models[0]["per_task"])
    names = [m["model"] for m in models]
    x = range(len(tasks))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(13, 5.4))
    palette = ["#3b7dd8", "#e08b2a", "#4c9f70", "#9b6bd6", "#c0504d"]
    for k, m in enumerate(models):
        vals = [m["per_task"][t]["accuracy"] for t in tasks]
        ax.bar([i + k * w for i in x], vals, width=w, label=m["model"],
               color=palette[k % len(palette)])
    ax.set_xticks([i + 0.4 - w / 2 for i in x])
    ax.set_xticklabels([TASK_LABEL.get(t, t) for t in tasks], rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("准确率")
    ax.set_title("跨模型同口径对比（15 个任务；同一批评测集、同一提示渲染）")
    ax.legend(ncols=2)
    ax.axhline(1 / 6, color="gray", ls=":", lw=1)
    ax.text(0.1, 1 / 6 + 0.012, "1/6（六选一的随机水平，供参照）", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "capability_compare.png", dpi=160)
    plt.close(fig)


def chart_radar(plt, models: list[dict]) -> None:
    """六域路由 recall 雷达：用报告里 class_stats 的逐类召回，比单一 accuracy 信息量大。"""
    import math

    doms = ["browser", "office", "code", "media", "blender", "system"]
    dom_cn = {"browser": "浏览器", "office": "办公", "code": "代码",
              "media": "媒体", "blender": "3D建模", "system": "系统"}
    angles = [2 * math.pi * i / len(doms) for i in range(len(doms))]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7.2, 6.4), subplot_kw=dict(polar=True))
    palette = ["#3b7dd8", "#e08b2a", "#4c9f70"]
    have = False
    for k, m in enumerate(models):
        cs = m["per_task"].get("directed_routing", {}).get("class_stats", {})
        vals = []
        for d in doms:
            c, t = cs.get(f"choice:{d}", [0, 0])
            vals.append((c / t) if t else 0.0)
        if not any(vals):
            continue
        have = True
        v = vals + vals[:1]
        ax.plot(angles, v, color=palette[k % 3], label=m["model"], lw=2)
        ax.fill(angles, v, color=palette[k % 3], alpha=0.12)
    ax.plot(angles, [1 / 6] * len(angles), color="gray", ls=":", lw=1.6, label="随机 1/6")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([dom_cn[d] for d in doms])
    ax.set_ylim(0, 1.0)
    ax.set_title("六域路由能力雷达（定向评测集 107 条，逐域召回）\n"
                 "radar 越小说明该域越弱；贴近虚线即等于瞎猜", pad=18)
    if have:
        ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.12))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "routing_radar.png", dpi=160)
    plt.close(fig)


def chart_tool_by_domain(plt, models: list[dict], dom_map: dict[str, str]) -> None:
    doms = ["browser", "system", "office", "media", "code", "blender"]
    dom_cn = {"browser": "浏览器", "office": "办公", "code": "代码",
              "media": "媒体", "blender": "3D建模", "system": "系统"}
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = range(len(doms))
    w = 0.8 / max(1, len(models))
    palette = ["#3b7dd8", "#e08b2a", "#4c9f70"]
    for k, m in enumerate(models):
        cs = m["per_task"].get("directed_tool_choice", {}).get("class_stats", {})
        agg = {d: [0, 0] for d in doms}
        for label, (ok, tot) in cs.items():
            tool = label.split("choice:", 1)[-1]
            d = dom_map.get(tool)
            if d in agg:
                agg[d][0] += ok
                agg[d][1] += tot
        vals = [(agg[d][0] / agg[d][1]) if agg[d][1] else 0.0 for d in doms]
        ax.bar([i + k * w for i in x], vals, width=w, label=m["model"],
               color=palette[k % len(palette)])
        for i, d in enumerate(doms):
            if agg[d][1]:
                ax.text(i + k * w, vals[i] + 0.012, f"{vals[i]:.2f}\nn={agg[d][1]}",
                        ha="center", fontsize=7)
    ax.set_xticks([i + 0.4 - w / 2 for i in x])
    ax.set_xticklabels([f"{dom_cn[d]}\n({d})" for d in doms])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("准确率")
    ax.set_title("域内工具选择：逐域准确率（定向评测集 93 条；n 为该域样本数）\n"
                 "注意 blender/code 的 n 很小，结论仅供方向参考")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tool_choice_by_domain.png", dpi=160)
    plt.close(fig)


def main() -> int:
    if not REPORT.exists():
        print(f"[FAIL] 找不到评测报告：{REPORT}", file=sys.stderr)
        return 2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _setup_fonts(plt)

    data = _load_json(REPORT)
    models = [r for r in data.get("results", []) if "per_task" in r]
    if not models:
        print("[FAIL] 报告里没有 per_task 结果", file=sys.stderr)
        return 2

    chance = _chance_by_task()
    dom_map = _domain_map()
    print(f"[info] 评测集任务数={len(chance)} 工具→域映射={len(dom_map)} 模型数={len(models)}")

    chart_per_model(plt, models, chance)
    chart_compare(plt, models)
    chart_radar(plt, models[:3])
    chart_tool_by_domain(plt, models[:3], dom_map)

    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"[ok] {f.relative_to(REPO)}  {f.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

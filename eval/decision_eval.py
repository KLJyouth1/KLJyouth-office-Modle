"""决策评测台：以**同一口径**给不同后端打分（自训模型 / 基座 / 关键词基线）。

与 train/decision_data.py 严格配套：评测集来自那边冻结下来的
`data/datasets/decision_eval/<task>.jsonl`，训练集与它同源不重叠（已去污染）。

指标：
- accuracy：决策正确率（noul 以 0.5 阈值为准；score 默认要求等级完全一致）。
- letter_ok：模型是否真的按"字母选项"作答（格式有效性）。这一项是我们能
  从 logprobs 稳定取到候选概率的前提；它低，说明概率不可信、上层应回退。
- mean_conf：被选中候选的平均概率（置信度，用于观察"自信地错"还是"犹豫地错"）。
- abstain_ok：仅 abstain 任务有意义，即"无依据时是否如实判定为无依据"。

用法：
    python -m omnipilot.eval.decision_eval --models qwen2.5-1.5b,omnipilot-1.5b-v2 --limit 200
    python -m omnipilot.eval.decision_eval --keyword-only      # 只看路由的关键词基线
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config import DATASET_DIR
from ..decision import schema
from ..decision.engine import DecisionEngine

EVAL_DIR = DATASET_DIR / "decision_eval"


@dataclass
class TaskResult:
    task: str
    n: int = 0
    correct: int = 0
    letter_ok: int = 0
    conf_sum: float = 0.0
    errors: list[dict] = field(default_factory=list)
    # 每类 [答对数, 总数]；均衡准确率的分子分母。
    class_stats: dict[str, list[int]] = field(default_factory=dict)
    # 预测标签分布（用于暴露"恒答同一类"这种退化行为）。
    pred_stats: dict[str, int] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return round(self.correct / self.n, 4) if self.n else 0.0

    @property
    def balanced_accuracy(self) -> float:
        """各类召回率的等权平均（macro recall）。

        为什么必须看它、而不能只看 accuracy：
        实测发现 nli_noul 评测集约 76% 为 true，于是"恒答 true"就能拿 0.76 的
        accuracy —— 一个完全没有推理能力的退化策略反而排名第一。只看 accuracy
        会把这类退化当作优秀，并诱导我们用"翻转偏置"去刷分，而不是真的提升能力。
        均衡准确率对每类一视同仁，恒答同一类只会有 1/类别数 的分数。
        """
        if not self.class_stats:
            return 0.0
        recalls = [c / t for c, t in self.class_stats.values() if t > 0]
        return round(sum(recalls) / len(recalls), 4) if recalls else 0.0

    @property
    def degenerate(self) -> bool:
        """是否退化为"几乎恒定作答"（单一预测标签占比 ≥95%）。"""
        if not self.pred_stats or not self.n:
            return False
        return max(self.pred_stats.values()) / self.n >= 0.95

    @property
    def majority_baseline(self) -> float:
        """多数类基线准确率。凡 accuracy 不显著高于它，就说明模型没有真本事。"""
        if not self.class_stats:
            return 0.0
        return round(max(t for _, t in self.class_stats.values()) / self.n, 4) if self.n else 0.0

    @property
    def letter_ok_rate(self) -> float:
        return round(self.letter_ok / self.n, 4) if self.n else 0.0

    @property
    def mean_conf(self) -> float:
        return round(self.conf_sum / self.n, 4) if self.n else 0.0

    def as_dict(self) -> dict:
        return {"task": self.task, "n": self.n, "accuracy": self.accuracy,
                "balanced_accuracy": self.balanced_accuracy,
                "majority_baseline": self.majority_baseline,
                "degenerate": self.degenerate,
                "letter_ok_rate": self.letter_ok_rate, "mean_conf": self.mean_conf,
                "class_stats": self.class_stats, "pred_stats": self.pred_stats}


@dataclass
class ModelResult:
    model: str
    per_task: dict[str, TaskResult] = field(default_factory=dict)

    @property
    def macro_accuracy(self) -> float:
        """各任务 accuracy 的等权平均（与官方 AVG 的算法口径一致）。

        注意：该指标会奖励"恒答多数类"的退化策略，必须与 `macro_balanced` 一起看。
        """
        if not self.per_task:
            return 0.0
        return round(sum(t.accuracy for t in self.per_task.values()) / len(self.per_task), 4)

    @property
    def macro_balanced(self) -> float:
        """各任务**均衡准确率**的等权平均。这是判断真实能力的指标。"""
        if not self.per_task:
            return 0.0
        return round(sum(t.balanced_accuracy for t in self.per_task.values()) / len(self.per_task), 4)

    @property
    def degenerate_tasks(self) -> list[str]:
        """退化为恒定作答的任务名（这些任务的 accuracy 不可信）。"""
        return sorted(t.task for t in self.per_task.values() if t.degenerate)

    @property
    def letter_ok_rate(self) -> float:
        total = sum(t.n for t in self.per_task.values())
        ok = sum(t.letter_ok for t in self.per_task.values())
        return round(ok / total, 4) if total else 0.0

    def as_dict(self) -> dict:
        return {"model": self.model, "macro_accuracy": self.macro_accuracy,
                "macro_balanced": self.macro_balanced,
                "degenerate_tasks": self.degenerate_tasks,
                "letter_ok_rate": self.letter_ok_rate,
                "per_task": {k: v.as_dict() for k, v in self.per_task.items()}}


def load_eval(eval_dir: Path | str = EVAL_DIR, tasks: list[str] | None = None) -> dict[str, list[dict]]:
    d = Path(eval_dir)
    out: dict[str, list[dict]] = {}
    for f in sorted(d.glob("*.jsonl")):
        task = f.stem
        if tasks and task not in tasks:
            continue
        rows = []
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        if rows:
            out[task] = rows
    return out


def _answer_confidence(answer: dict) -> float:
    """被选中项的置信度：choice=选中项概率；noul=判定侧的 P；score=离散度低者更自信。"""
    if not isinstance(answer, dict):
        return 0.0
    t = answer.get("type")
    try:
        if t == schema.CHOICE:
            return float((answer.get("probabilities") or {}).get(answer.get("choice"), 0.0))
        if t == schema.NOUL:
            p = float(answer.get("noul", 0.0))
            return p if p >= 0.5 else 1.0 - p
        if t == schema.SCORE:
            probs = [float(v) for v in (answer.get("probabilities") or {}).values()]
            return max(probs) if probs else 0.0
    except (TypeError, ValueError):
        return 0.0
    return 0.0


def _predicted_label(gold_type: str, answer: dict) -> str:
    """把答案归一成一个可比较的标签：用于统计预测分布、算均衡准确率、识别退化作答。

    与 `schema.answer_is_correct` 的判定口径**必须一致**（同样是 score 取 argmax），
    否则会出现"判正确但预测标签对不上"的自相矛盾统计。
    """
    if not isinstance(answer, dict):
        return "(invalid)"
    try:
        if gold_type == schema.CHOICE:
            return str(answer.get("choice"))
        if gold_type == schema.NOUL:
            return "true" if float(answer.get("noul", 0.0)) >= 0.5 else "false"
        if gold_type == schema.SCORE:
            probs = answer.get("probabilities")
            if isinstance(probs, dict) and probs:
                return str(max(range(len(probs)), key=lambda i: float(probs[str(i)])))
            return str(int(round(float(answer.get("score", -99.0)))))
    except (TypeError, ValueError, KeyError):
        return "(invalid)"
    return "(invalid)"


def evaluate_model(engine: DecisionEngine, eval_dir: Path | str = EVAL_DIR,
                   tasks: list[str] | None = None, limit: int | None = None,
                   *, score_tol: int = 0, verbose: bool = True,
                   collect_errors: int = 20) -> ModelResult:
    """用 DecisionEngine 给一个后端（Ollama 模型）打分。"""
    res = ModelResult(model=engine.model)
    for task, rows in load_eval(eval_dir, tasks).items():
        tr = TaskResult(task=task)
        for rec in rows[:limit] if limit else rows:
            golds = {qid: schema.Gold(qid, g["type"], g["value"])
                     for qid, g in rec["gold"].items()}
            try:
                out = engine.decide_request({"state": rec["state"], "questions": rec["questions"]})
            except Exception as e:  # noqa: BLE001
                if len(tr.errors) < collect_errors:
                    tr.errors.append({"state": rec["state"][:120], "error": f"{type(e).__name__}: {e}"})
                tr.n += 1                      # 调用失败同样计为一次错误决策（不掩盖）
                continue
            answers = out.get("answers") or {}
            diags = out.get("diagnostics") or {}
            ok_all, conf = True, 0.0
            for qid, gold in golds.items():
                ans = answers.get(qid)
                correct = schema.answer_is_correct(gold, ans, score_tol=score_tol)
                if not correct:
                    ok_all = False
                conf += _answer_confidence(ans)
                # 每类 recall：均衡准确率的分子/分母。
                cs = tr.class_stats.setdefault(f"{gold.type}:{gold.value}", [0, 0])
                cs[1] += 1
                cs[0] += 1 if correct else 0
                # 预测分布：用于识别"恒答同一类"的退化行为。
                pk = _predicted_label(gold.type, ans)
                tr.pred_stats[pk] = tr.pred_stats.get(pk, 0) + 1
            tr.n += 1
            tr.correct += 1 if ok_all else 0
            tr.conf_sum += conf / max(1, len(golds))
            if all((diags.get(q) or {}).get("letter_ok") for q in golds):
                tr.letter_ok += 1
        res.per_task[task] = tr
        if verbose:
            flag = " [退化!]" if tr.degenerate else ""
            print(f"  {task:20s} n={tr.n:4d} acc={tr.accuracy:.3f} "
                  f"bal={tr.balanced_accuracy:.3f} 多数类={tr.majority_baseline:.3f} "
                  f"letter_ok={tr.letter_ok_rate:.3f}{flag}")
    return res


def evaluate_keyword_baseline(eval_dir: Path | str = EVAL_DIR, task: str = "routing",
                             verbose: bool = True) -> TaskResult:
    """关键词启发式基线（现有 agent/router.py 的逻辑），用同一批评测集打分。"""
    from ..agent.router import classify_heuristic

    rows = load_eval(eval_dir, [task]).get(task, [])
    tr = TaskResult(task=f"{task}(keyword)")
    for rec in rows:
        gold = next(iter(rec["gold"].values()))["value"]
        hits = classify_heuristic(rec["state"])
        tr.n += 1
        if gold in hits:
            tr.correct += 1
    if verbose:
        print(f"  {tr.task:20s} n={tr.n:4d} acc={tr.accuracy:.3f}")
    return tr


def main() -> None:
    ap = argparse.ArgumentParser(description="决策评测台")
    ap.add_argument("--models", default="", help="逗号分隔的 Ollama 模型名")
    ap.add_argument("--tasks", default="", help="逗号分隔的任务名，留空=全部")
    ap.add_argument("--limit", type=int, default=0, help="每个任务最多评测多少条")
    ap.add_argument("--keyword-only", action="store_true", help="只跑关键词基线")
    ap.add_argument("--out", default="", help="结果 JSON 落盘路径")
    args = ap.parse_args()

    tasks = [t for t in args.tasks.split(",") if t] or None
    limit = args.limit or None
    report: dict = {"eval_dir": str(EVAL_DIR), "limit": limit, "results": []}

    if args.keyword_only or not args.models:
        print("[keyword] 关键词启发式基线：")
        kr = evaluate_keyword_baseline(EVAL_DIR, "routing")
        report["results"].append(kr.as_dict())

    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"[model] {name}")
        eng = DecisionEngine(model=name)
        r = evaluate_model(eng, EVAL_DIR, tasks, limit)
        print(f"  --> macro_accuracy={r.macro_accuracy:.4f} letter_ok={r.letter_ok_rate:.4f}")
        report["results"].append(r.as_dict())

    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[report] 已写出 {args.out}")

    print("\n==== 汇总 ====")
    for r in report["results"]:
        if "macro_accuracy" in r:
            print(f"  {r['model']:22s} macro={r['macro_accuracy']:.4f} "
                  f"macro均衡={r['macro_balanced']:.4f} letter_ok={r['letter_ok_rate']:.4f}")
            if r.get("degenerate_tasks"):
                print(f"      退化任务（accuracy 不可信）：{', '.join(r['degenerate_tasks'])}")
        else:
            print(f"  {r['task']:22s} acc={r['accuracy']:.4f}")
    print("\n说明：macro 会奖励'恒答多数类'；macro均衡（各类召回等权）才反映真实能力。")


if __name__ == "__main__":
    main()

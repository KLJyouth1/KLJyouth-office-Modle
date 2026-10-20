# 历史报告（historical）

这里的文件是**早期版本**的评测报告与调研文档，**生成于评测集变更之前**，因此：

- **不能与 [`../eval/omnipilot17-eval-report.json`](../eval/omnipilot17-eval-report.json) 直接对比**。
  原因：`abstain` / `grounded` 等任务的**指令措辞**在此之后改过（为消除极性歧义），评测文件随之重新生成。
  同一个 `omnipilot16d` 在旧报告里 `abstain` 是 `acc=0.0 / letter_ok=0.0 / 退化`，在新报告里是 `0.500 / letter_ok=1.0` ——
  同一模型、同名任务、结论相反，正是"评测集变了"的直接证据。
- 保留它们的唯一目的是**留痕与追溯**（说明当时的结论建立在什么数据上）。
- **任何跨版本比较，都必须用同一批 `data/datasets/decision_eval/*.jsonl` 重跑全部模型**，而不是拿这些旧文件比。

| 文件 | 内容 |
|---|---|
| `omnipilot16-eval-report.json` | omnipilot16 时期评测（11 类任务，旧口径） |
| `omnipilot16b-eval-report.json` | omnipilot16b 时期评测（旧口径） |
| `omnipilot16d-eval-report.json` | omnipilot16d 时期评测（旧口径） |
| `omnipilot-kv-cache-report.json` | K/V cache 相关实测报告 |
| `omnipilot-architecture-roadmap.md` | 早期架构路线图（设计思路留档，**其中的计划不代表当前进度**） |

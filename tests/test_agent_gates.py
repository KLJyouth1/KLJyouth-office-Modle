"""OmniPilot 智能体闸门回归测试 —— 不依赖模型与网络。

覆盖本轮实测暴露的三个真实缺陷及其修复：
1. **假成功**：模型只写了 1 条要点（任务要求 3 条）、还把路径写错，agent 却报 `成功: True`。
   现在收尾前会**读回真实产物**核对"目标路径 + 条目数"，不达标就补写，补两次仍不达标就如实判失败。
2. **路径幻觉**：任务给了死路径，模型输出 `…\\yolo-ultralytics_\\points.txt`（真目录的同级兄弟）。
   现在任务唯一给定路径时必须逐字一致，否则拦下要求重写。
3. **画像自我投毒**：失败原文被当"已知事实"学进画像、又注入计划字段，模型据此判断
   "我正在失败"而放弃正常执行路径。现在失败轮次不学事实，且呈现端过滤历史毒数据。
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from omnipilot.agent.loop import (  # noqa: E402
    _SPEC_RETRY_BUDGET,
    _WRITE_TOOLS,
    _deliverable_spec_gap,
    _required_item_count,
    _should_realign_path,
    _task_needs_write,
    _task_target_path,
)
from omnipilot.agent.memory import Step  # noqa: E402
from omnipilot.personal.profile import PersonalProfile, _is_failure_noise  # noqa: E402
from omnipilot.prompts import _PLAN_PROMPT, _PLAN_SYSTEM  # noqa: E402


def write_step(path: str, ok: bool = True, tool: str = "file_write") -> Step:
    return Step(1, tool, {"path": path, "content": "x"}, ok, "已写入")


class TestRequiredItemCount:
    def test_reads_chinese_and_arabic_counts(self):
        assert _required_item_count("把三条不同的项目要点写入文件，每条一行") == 3
        assert _required_item_count("写 5 条要点") == 5
        assert _required_item_count("生成四个文件") == 4

    def test_returns_zero_when_the_task_specifies_no_count(self):
        assert _required_item_count("把结果写入文件") == 0
        assert _required_item_count("") == 0

    def test_clamps_implausible_counts_instead_of_inventing_a_constraint(self):
        # 30 条多半不是计数要求；宁可不校验，也不要造出假约束把模型逼进死循环。
        assert _required_item_count("写 30 条要点") == 0


class TestTaskTargetPath:
    def test_returns_the_single_explicit_path(self):
        got = _task_target_path(r"在 D:\Downloads\yolo-ultralytics\_t\points.txt 写入三条要点")
        assert got == r"D:\Downloads\yolo-ultralytics\_t\points.txt"

    def test_returns_none_when_several_paths_compete(self):
        # 两个候选无法判定哪个是交付目标 → 不做断言（避免假约束）。
        assert _task_target_path(r"比较 D:\a\one.txt 与 D:\b\two.txt") is None

    def test_returns_none_without_any_path(self):
        assert _task_target_path("写三条要点到文件") is None


class TestDeliverableSpecGap:
    def test_no_writes_defers_to_the_existing_gate(self):
        assert _deliverable_spec_gap(r"写入 D:\a\three.txt 三条要点", []) is None

    def test_rejects_a_write_to_the_wrong_directory(self, tmp_path):
        target = tmp_path / "sub" / "points.txt"
        wrong = tmp_path.parent / (tmp_path.name + "_") / "points.txt"

        gap = _deliverable_spec_gap(f"在 {target} 写入三条要点", [write_step(str(wrong))])

        assert gap is not None, "写错目录必须被拦下，而不是报成功"
        assert "逐字复制" in gap

    def test_rejects_too_few_items(self, tmp_path):
        target = tmp_path / "points.txt"
        target.write_text("要点一\n", encoding="utf-8")

        gap = _deliverable_spec_gap(f"在 {target} 写入三条要点", [write_step(str(target))])

        assert gap is not None, "只写 1 条（要求 3 条）必须被拦下"
        assert "1 条" in gap and "3 条" in gap

    def test_accepts_a_complete_artifact(self, tmp_path):
        target = tmp_path / "points.txt"
        target.write_text("要点一\n要点二\n要点三\n", encoding="utf-8")

        assert _deliverable_spec_gap(f"在 {target} 写入三条要点", [write_step(str(target))]) is None

    def test_ignores_failed_writes(self, tmp_path):
        target = tmp_path / "points.txt"
        assert _deliverable_spec_gap(
            f"在 {target} 写入三条要点", [write_step(str(target), ok=False)]
        ) is None

    def test_counts_only_non_blank_lines(self, tmp_path):
        target = tmp_path / "points.txt"
        target.write_text("要点一\n\n  \n要点二\n", encoding="utf-8")

        gap = _deliverable_spec_gap(f"在 {target} 写入三条要点", [write_step(str(target))])

        assert gap is not None and "2 条" in gap


class TestShouldRealignPath:
    """任务给死路径时必须逐字复制：宿主端只在"同名不同目录"时纠正，绝不猜路径。"""

    def test_realigns_a_near_miss_with_the_same_basename(self):
        assert _should_realign_path(
            r"D:\Downloads\yolo-ultralytics_\t\points.txt",
            r"D:\Downloads\yolo-ultralytics\_t\points.txt",
        )

    def test_keeps_an_already_correct_path(self):
        assert not _should_realign_path(r"D:\a\points.txt", r"D:\a\points.txt")
        # 大小写与分隔符差异不算"不同路径"
        assert not _should_realign_path(r"d:/A/points.txt", r"D:\a\points.txt")

    def test_refuses_to_guess_when_the_filename_differs(self):
        assert not _should_realign_path(r"D:\b\other.txt", r"D:\a\points.txt")

    def test_no_target_means_no_realign(self):
        assert not _should_realign_path(r"D:\b\points.txt", None)
        assert not _should_realign_path("", r"D:\a\points.txt")


class TestWriteToolsAreHonest:
    def test_read_is_not_counted_as_a_deliverable(self):
        # 旧版本把 file_read 列进 _WRITE_TOOLS，于是"读过一个文件"也算"已交付"。
        assert "file_read" not in _WRITE_TOOLS

    def test_task_needs_write_still_detects_write_tasks(self):
        assert _task_needs_write(r"把结果写入 D:\a\out.txt")
        assert not _task_needs_write("介绍一下你自己")

    def test_task_needs_write_is_order_independent(self):
        """实测缺陷：路径在前、动词在后时旧判据漏判 → 交付物闸门被整体跳过 → 零步假成功。"""
        assert _task_needs_write(
            r"在 D:\Downloads\yolo-ultralytics\_t\points.txt 写入三条不同的项目要点，每条一行"
        )
        assert _task_needs_write(r"D:\a\report.md 保存为最终稿")
        assert _task_needs_write("把 E:/tmp/data.csv 导出一份")

    def test_order_independent_path_does_not_over_trigger(self):
        # 只有动词或只有文件名，都不算"要产出文件"。
        assert not _task_needs_write("帮我写一份总结")
        assert not _task_needs_write("解释一下 out.txt 这个文件名的含义")
        assert not _task_needs_write("搜索 YOLO 的最新进展")


class TestSpecRetryBudget:
    """补写预算必须容得下"逐条 append"型轨迹。

    实测：v2 学的是"一条要点一次 file_write(append)"，写 3 条要 3 步；旧预算 2 会在
    第 2 条就放弃并判失败，把本可完成的 3 条任务错杀。
    """

    def test_budget_covers_a_per_item_append_trajectory(self):
        need = _required_item_count(r"在 D:\a\points.txt 写入三条不同的项目要点，每条一行")
        assert need == 3
        assert _SPEC_RETRY_BUDGET >= need, "补写预算必须 ≥ 任务要求的条数"

    def test_budget_stays_bounded(self):
        # 必须有限：否则模型可以无限补写，变成死循环。
        assert 3 <= _SPEC_RETRY_BUDGET <= 10


class TestPlannerIsToolAware:
    """规划器从未被训练过，必须靠提示把它约束到真实工具上。

    实测：不给工具清单时，1.5B 会把「在 …\\points.txt 写入三条要点」规划成
    「1. 打开记事本 2. 输入要点 3. 按 Ctrl+S」——全是工具清单里不存在的界面操作，
    后续 decide 照着错计划走，任务必然失败。
    """

    def test_plan_prompt_carries_the_tool_list(self):
        rendered = _PLAN_PROMPT.format(
            task="写入三条要点", tools="- file_write(path*, content*): 写入文本"
        )
        assert "file_write" in rendered, "规划提示必须带工具清单"
        assert "{tools}" not in rendered, "占位符必须已被填充"

    def test_plan_system_forbids_gui_operations(self):
        for word in ("鼠标", "键盘", "记事本"):
            assert word in _PLAN_SYSTEM, f"规划系统提示必须点名禁止界面操作：{word}"


class TestProfileStopsSelfPoisoning:
    def test_failure_noise_detection(self):
        assert _is_failure_noise("失败：达到最大步数(12)仍未完成")
        assert _is_failure_noise("exit=1")
        assert _is_failure_noise("com_error: (-2147...)")
        assert not _is_failure_noise("偏好简体中文")

    def test_failed_episodes_are_not_learned_as_facts(self, tmp_path):
        p = PersonalProfile.load(tmp_path / "p.json")

        p.learn_from_episode("写三条要点", "达到最大步数(12)仍未完成", ok=False)

        assert p.stats()["banks"].get("fact", 0) == 0, "失败不是事实，不得写入 fact 库"
        assert "失败" not in p.summary()

    def test_successful_episodes_are_learned_as_skills(self, tmp_path):
        p = PersonalProfile.load(tmp_path / "p.json")

        p.learn_from_episode("写三条要点", "已完成，三条都写好了", ok=True)

        assert "擅长/常用" in p.summary()

    def test_summary_filters_poison_already_on_disk(self, tmp_path):
        # 画像**是落盘的**：旧版本写进去的毒数据必须在呈现端也被过滤掉。
        p = PersonalProfile.load(tmp_path / "p.json")
        p.observe("fact", "旧任务", "失败：达到最大步数(12)仍未完成", signal=0.9)

        assert "达到最大步数" not in p.summary()

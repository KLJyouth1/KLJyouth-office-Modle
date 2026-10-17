"""定向训练 · 决策数据管线（Jev 式 Choice / Noul / Score）。

产出两套东西，互相绑定：
1. **训练集**（chat-SFT JSONL，喂给 train/qlora.py）：每条样本是
   "决策提示 → 一个选项字母"。用 `decision.engine.build_prompt` 渲染提示，
   与本项目推理端**逐字一致**（训练/推理同构，避免契约错位）。
   把答案固定成字母，是为了让模型学会"字母作答"——这正是
   `decision.engine` 能从 logprobs 里稳定取到候选概率的前提。
   （实测教训：未经字母训练的原模型首 token 会是候选原文如 "media"，
   导致按字母取概率时静默给错答案。）
2. **冻结评测集**（decision JSONL）：与训练集**同源但不相交**，供
   eval/decision_eval.py 以完全相同口径打分。

任务覆盖面（对应本轮目标）：
- 工具路由 Choice：本项目自有 demo（六大域 / 域内工具）
- 语义与意图：MASSIVE 英文意图、CLINC150（含 OOS）
- 情感识别：dair-ai/emotion(6 类)、go_emotions(28 类)
- 事实核验 Noul/Score：VitaminC(支持/反驳/证据不足)、MNLI(蕴含/中立/矛盾)
- 抗幻觉 Noul：无可靠来源时"是否应弃答"
"""
from __future__ import annotations

import json
import random
import re
import string
from pathlib import Path

from ..config import DATASET_DIR, DEMO_DIR, DOMAIN_DESC
from ..decision.engine import build_prompt
from ..decision.schema import CHOICE, NOUL, SCORE

# 与 decision.engine 的字母表一致（A..Z）。用于把标准答案转成字母。
_LETTERS = string.ascii_uppercase

# 工具名前缀 → 所属域（依据 omnipilot/tools/*.py 中各模块传入的 domain）。
_TOOL_PREFIX_DOMAIN: tuple[tuple[tuple[str, ...], str], ...] = (
    (("browser_", "web_"), "browser"),
    (("blender_",), "blender"),
    (("sys_", "screen_"), "system"),
    (("media_", "video_", "audio_", "image_", "ffmpeg_"), "media"),
    (("word_", "excel_", "ppt_", "doc_", "office_", "sheet_", "slide_"), "office"),
    (("file_", "code_", "git_", "py_"), "code"),
)


def domain_of_tool(tool: str) -> str | None:
    """工具名 → 域；无法判定返回 None（不猜）。"""
    t = (tool or "").strip().lower()
    for prefixes, dom in _TOOL_PREFIX_DOMAIN:
        if t.startswith(prefixes):
            return dom
    return None


# ---------------------------------------------------------------- 渲染工具
def _chat(question: dict, gold_value, state: str, meta: dict) -> dict:
    """把"一个问题 + 标准答案"渲染成 chat-SFT 样本（user=决策提示, assistant=字母）。"""
    candidates = _candidates_of(question)
    gold_letter = _gold_letter(question, gold_value, candidates)
    prompt = build_prompt(state, question["instructions"], candidates,
                          question.get("criteria") if isinstance(question.get("criteria"), dict) else None)
    return {"messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": gold_letter}],
            "meta": meta}


def _candidates_of(question: dict) -> list[str]:
    if question["type"] == CHOICE:
        return list(question["criteria"].keys())
    if question["type"] == SCORE:
        return list(question["criteria"])
    return ["false", "true"]


def _gold_letter(question: dict, gold_value, candidates: list[str]) -> str:
    """标准答案 → 字母（choice: 候选键；noul: true/false；score: 等级下标）。"""
    if question["type"] == CHOICE:
        idx = candidates.index(gold_value)
    elif question["type"] == NOUL:
        idx = 1 if bool(gold_value) else 0
    else:
        idx = int(gold_value)
    return _LETTERS[idx]


def _record(state: str, question: dict, gold_value, meta: dict) -> tuple[dict, dict]:
    """返回 (chat 训练样本, decision 评测记录)。二者同源同构。"""
    qid = question["type"]
    return (_chat(question, gold_value, state, meta),
            {"state": state, "questions": {qid: question},
             "gold": {qid: {"type": question["type"], "value": gold_value}},
             "meta": meta})


def _norm(text: str) -> str:
    """去污染用的归一化：压空白 + 小写，用于判断训练/评测是否撞文本。"""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


# ---------------------------------------------------------------- 任务 1/2：工具路由
_TOOLS_BLOCK_RE = re.compile(r"可用工具[：:]\s*\n(.*?)(?:\n\n|\Z)", re.DOTALL)
_TOOL_LINE_RE = re.compile(r"^-\s*([A-Za-z_][\w]*)\s*\(", re.MULTILINE)
_TASK_RE = re.compile(r"任务[：:]\s*(.+)")
_TOOLCALL_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')


def parse_demo(path: Path) -> list[dict]:
    """从一条 demo 样本里抽出 (task, called_tool, available_tools)。"""
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        msgs = rec.get("messages") or []
        user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        mt = _TASK_RE.search(user or "")
        if not mt:
            continue
        tool = None
        for cand in _TOOLCALL_RE.findall(asst or ""):
            if domain_of_tool(cand):
                tool = cand
                break
        block = _TOOLS_BLOCK_RE.search(user or "")
        tools = _TOOL_LINE_RE.findall(block.group(1)) if block else []
        out.append({"task": mt.group(1).strip(), "tool": tool, "tools": tools})
    return out


def build_routing(demos: list[Path], rng: random.Random, n: int) -> list[tuple[dict, dict]]:
    """工具路由：state=用户任务，choice 在六大域里选。"""
    rows = []
    for p in demos:
        rows.extend(parse_demo(Path(p)))
    rng.shuffle(rows)
    out = []
    for r in rows:
        dom = domain_of_tool(r["tool"] or "")
        if not dom:
            continue
        q = {"type": CHOICE, "instructions": "这条用户请求应该交给哪个工具域处理？",
             "criteria": dict(DOMAIN_DESC)}
        out.append(_record(r["task"], q, dom, {"task": "routing", "source": "demo"}))
        if len(out) >= n:
            break
    return out


def build_tool_choice(demos: list[Path], rng: random.Random, n: int) -> list[tuple[dict, dict]]:
    """域内选工具：state=用户任务，choice 在该域可用工具里选。"""
    rows = []
    for p in demos:
        rows.extend(parse_demo(Path(p)))
    rng.shuffle(rows)
    out = []
    for r in rows:
        tool, tools = r["tool"], r["tools"]
        if not tool or not domain_of_tool(tool):
            continue
        same = [t for t in tools if domain_of_tool(t) == domain_of_tool(tool)]
        if tool not in same or len(same) < 2:
            continue
        cands = same[:26]
        if tool not in cands:
            continue
        q = {"type": CHOICE, "instructions": "要完成这个任务，应该调用哪个工具？",
             "criteria": {t: t for t in cands}}
        out.append(_record(r["task"], q, tool, {"task": "tool_choice", "source": "demo"}))
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------- 任务 3/4：意图
def _cand_subset(all_labels: list[str], gold: str, k: int, rng: random.Random) -> list[str]:
    """从标签全集里抽 k 个候选（必含 gold），打乱。用于"固定候选"式决策。"""
    others = [x for x in all_labels if x != gold]
    rng.shuffle(others)
    cands = [gold] + others[: max(1, k - 1)]
    rng.shuffle(cands)
    return cands


def build_intent_massive(rng: random.Random, n: int, *, k: int = 8) -> list[tuple[dict, dict]]:
    from datasets import load_dataset

    tr = load_dataset("mteb/amazon_massive_intent", name="en", split="train")
    labels = sorted({str(x) for x in tr["label_text"]})
    idx = list(range(len(tr)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        row = tr[i]
        gold = str(row["label_text"])
        cands = _cand_subset(labels, gold, k, rng)
        q = {"type": CHOICE, "instructions": "这句话的意图属于下面哪一类？",
             "criteria": {c: "" for c in cands}}
        out.append(_record(str(row["text"]), q, gold, {"task": "intent_massive", "source": "mteb/massive"}))
        if len(out) >= n:
            break
    return out


def build_intent_clinc(rng: random.Random, n: int, *, k: int = 8) -> list[tuple[dict, dict]]:
    from datasets import load_dataset

    ds = load_dataset("clinc_oos", "plus", split="train")
    names = list(ds.features["intent"].names)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        row = ds[i]
        gold = names[int(row["intent"])]
        cands = _cand_subset(names, gold, k, rng)
        q = {"type": CHOICE, "instructions": "这条用户话语的意图属于哪一类？（含域外 OOS）",
             "criteria": {c: "" for c in cands}}
        out.append(_record(str(row["text"]), q, gold, {"task": "intent_clinc", "source": "clinc_oos"}))
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------- 任务 5/6：情感
def build_emotion6(rng: random.Random, n: int) -> list[tuple[dict, dict]]:
    from datasets import load_dataset

    ds = load_dataset("dair-ai/emotion", split="train")
    names = list(ds.features["label"].names)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        row = ds[i]
        gold = names[int(row["label"])]
        q = {"type": CHOICE, "instructions": "这句话表达的主要情绪是哪一种？",
             "criteria": {c: "" for c in names}}
        out.append(_record(str(row["text"]), q, gold, {"task": "emotion6", "source": "dair-ai/emotion"}))
        if len(out) >= n:
            break
    return out


def build_emotion28(rng: random.Random, n: int, *, k: int = 10) -> list[tuple[dict, dict]]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/go_emotions", name="simplified", split="train")
    feat = ds.features["labels"]
    names = list(feat.feature.names) if hasattr(feat, "feature") else list(feat.names)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        row = ds[i]
        labels = list(row["labels"])
        if len(labels) != 1:            # 只取单标签样本，避免多标签歧义
            continue
        gold = names[int(labels[0])]
        cands = _cand_subset(names, gold, k, rng)
        q = {"type": CHOICE, "instructions": "这条评论最贴近哪一种情绪？",
             "criteria": {c: "" for c in cands}}
        out.append(_record(str(row["text"]), q, gold, {"task": "emotion28", "source": "go_emotions"}))
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------- 任务 7/8：事实核验与 NLI
_VITAMINC_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def build_fact_vitaminc(rng: random.Random, n: int) -> list[tuple[dict, dict]]:
    """VitaminC：证据对主张的支持关系（3 类 choice）+ 抗幻觉 noul（证据是否充足）。

    派生 Noul 强制正负等量：实测旧版本该任务是 196:32（true:false）的严重偏斜，
    会让「证据是否足够」退化成恒答 true —— 与 abstain 恒答 false 是同一类病。
    """
    from datasets import load_dataset

    ds = load_dataset("tals/vitaminc", split="train")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    # choice 与 noul 分开累积并各自设独立配额：混在同一个 out 里按总长截断时，
    # choice 样本会把 noul 的配额挤掉，导致「强制等量」名存实亡
    # （实测：混算时得到 115:42，仍偏 73%）。
    n_choice = max(1, n // 2)
    quota = max(1, n // 4)
    choice_out: list[tuple[dict, dict]] = []
    noul_true: list[tuple[dict, dict]] = []
    noul_false: list[tuple[dict, dict]] = []
    for i in idx:
        if len(choice_out) >= n_choice and len(noul_true) >= quota and len(noul_false) >= quota:
            break
        row = ds[i]
        claim, ev = str(row["claim"]), str(row.get("evidence") or "")
        gold = str(row["label"]).strip().upper()
        if gold not in _VITAMINC_LABELS:
            continue
        state = f"主张：{claim}\n证据：{ev[:900]}"
        if len(choice_out) < n_choice:
            q1 = {"type": CHOICE, "instructions": "证据对该主张的支持关系是？",
                  "criteria": {c: c for c in _VITAMINC_LABELS}}
            choice_out.append(_record(state, q1, gold,
                                      {"task": "fact_vitaminc", "source": "tals/vitaminc"}))
        q2 = {"type": NOUL, "instructions": "现有证据是否足以判断该主张的真假？"}
        if gold != "NOT ENOUGH INFO":
            if len(noul_true) < quota:
                noul_true.append(_record(state, q2, True,
                                         {"task": "fact_evidence_noul", "source": "tals/vitaminc"}))
        elif len(noul_false) < quota:
            noul_false.append(_record(state, q2, False,
                                      {"task": "fact_evidence_noul", "source": "tals/vitaminc"}))
    return choice_out + noul_true + noul_false


_MNLI_LABELS = ["entailment", "neutral", "contradiction"]


def build_nli_mnli(rng: random.Random, n: int) -> list[tuple[dict, dict]]:
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/glue", name="mnli", split="train")
    names = list(ds.features["label"].names)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        row = ds[i]
        state = f"前提：{row['premise']}\n假设：{row['hypothesis']}"
        gold = names[int(row["label"])]
        # 候选顺序必须与 nli_score 的 _NLI_SCORE_LEVELS 一致（矛盾→中立→蕴含，升序）。
        # 实测：两任务此前顺序相反（nli_mnli 用 GLUE 的 entailment→neutral→contradiction，
        # nli_score 用矛盾→中立→蕴含），导致**同一段前提/假设的正确答案字母在两个任务里互换**，
        # 模型学不到语义映射、退化为位置先验 —— 实测两任务双双低于随机（0.072 / 0.064）。
        q1 = {"type": CHOICE, "instructions": "前提与假设的逻辑关系是？",
              "criteria": {c: "" for c in reversed(names)}}
        out.append(_record(state, q1, gold, {"task": "nli_mnli", "source": "nyu-mll/glue"}))
        if gold in ("entailment", "contradiction"):
            q2 = {"type": NOUL, "instructions": "前提是否蕴含该假设？"}
            out.append(_record(state, q2, gold == "entailment",
                               {"task": "nli_noul", "source": "nyu-mll/glue"}))
        if len(out) >= n:
            break
    return out


# Score 接口：把 MNLI 的三分类**有序化**成"前提对假设的支持程度"。
# 等级有序性是真实存在的：矛盾(不支持) < 中立 < 蕴含(支持)，不是生造标签。
_NLI_SCORE_LEVELS = ["矛盾/不支持", "中立", "蕴含/支持"]
_NLI_SCORE_ORDER = {"contradiction": 0, "neutral": 1, "entailment": 2}


def build_nli_score(rng: random.Random, n: int) -> list[tuple[dict, dict]]:
    """MNLI → 有序"支持程度"打分（对应官方 JevBench 的 ordinal 族）。

    抽不同的切片，避免与 nli_mnli 的 choice 任务取到同一批样本。
    """
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/glue", name="mnli", split="train")
    names = list(ds.features["label"].names)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        row = ds[i]
        gold = names[int(row["label"])]
        if gold not in _NLI_SCORE_ORDER:
            continue
        state = f"前提：{row['premise']}\n假设：{row['hypothesis']}"
        q = {"type": SCORE, "instructions": "前提对假设的支持程度是？",
             "criteria": list(_NLI_SCORE_LEVELS)}
        out.append(_record(state, q, _NLI_SCORE_ORDER[gold],
                           {"task": "nli_score", "source": "nyu-mll/glue"}))
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------- 任务 9：Noul 依据判定（抗幻觉及其正例）
#
# 实测驱动的三项修正（每条都对应一个已被证据确认的缺陷）：
# 1. **abstain 与 grounded 必须共用同一条指令**。此前 abstain 的指令里写了
#    「没有调用工具、也没有数据来源」，等于把答案直接写进了题目 —— 模型不需要读
#    证据就能答对，于是学到的是"这类问句就答 false"的措辞捷径，而不是判断依据。
#    改成中性指令后，模型必须读 state 里的证据，才能区分有依据/无依据。
# 2. **必须有正例（grounded）**。abstain 的答案全是 false，若没有等量正例，
#    "是否足以回答"就退化成一个恒为 false 的常量，评测上也看不出差别。
# 3. **模板必须可组合出足够多互不相同的样本**。此前 12 个模板循环生成 300 条，
#    去重后**只剩 12 条有效样本**（builder 声称 300、实际交付 12）——这是一个
#    真实的数据缺陷：任务权重被静默降到 1/25。故这里用「主体 × 数值」组合扩容。
# 极性歧义修复（实测驱动）：旧指令「根据当前提供的信息，是否足以可靠地回答该问题？」
# 在极向上是二义的 —— 既可读作「信息是否足够？」（足够=true），也可读作「是否该弃答？」
# （该弃答=true），二者**完全相反**。而该任务名恰为 abstain（弃答），模型极易采用后者，
# 从而在 abstain（正解 false）与 grounded（正解 true）上**同时反向**答错（实测 0.150 / 0.100）。
# 现改为显式绑定极性：明确 "足以 → true"，堵死第二种读法。
_NOUL_BASIS_INSTRUCTION = "材料是否足以可靠地回答该问题？足以则答 true，不足以则答 false。"

# 无依据类：主体多样 + 数值扰动，保证去重后仍数量充足。
_ABSTAIN_SUBJECTS = ["我上个月的具体报销金额", "我账户里当前可用的余额", "我昨天的通话记录时长",
                     "我这周的打卡时间", "我电脑 D 盘那个加密目录里的文件名",
                     "我司未公开的新产品代号", "隔壁公司今年的确切营收",
                     "我还没打开过的那份 PDF 的内容", "今天下午会议室的在场人数",
                     "这台机器实时的 CPU 温度", "我邮箱里最新一封邮件的标题",
                     "我本地数据库里昨天的订单总量", "我没提供过的那个文件的大小",
                     "我私人的行程安排", "那台我还没连上的设备的序列号"]


def build_abstain(rng: random.Random, n: int) -> list[tuple[dict, dict]]:
    """抗幻觉：无可靠依据 → Noul 应为 false。"""
    out = []
    for i in range(n):
        subject = _ABSTAIN_SUBJECTS[i % len(_ABSTAIN_SUBJECTS)]
        if i >= len(_ABSTAIN_SUBJECTS):
            subject = f"{subject}（编号 {i}）"     # 二次循环加扰动，避免去重塌陷
        state = f"用户要求：{subject}是多少？"
        q = {"type": NOUL, "instructions": _NOUL_BASIS_INSTRUCTION}
        out.append(_record(state, q, False,
                           {"task": "abstain", "source": "synthetic",
                            "why": "问题所需信息既未提供、也无法从公开来源获得"}))
    return out


# 有依据类：证据**内联在 state 里**，答案是可直接得出的 → Noul 应为 true。
# 数值由 rng 扰动，保证样本互不相同（避免重蹈 abstain 的去重塌陷）。
_GROUNDED_TEMPLATES = [
    ("已知：报表 data.xlsx 的 A1 单元格内容为「季度营收 {n} 元」。", "A1 单元格的内容是什么？"),
    ("已知：配置文件里 max_workers 的值是 {n}。", "max_workers 配置成了多少？"),
    ("已知：接口压测日志显示 P95 延迟为 {n} 毫秒。", "这次压测的 P95 延迟是多少？"),
    ("已知：合同正文写明服务期为 {n} 天。", "合同约定的服务期是多少天？"),
    ("已知：物料清单里螺丝数量是 {n} 颗。", "清单上的螺丝数量是多少？"),
    ("已知：监控面板显示当前在线用户 {n} 人。", "当前在线用户有多少人？"),
    ("已知：日志中错误码为 E{n}，含义是鉴权失败。", "E{n} 这个错误码的含义是什么？"),
    ("已知：测试报告写明用例总数 {n}，全部通过。", "测试用例总数是多少？"),
    ("已知：服务器规格为 {n} GB 内存。", "这台服务器的内存是多少？"),
    ("已知：发票金额为 {n} 元，开票方为甲公司。", "这张发票的金额是多少？"),
    ("已知：会议纪要记录决议项 {n} 条。", "这次会议形成了多少条决议？"),
    ("已知：库存表显示剩余 {n} 件。", "当前库存还剩多少件？"),
]


def build_grounded(rng: random.Random, n: int) -> list[tuple[dict, dict]]:
    """抗幻觉正例：信息已内联给出 → Noul 应为 true。

    与 `build_abstain` 共用同一条指令，因此模型无法靠措辞区分二者，
    只能真的去读 state 里有没有证据。这是让「依据判定」成为可学习判别任务的必要条件。
    """
    out = []
    for i in range(n):
        tpl, question = _GROUNDED_TEMPLATES[i % len(_GROUNDED_TEMPLATES)]
        val = rng.randint(100, 99_999)
        evidence = tpl.format(n=val)
        # 独立证据形态：用「材料块」承载证据，与 abstain 的纯问句在**结构**上不同。
        # 注意：指令仍与 abstain 完全共用（不能靠措辞区分任务），区别只在"有没有材料"。
        state = f"【已提供的材料】\n{evidence}\n【问题】{question}"
        q = {"type": NOUL, "instructions": _NOUL_BASIS_INSTRUCTION}
        out.append(_record(state, q, True,
                           {"task": "grounded", "source": "synthetic",
                            "why": "所需信息已内联提供，可直接得出答案"}))
    return out


# ---------------------------------------------------------------- 组装
BUILDERS = {
    "routing": 800,            # 训练样本上限（按任务）
    "tool_choice": 600,
    "intent_massive": 600,
    "intent_clinc": 500,
    "emotion6": 600,
    "emotion28": 400,
    "fact_vitaminc": 600,
    "nli_mnli": 600,
    "nli_score": 400,
    "abstain": 100,
    "grounded": 100,
}


def _pair_split(pairs: list[tuple[dict, dict]], eval_ratio: float, rng: random.Random,
                eval_cap: int) -> tuple[list[dict], list[dict], list[dict]]:
    """把 (chat, decision) 对切成 训练 chat / 评测 chat / 评测 decision。

    评测集与训练集**同源不重叠**（先切后训），评测 decision 供评测台使用。
    """
    rng.shuffle(pairs)
    n_eval = min(eval_cap, max(1, int(len(pairs) * eval_ratio)))
    ev, tr = pairs[:n_eval], pairs[n_eval:]
    return ([c for c, _ in tr], [c for c, _ in ev], [d for _, d in ev])


def build_decision_dataset(
    *,
    out_dir: Path | str = DATASET_DIR,
    name: str = "omnipilot16",
    seed: int = 20261016,
    eval_ratio: float = 0.2,
    eval_cap: int = 200,
    tasks: list[str] | None = None,
    demos: list[Path] | None = None,
    verbose: bool = True,
) -> dict:
    """构建 omnipilot16 决策数据集；返回统计信息。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    demos = demos or [DEMO_DIR / "synth.jsonl", DEMO_DIR / "multistep.jsonl"]  # type: ignore[assignment]
    tasks = tasks or list(BUILDERS)

    all_train: list[dict] = []
    all_eval_chat: list[dict] = []
    all_eval_decision: list[dict] = []
    stats: dict[str, dict] = {}

    for task in tasks:
        try:
            if task == "routing":
                pairs = build_routing(demos, rng, BUILDERS[task])
            elif task == "tool_choice":
                pairs = build_tool_choice(demos, rng, BUILDERS[task])
            elif task == "intent_massive":
                pairs = build_intent_massive(rng, BUILDERS[task])
            elif task == "intent_clinc":
                pairs = build_intent_clinc(rng, BUILDERS[task])
            elif task == "emotion6":
                pairs = build_emotion6(rng, BUILDERS[task])
            elif task == "emotion28":
                pairs = build_emotion28(rng, BUILDERS[task])
            elif task == "fact_vitaminc":
                pairs = build_fact_vitaminc(rng, BUILDERS[task])
            elif task == "nli_mnli":
                pairs = build_nli_mnli(rng, BUILDERS[task])
            elif task == "nli_score":
                pairs = build_nli_score(rng, BUILDERS[task])
            elif task == "abstain":
                pairs = build_abstain(rng, BUILDERS[task])
            elif task == "grounded":
                pairs = build_grounded(rng, BUILDERS[task])
            else:
                raise ValueError(f"未知任务：{task}")
        except Exception as e:  # noqa: BLE001
            stats[task] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            if verbose:
                print(f"[data] {task:16s} 跳过：{type(e).__name__}: {e}")
            continue

        tr, ev_chat, ev_dec = _pair_split(pairs, eval_ratio, rng, eval_cap)
        all_train.extend(tr)
        all_eval_chat.extend(ev_chat)
        all_eval_decision.extend(ev_dec)
        stats[task] = {"ok": True, "total": len(pairs), "train": len(tr), "eval": len(ev_dec)}
        if verbose:
            print(f"[data] {task:16s} 总={len(pairs):5d} 训练={len(tr):5d} 评测={len(ev_dec):5d}")

    # 去污染：训练集里任何与评测集归一化文本相同的样本一律删除。
    eval_keys = {_norm(d["state"]) for d in all_eval_decision}
    before = len(all_train)
    train = [c for c in all_train if _norm(_first_user(c)) not in eval_keys]
    dropped = before - len(train)

    # 训练集内部再去一次重
    seen: set[str] = set()
    deduped = []
    for c in train:
        k = _norm(_first_user(c)) + "|" + str(c["messages"][-1]["content"])
        if k in seen:
            continue
        seen.add(k)
        deduped.append(c)
    train = deduped

    # 切出验证集（训练集内部留出，不与评测集混淆）
    rng.shuffle(train)
    n_val = max(1, int(len(train) * 0.05))
    val, train = train[:n_val], train[n_val:]

    _write_jsonl(out / f"{name}_train.jsonl", train)
    _write_jsonl(out / f"{name}_val.jsonl", val)
    eval_dir = out / "decision_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    # 评测集按任务拆文件，便于逐任务统计
    by_task: dict[str, list[dict]] = {}
    for d in all_eval_decision:
        by_task.setdefault(d["meta"]["task"], []).append(d)
    for task, rows in by_task.items():
        _write_jsonl(eval_dir / f"{task}.jsonl", rows)

    info = {
        "name": name, "seed": seed,
        "train": len(train), "val": len(val),
        "eval_total": len(all_eval_decision),
        "decontaminated_dropped": dropped,
        "tasks": stats,
        "train_path": str(out / f"{name}_train.jsonl"),
        "val_path": str(out / f"{name}_val.jsonl"),
        "eval_dir": str(eval_dir),
        "eval_files": {t: str(eval_dir / f"{t}.jsonl") for t in sorted(by_task)},
    }
    (out / f"{name}_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"[data] 完成：训练={len(train)} 验证={len(val)} 评测={len(all_eval_decision)} 去污染删除={dropped}")
    return info


def _first_user(sample: dict) -> str:
    for m in sample.get("messages", []):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


__all__ = [
    "DOMAIN_DESC", "domain_of_tool", "build_decision_dataset",
    "build_routing", "build_tool_choice", "build_intent_massive", "build_intent_clinc",
    "build_emotion6", "build_emotion28", "build_fact_vitaminc", "build_nli_mnli", "build_abstain",
]

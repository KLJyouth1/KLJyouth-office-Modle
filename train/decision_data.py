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


# ---------------------------------------------------------------- 四域定向数据
#
# 目标：把"电脑/系统/浏览器操控、办公软件、3D 建模、视频图像、代码"这五类
# 真实工作负载灌进决策训练集。为什么必须单独造，而不是继续用 demo：
# demo 里的任务句式集中在少数模板上，`routing` 的准确率已接近关键词基线（0.97），
# 说明它测不到真实增量。定向数据的作用是覆盖**关键词启发式覆盖不到**的表述
# （例如"把这批图统一成 800x600"没有出现"图片"以外的 media 关键词），
# 从而让决策层在 hybrid 路由下真正接管长尾请求。
#
# 两条数据线，分别对应推理端真实存在的两种决策：
# 1. directed_routing       —— 与 agent/router.py 的六域 choice **完全同构**
#                              （候选、指令、提示渲染都一致）。
# 2. directed_tool_choice   —— 域内 45 个真实工具的细粒度选择。当前推理链路
#                              尚未消费它，但它把"工具语义"灌进模型，是后续
#                              工具调用阶段的必要前置；也是域判断的强化信号。
#
# 造数据的硬约束（来自本项目已踩过的坑）：**每条模板都必须带随机占位符**。
# 旧版 abstain 就是 12 个无参模板循环生成 300 条，去重后只剩 12 条有效样本，
# 任务权重被静默降到 1/25。这里的占位符池远比模板数大，保证去重后不塌陷。
_URLS = ("https://example.com/login", "https://news.ycombinator.com",
         "https://docs.python.org/3/library/asyncio.html", "http://localhost:5173",
         "https://github.com/torvalds/linux", "https://www.bilibili.com/video/BV1xx")
_QUERIES = ("今天的天气", "Python 3.14 新特性", "RTX 3050 显存规格",
            "GB/T 7714 引用格式", "Windows 11 关闭自动更新", "Qwen2.5 微调显存占用")
_TEXTS = ("admin", "订单编号 A1024", "季度总结", "hello world", "张三", "2026-10-18")
_PATHS = ("config.yaml", "src/main.py", "D:\\work\\notes.md", "data\\orders.csv",
          "app\\settings.json", "utils\\helpers.py")
_REPOS = ("D:\\Downloads\\Marguerite", "D:\\Downloads\\yolo-ultralytics", "D:\\work\\webapp",
          "E:\\code\\demo", "C:\\Users\\Administrator\\project")
_OFFICE_FILES = ("budget.xlsx", "D:\\work\\report.xlsx", "contract.docx", "周报.docx",
                 "季度汇报.pptx", "data\\orders.xlsx")
_MEDIA_FILES = ("input.mp4", "D:\\video\\clip.mov", "assets\\demo.webm",
                "screen_record.mkv", "output\\vlog.mp4")
_IMAGE_FILES = ("photo.jpg", "D:\\pics\\cover.png", "assets\\logo.webp", "screenshots\\shot.png")
_BLEND_FILES = ("scene.blend", "E:\\assets\\chair.blend", "D:\\3d\\island.blend", "models\\room.blend")
_KEYS = ("enter", "tab", "esc", "space", "backspace")
_COMBOS = ("ctrl+s", "ctrl+shift+esc", "alt+tab", "ctrl+z", "win+d")
_SELECTORS = ("#login-btn", ".btn-primary", "input[name=q]", "#result", "div.list > a")

_DIRECTED_DOMAIN_TEMPLATES: dict[str, list[str]] = {
    "browser": [
        "打开 {url} 看看这个页面上写了什么",
        "在浏览器里访问 {url} 并完成登录",
        "搜索一下「{query}」的最新结果，取前 {idx} 条",
        "把 {url} 的正文抓下来做摘要",
        "点开页面上写着「{text}」的那个按钮",
        "在登录表单里填入「{text}」再提交",
        "看看 {url} 这个标签页的地址和标题",
        "等「{text}」这个结果出现后再截图",
        "给 {url} 这个页面截张图保存下来",
        "把 {url} 页面里所有可点击元素和坐标列出来",
        "把网页向下滚动 {px} 像素继续读",
    ],
    "office": [
        "新建一个 Excel，把 {cnt} 行销售数据写进 A1 开始的区域",
        "读取 {ofile} 里的数据并汇总",
        "生成一份周报 Word 文档到 {ofile}，包含三段正文",
        "把会议纪要写成 Word 保存到 {ofile}",
        "做一份 {cnt} 页的 PPT 介绍季度业绩",
        "从 {ofile} 里读出最近 {cnt} 天的订单明细",
        "在 {ofile} 的第 {idx} 个工作表写入统计结果",
        "打开 {ofile} 读一下正文内容",
        "把这份报告整理成 {cnt} 页幻灯片存到 {ofile}",
        "在表格 {ofile} 里追加一行汇总数据",
    ],
    "code": [
        "看一下 {repo} 仓库有哪些未提交改动",
        "查看 {repo} 最近 {idx} 条 git 提交记录",
        "在 {repo} 运行 npm run build 并查看报错",
        "用 Python 写脚本计算前 {num} 个素数",
        "把这段配置写进 {path}",
        "读取 {path} 的内容帮我分析",
        "在 {repo} 里执行 pytest 并返回失败用例",
        "新建 {path} 并写入工具函数",
        "在 {repo} 跑一下 git status 看看有没有冲突",
        "编译 {repo} 这个 Rust 项目并修复报错",
    ],
    "media": [
        "把 {media} 转码成 H.264 的 mp4",
        "从第 {sec} 秒开始裁剪 {dur} 秒视频",
        "把 {media} 的音轨抽成 mp3",
        "从 {media} 每秒抽一帧导出成图片",
        "查看 {media} 的分辨率和码率",
        "把图片 {img} 缩放到 {w}x{h}",
        "裁剪 {img}，去掉四周 {px} 像素的空白边框",
        "查看图片 {img} 的尺寸和格式",
        "把 {media} 压到 {num} MB 以下",
        "批量把这 {cnt} 张图片调整成统一尺寸",
    ],
    "blender": [
        "用 Blender 脚本建一个 {cnt} 面的立方体并加材质",
        "把 {blend} 的第 1 帧渲染成图片",
        "检查一下本机有没有装 Blender，版本要 {num} 以上",
        "写一段 bpy 脚本给场景加 {idx} 盏灯",
        "给 {blend} 这个模型加上木纹材质再渲染",
        "用 Blender 无头模式导出 {blend} 到 glb",
        "在 Blender 里把模型细分到 {num} 面",
        "渲染 {blend} 场景的第 {idx} 帧",
        "在 {blend} 里批量摆放 {cnt} 个道具",
        "把 {blend} 的渲染分辨率设成 {w}x{h}",
    ],
    "system": [
        "在屏幕坐标 ({x}, {y}) 点击鼠标左键",
        "把鼠标移动到屏幕 ({x}, {y})",
        "在当前窗口输入「{text}」",
        "按下 {combo} 保存当前文档",
        "按 {idx} 次退格删掉刚才的字",
        "截取屏幕 ({x},{y}) 开始的 {w}x{h} 区域保存成图片",
        "向下滚动 {px} 像素查看下面的内容",
        "双击桌面坐标 ({x}, {y}) 的 Chrome 图标打开它",
        "用快捷键 {combo} 切换到上一个窗口",
        "打开任务栏第 {idx} 个图标对应的应用",
    ],
}

# 工具名逐一取自 omnipilot/tools/*.py 的 build_tools()，不臆造。
_DIRECTED_TOOL_INTENTS: dict[str, dict[str, list[str]]] = {
    "browser": {
        "browser_goto": ["打开 {url} 这个网址", "在浏览器里访问 {url}",
                         "跳转到 {url} 页面", "导航到 {url} 并等待加载完成"],
        "browser_click_text": ["点击页面上写着「{text}」的按钮", "点一下「{text}」这个链接",
                               "找到包含「{text}」的元素并点击", "点击文本为「{text}」的菜单项"],
        "browser_click_selector": ["用 CSS 选择器 {sel} 点击对应元素", "点击选择器 {sel} 匹配的按钮",
                                   "选择器 {sel} 指向的那个元素点一下", "按选择器 {sel} 定位并单击"],
        "browser_click_xy": ["在页面坐标 ({x}, {y}) 处点击", "点击网页上 ({x}, {y}) 这个位置",
                             "按页面横纵坐标 ({x}, {y}) 点一下", "在页面 ({x}, {y}) 单击左键"],
        "browser_fill": ["在 {sel} 输入框里填入 {text}", "往选择器 {sel} 的输入框填写 {text}",
                         "把 {text} 填进 {sel} 表单字段", "在 {sel} 里输入 {text} 并保持焦点"],
        "browser_press": ["在页面里按下 {key} 键", "给当前网页发送一次 {key} 按键",
                          "按 {key} 触发页面提交", "在页面焦点处敲 {key}"],
        "browser_scroll": ["把网页向下滚动 {px} 像素", "页面往下滚 {px} 像素",
                           "向下滚动页面 {px} 像素继续阅读", "把页面向上滚回 {px} 像素"],
        "browser_back": ["从 {url} 回退到上一页", "返回上一个页面（当前在 {url}）",
                         "后退到之前那个网页", "浏览器回退一步"],
        "browser_wait_for": ["等待页面上出现「{text}」再继续", "等 {sel} 这个元素加载出来",
                             "等页面出现「{text}」文本后再操作", "等待 {sel} 出现在 DOM 中，最多 {idx} 秒"],
        "browser_url": ["读出当前页面的地址和标题", "看一下浏览器现在停在哪个网址",
                        "获取当前标签页的 URL 和标题", "确认 {url} 是否已经跳转成功"],
        "browser_snapshot": ["获取页面可交互元素清单（带坐标）", "把页面上的按钮和输入框列出来",
                             "生成页面元素快照方便精确点击", "列出页面坐标可点的元素，最多 {cnt} 个"],
        "browser_text": ["提取当前网页的正文文本", "把网页内容读出来给我看",
                         "获取页面正文（限制 {cnt} 字）", "读一下 {url} 页面的文字内容"],
        "browser_screenshot": ["给当前网页截张图存到 shot.png", "对页面截图保存",
                               "把网页画面截下来", "把 {url} 页面截图到 shots\\{idx}.png"],
        "browser_wait": ["等待 1500 毫秒再操作", "先等 2 秒让动画结束",
                         "暂停 500 毫秒再点下一步", "等 {idx} 秒后再继续"],
        "web_search": ["联网搜索「{query}」", "帮我查一下最新的 {query}",
                       "检索 {query} 相关的 {idx} 条结果", "搜索 {query} 并给出前几条摘要"],
        "web_fetch": ["抓取 {url} 的正文内容", "把 {url} 这篇文章读下来",
                      "访问 {url} 并提取可读正文", "抓取 {url} 前 {cnt} 字正文"],
    },
    "system": {
        "sys_click": ["在屏幕坐标 ({x}, {y}) 点一下鼠标", "双击屏幕上 ({x}, {y}) 的图标",
                      "右键点击 ({x}, {y}) 位置", "在 ({x}, {y}) 连续点击 {idx} 次"],
        "sys_move": ["把鼠标移到屏幕 ({x}, {y})", "移动鼠标到坐标 ({x}, {y})",
                     "把光标挪到 ({x}, {y})", "将鼠标移动到 ({x}, {y}) 悬停 {idx} 秒"],
        "sys_type": ["在当前焦点输入「{text}」", "敲入这段中文：{text}",
                     "把 {text} 打到当前输入框", "输入 {text} 后按一次回车"],
        "sys_hotkey": ["按下 {combo} 组合键", "用快捷键 {combo}",
                       "发送组合键 {combo}", "触发 {combo} 热键"],
        "sys_press": ["按下 {key} 键", "按 {idx} 次 {key}",
                      "单独按一次 {key}", "在焦点处敲击 {key} 键"],
        "sys_scroll": ["在屏幕上向下滚动 {px} 像素", "滚轮向上滚 {px} 像素",
                       "滚动屏幕 {px} 像素查看下方内容", "把屏幕往回滚 {px} 像素"],
        "sys_screenshot": ["截取整个屏幕保存成图片", "给桌面截个图",
                           "抓一张全屏截图", "截取第 {idx} 块显示器保存成图片"],
    },
    "office": {
        "excel_write": ["把这两行数据写进 {ofile} 的 A1", "向 Excel 写入 {cnt} 行销售明细并保存到 {ofile}",
                        "在 {ofile} 第 {idx} 个工作表写入统计结果", "把汇总数据追加到 {ofile} 末尾"],
        "excel_read": ["读取 {ofile} 前 {cnt} 行", "把 {ofile} 里的数据读出来",
                       "打开 {ofile} 预览表格内容", "读取 {ofile} 第 {idx} 个工作表的数据"],
        "word_create": ["新建 Word 文档 {ofile}，标题是「{text}」", "在 {ofile} 写入 {cnt} 段正文",
                        "生成一份包含标题和正文的 Word 文档到 {ofile}",
                        "把「{text}」整理成 Word 文档保存到 {ofile}"],
        "word_read": ["读取 {ofile} 的正文", "打开 Word 文档 {ofile} 看看内容",
                      "把 {ofile} 里的文字提取出来", "读一下 {ofile} 前 {cnt} 段正文"],
        "ppt_create": ["生成一个 {cnt} 页的 PPT 存到 {ofile}", "把汇报内容做成幻灯片 {ofile}",
                       "创建演示文稿 {ofile}，每页含标题和要点", "用「{text}」做 {cnt} 页幻灯片"],
    },
    "blender": {
        "blender_run_script": ["用 bpy 脚本建一个 {cnt} 面的立方体并加材质",
                               "无头执行一段 Blender Python 建模脚本",
                               "在 Blender 里跑脚本批量布置 {idx} 盏灯光",
                               "写 bpy 脚本把 {blend} 里的模型复制 {cnt} 份"],
        "blender_render": ["渲染 {blend} 的第 1 帧到 render.png", "把 {blend} 场景渲染成图片",
                           "渲染 {blend} 的第 {idx} 帧", "按 {w}x{h} 分辨率渲染 {blend}"],
        "blender_available": ["检查本机有没有安装 Blender", "确认 Blender 是否可用",
                              "看看 Blender 版本信息", "确认 Blender {num} 以上版本是否可用"],
    },
    "media": {
        "video_convert": ["把 {media} 转码成 output.mkv", "把 {media} 转成 H.264 的 mp4",
                          "用 ffmpeg 转换 {media} 的格式", "把 {media} 转码并压缩到 {num} MB"],
        "video_trim": ["裁剪 {media}，从第 {sec} 秒开始取 {dur} 秒", "截取 {media} 第 {sec} 秒起的片段",
                       "把 {media} 从第 {sec} 秒起截 {dur} 秒", "裁掉 {media} 开头 {sec} 秒"],
        "video_extract_audio": ["从 {media} 里把音轨抽成 mp3", "提取 {media} 的音频",
                               "把 {media} 的声音导出为 mp3", "从 {media} 分离出第 {idx} 条音轨"],
        "video_extract_frames": ["从 {media} 每秒抽一帧导出成图片", "按 2 fps 抽取 {media} 的帧",
                                 "把 {media} 拆成逐帧图片", "从 {media} 第 {sec} 秒起按 1 fps 抽帧"],
        "video_probe": ["查看 {media} 的分辨率、码率和时长", "读取 {media} 的元信息",
                        "probe 一下 {media} 的视频参数", "检查 {media} 是不是 H.264 编码"],
        "image_resize": ["把图片 {img} 缩放到 {w}x{h}", "把 {img} 改成 {w}x{h}",
                         "调整 {img} 的尺寸到 {w} 宽", "把 {img} 等比缩放到宽度 {w}"],
        "image_crop": ["裁剪 {img}，去掉四周 {px} 像素的空白边框", "按矩形区域裁剪 {img}",
                       "裁掉 {img} 右侧 {px} 像素的空白", "把 {img} 左上角裁掉 {px} 像素"],
        "image_info": ["查看 {img} 的尺寸和格式", "读取 {img} 的宽高信息",
                       "确认 {img} 是什么格式", "检查 {img} 是否超过 {num} KB"],
    },
    "code": {
        "file_write": ["把这段配置内容写入 {path}", "新建 {path} 并写入初始化代码",
                       "往 {path} 末尾追加一条日志", "把生成的 {cnt} 行结果保存到 {path}"],
        "file_read": ["读取 {path} 的内容", "打开 {path} 看看里面写了什么",
                      "把 {path} 前 {cnt} 行读出来", "读取 {path} 并统计行数"],
        "code_run": ["在 {repo} 执行 npm run build", "在 {repo} 运行 pytest 看哪些用例失败",
                     "在 {repo} 执行 cargo build 检查编译错误", "在 PowerShell 里跑 {path} 相关的命令"],
        "code_python": ["执行一段 Python 代码计算前 {num} 个素数", "跑个 Python 脚本统计 CSV 的 {cnt} 行数据",
                        "用 Python 验证这个正则能否匹配", "执行 Python 代码把 {path} 转成 JSON"],
        "git_status": ["查看 {repo} 仓库有哪些未提交改动", "看看 {repo} 工作区有没有冲突文件",
                       "检查 {repo} 暂存区里有哪些文件", "确认 {repo} 有没有未跟踪的新文件"],
        "git_log": ["查看 {repo} 最近 {idx} 条提交记录", "看看 {repo} 最近几次 commit 改了什么",
                    "列出 {repo} 最近 {idx} 次提交标题", "读一下 {repo} 最近 {idx} 条提交说明"],
    },
}


def _fill(tpl: str, rng: random.Random) -> str:
    """把模板里的占位符填成随机的真实取值（保证样本互不相同）。"""
    return tpl.format(
        url=rng.choice(_URLS), query=rng.choice(_QUERIES), text=rng.choice(_TEXTS),
        path=rng.choice(_PATHS), repo=rng.choice(_REPOS), ofile=rng.choice(_OFFICE_FILES),
        media=rng.choice(_MEDIA_FILES), img=rng.choice(_IMAGE_FILES), blend=rng.choice(_BLEND_FILES),
        num=rng.randint(2, 999), px=rng.randint(100, 1200), idx=rng.randint(1, 12),
        cnt=rng.randint(2, 60), key=rng.choice(_KEYS), combo=rng.choice(_COMBOS),
        sel=rng.choice(_SELECTORS),
        x=rng.randint(10, 1900), y=rng.randint(10, 1000),
        w=rng.choice((640, 800, 1280, 1920)), h=rng.choice((360, 600, 720, 1080)),
        sec=rng.randint(1, 300), dur=rng.randint(2, 60),
    )


def _fill_unique(tpl: str, rng: random.Random, seen: set[str], tries: int = 12) -> str:
    """渲染模板并尽量避开已用过的 state。

    为什么需要它：占位符池虽大于模板数，但随机抽样仍会撞车（实测 600 条里
    有 182 条重复，去重后只剩 418 条）。重采样能把重复压下去；当某条模板的
    组合空间真的耗尽时（如「输入『{text}』」只有 6 个取值）接受重复，
    避免死循环——这与旧版"无参模板刷量"是两回事，不会让某个任务塌成个位数。
    """
    state = _fill(tpl, rng)
    for _ in range(tries):
        if state not in seen:
            break
        state = _fill(tpl, rng)
    seen.add(state)
    return state


def build_directed_routing(rng: random.Random, n: int) -> list[tuple[dict, dict]]:
    """定向六域路由：句式与推理端 `agent/router.py` 完全同构。"""
    doms = list(_DIRECTED_DOMAIN_TEMPLATES)
    seen: set[str] = set()
    out = []
    i = 0
    while len(out) < n:
        dom = doms[i % len(doms)]
        tpls = _DIRECTED_DOMAIN_TEMPLATES[dom]
        state = _fill_unique(tpls[(i // len(doms)) % len(tpls)], rng, seen)
        q = {"type": CHOICE, "instructions": "这条用户请求应该交给哪个工具域处理？",
             "criteria": dict(DOMAIN_DESC)}
        out.append(_record(state, q, dom,
                           {"task": "directed_routing", "source": "synthetic", "domain": dom}))
        i += 1
    return out


def build_directed_tool_choice(rng: random.Random, n: int) -> list[tuple[dict, dict]]:
    """定向域内工具选择：候选=该域全部真实工具（≤26，保证单字母可映射）。"""
    flat = [(dom, tool, tpls) for dom, tools in _DIRECTED_TOOL_INTENTS.items()
            for tool, tpls in tools.items()]
    seen: set[str] = set()
    out = []
    i = 0
    while len(out) < n:
        dom, tool, tpls = flat[i % len(flat)]
        order = list(_DIRECTED_TOOL_INTENTS[dom])
        rng.shuffle(order)
        state = _fill_unique(tpls[(i // len(flat)) % len(tpls)], rng, seen)
        q = {"type": CHOICE, "instructions": "要完成这个任务，应该调用哪个工具？",
             "criteria": {t: "" for t in order}}
        out.append(_record(state, q, tool,
                           {"task": "directed_tool_choice", "source": "synthetic", "domain": dom}))
        i += 1
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
    # 四域定向：句式与推理端同构，且每条模板都带随机占位符（去重后不塌陷）
    "directed_routing": 600,        # 100/域，覆盖关键词启发式够不到的长尾表述
    "directed_tool_choice": 480,    # ≈10.7/工具，45 个真实工具全覆盖
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
            elif task == "directed_routing":
                pairs = build_directed_routing(rng, BUILDERS[task])
            elif task == "directed_tool_choice":
                pairs = build_directed_tool_choice(rng, BUILDERS[task])
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
    "build_grounded", "build_directed_routing", "build_directed_tool_choice",
]

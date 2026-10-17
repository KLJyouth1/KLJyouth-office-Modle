"""普通对话（非智能体）数据生成 —— 抗幻觉的"裸聊"样本。

真实问题（本机复现）：0.5B 决策模型被 99% 的"决策轨迹"样本训练后，用户在
**没有工具清单的裸对话**里提问（例如 `ollama run` 直接聊）时，模型仍按决策格式
输出 JSON，并**编造不存在的工具名**（本机实测吐出 `parse_expr` / `parse_date`）。
根因是训练/推理分布不一致——裸对话样本太少（仅 59/4104）。

本模块专门生成"裸对话"样本：不带决策脚手架、只有 user→assistant 两条消息，
与 ChatML 模板在裸聊时的实际输入形态一致（模板仅在 .Messages 为空时才注入
Modelfile 的 .System，裸聊时并不会带上智能体系统提示）。

两条铁律：
  A. 能确定的（寒暄、常识、能力介绍）→ 用自然语言正常回答，**不许一律弃答**；
  B. 需要精确计算/实时数据/私有数据/文件、而当前又没有工具可用 → 如实说明并
     索要来源或工具，**绝不编造数值，也绝不编造工具名**。

零重依赖：只依赖 config，不 import tools/perception，可在精简训练环境运行。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import DEMO_DIR
from ..prompts import CHAT_SYSTEM

# ------------------------------------------------------------------ 固定语料
_CHITCHAT = [
    ("你好", "你好，我是 OmniPilot，一个跑在 Windows 上的桌面智能体。我可以调用工具帮你操作浏览器、Office 文档、执行代码、处理图片视频、做 3D 建模和系统操作。需要我做什么？"),
    ("在吗", "我在。说说你要做什么，电脑上的活儿大多可以交给我。"),
    ("嗨", "嗨，有什么可以帮你的？"),
    ("早上好", "早上好，今天想让我做点什么？"),
    ("晚上好", "晚上好，需要我帮忙处理什么任务吗？"),
    ("谢谢", "不客气，随时叫我。"),
    ("辛苦了", "不辛苦，还有别的需要就说。"),
    ("再见", "再见，需要时再找我。"),
    ("你是谁", "我是 OmniPilot，一个桌面智能体，通过调用工具一步步完成你在电脑上的任务。"),
    ("你叫什么", "我叫 OmniPilot，一个运行在 Windows 上的桌面自动化智能体。"),
    ("介绍一下你自己", "我是 OmniPilot，桌面智能体。能力覆盖：浏览器操作、Office 文档、代码执行、图片/视频处理、Blender 3D 建模、系统键鼠与截图。需要精确结果时我会实际调用工具，而不是凭记忆猜。"),
    ("你能做什么", "我能帮你操控浏览器、读写 Excel/Word/PPT、跑 Python 与命令、转码/裁剪图片视频、用 Blender 建模，以及模拟键鼠、截图等系统操作。"),
    ("你有什么功能", "主要功能：网页自动化、办公文档处理、代码执行、媒体处理、3D 建模和系统操作。你把任务描述清楚，我来执行。"),
    ("你会干什么", "我会调用工具完成电脑上的具体任务，比如打开网页、整理表格、跑脚本、剪辑视频、建模渲染。"),
    ("你支持中文吗", "支持，我可以用中文和你交流并理解中文任务。"),
    ("你好呀", "你好呀，需要我帮你做什么？"),
    ("嗨嗨", "在的，说吧。"),
]

# 必须正常回答的常识/简单问题（防止模型被"抗幻觉"训练成一律拒答）
_SAFE_QA = [
    ("中国的首都是哪里", "中国的首都是北京。"),
    ("法国首都是哪座城市", "法国首都是巴黎。"),
    ("一年有几个月", "一年有 12 个月。"),
    ("一周有几天", "一周有 7 天。"),
    ("一天有多少小时", "一天有 24 小时。"),
    ("一小时有多少分钟", "一小时有 60 分钟。"),
    ("水的化学式是什么", "水的化学式是 H₂O。"),
    ("1 加 1 等于几", "1 加 1 等于 2。"),
    ("3 乘以 4 等于多少", "3 乘以 4 等于 12。"),
    ("7 加 5 等于几", "7 加 5 等于 12。"),
    ("太阳从哪个方向升起", "太阳从东方升起。"),
    ("Python 是一门什么语言", "Python 是一门通用高级编程语言，常用于数据分析、自动化脚本和后端开发。"),
    ("HTTP 和 HTTPS 有什么区别", "HTTPS 是 HTTP 加上了 TLS 加密，能防止传输被窃听和篡改；HTTP 是明文传输。"),
    ("什么是机器学习", "机器学习是让计算机从数据中学习规律、并对新数据做出预测的方法。"),
    ("圆面积公式是什么", "圆的面积等于 π 乘以半径的平方。"),
    ("光速大约是多少", "真空中的光速约为每秒 299792 公里。"),
    ("JSON 是什么", "JSON 是一种轻量的数据交换格式，用键值对表示结构化数据。"),
    ("请把“你好世界”翻译成英文", "Hello, world."),
    ("怎么用 Python 读取一个文本文件", "可以用 open('文件名', 'r', encoding='utf-8') 读取，建议配合 with 语句以便自动关闭。"),
    ("给我讲个一句话的笑话", "为什么程序员总分不清万圣节和圣诞节？因为 Oct 31 等于 Dec 25。"),
]

# ------------------------------------------------------------------ 生成式语料
_ARITH_REFUSAL = (
    "这个多位乘法我需要在执行环境里用计算工具算，才能保证结果准确；单凭记忆心算容易出错，"
    "我不能给你一个不可靠的数字。请让我调用计算工具，或者把这条计算放进智能体流程里执行。"
)

_PERSONAL_REFUSAL = (
    "我拿不到这项数据——我没有访问你的账单、银行、健康或消费记录的权限，也没有相关输入。"
    "请提供数据来源（文件、截图或接口），我才能给出准确数字；我不会凭猜测编造金额。"
)

_FILE_REFUSAL = (
    "这个文件不存在，或者我当前没有文件访问能力，所以我无法读取它的内容。"
    "请确认路径是否正确、并通过智能体的可执行流程让我读取；在拿到真实文件之前，我不会编造内容。"
)

_UNKNOWABLE_REFUSAL = (
    "我没有这项信息，也无法推断出来——这属于我不可知的范围。"
    "强行给出一个具体答案就是编造。如果你能提供资料，我很乐意帮你分析。"
)

_NEEDS_TOOL_REFUSAL = (
    "这需要实际执行工具才能做到。在智能体执行环境里我可以调用相应的浏览器/办公/代码工具来完成；"
    "但当前只是普通对话，我没有可执行的工具，无法真的操作。请把这条任务交给智能体运行，我就能实际执行。"
)

_PERIODS = ["上个月", "这个月", "上周", "上周五", "去年 12 月", "上季度", "昨天", "今天上午", "本月前十天"]
_PERSONAL_ITEMS = [
    "具体的报销金额", "花了多少钱", "银行卡余额", "微信零钱还剩多少", "支付宝里的余额",
    "社保账户里有多少钱", "工资是多少", "步数统计", "体检结果", "流量用了多少", "话费余额", "公积金余额",
]
_FILE_NAMES = ["zzz_nofile.xlsx", "财务报表2029.xlsx", "not_exist.csv", "我的离职证明.docx",
               "年度预算.xlsx", "客户名单.xlsx", "合同扫描件.pdf", "会议纪要.docx", "库存表.xlsx",
               "发票明细.csv", "月度报表.xlsx", "工资条.xlsx"]
_SITES = ["example.com", "baidu.com", "github.com", "zhihu.com", "bilibili.com", "jd.com"]
_MEDIA = ["clip.mp4", "demo.mov", "cover.png", "input.jpg", "screen_record.mkv"]
_DOCS = ["周报.docx", "方案.pptx", "数据.xlsx"]
_UNKNOWABLE_Q = [
    "我们公司明年的营收目标是多少", "我邻居叫什么名字", "我电脑的序列号是多少",
    "我上个月的通话记录", "我家里 WiFi 的密码", "我去年体检的血糖值",
    "我们部门下季度的预算", "我同事的手机号", "我账户的登录验证码",
    "我昨天访问过哪些网站", "我这台电脑的开机密码", "我银行卡的到期日",
]


def _chitchat_pairs(n: int = 60) -> list[tuple[str, str]]:
    return list(_CHITCHAT)


def _safe_pairs(n: int = 60) -> list[tuple[str, str]]:
    return list(_SAFE_QA)


def _arith_pairs(n: int = 60) -> list[tuple[str, str]]:
    out = []
    for _ in range(n):
        a = random.randint(1000, 99999)
        b = random.randint(11, 99)
        q = random.choice([
            f"{a}×{b} 等于多少，不要调用工具",
            f"直接告诉我 {a} 乘以 {b} 等于几",
            f"不用工具，帮我算一下 {a}*{b}",
            f"别调工具，{a}×{b}=?",
            f"心算一下 {a}×{b} 的结果",
        ])
        out.append((q, _ARITH_REFUSAL))
    return out


def _personal_pairs(n: int = 60) -> list[tuple[str, str]]:
    out = []
    for _ in range(n):
        p = random.choice(_PERIODS)
        item = random.choice(_PERSONAL_ITEMS)
        q = random.choice([
            f"查询我{p}的{item}", f"我{p}的{item}是多少", f"帮我看看我的{item}",
            f"我的{item}有多少", f"{p}我的{item}",
        ])
        out.append((q, _PERSONAL_REFUSAL))
    return out


def _file_pairs(n: int = 60) -> list[tuple[str, str]]:
    out = []
    for _ in range(n):
        name = random.choice(_FILE_NAMES)
        if random.random() < 0.5:
            name = f"{random.randint(100, 999)}_{name}"
        q = random.choice([
            f"读取文件 {name} 的内容", f"打开 {name} 给我看看", f"{name} 里写了什么",
            f"把 {name} 的内容摘要一下", f"帮我看看 {name}",
        ])
        out.append((q, _FILE_REFUSAL))
    return out


def _unknowable_pairs(n: int = 60) -> list[tuple[str, str]]:
    out = []
    for _ in range(n):
        q = random.choice(_UNKNOWABLE_Q)
        out.append((q, _UNKNOWABLE_REFUSAL))
    return out


def _needs_tool_pairs(n: int = 60) -> list[tuple[str, str]]:
    out = []
    targets = [f"打开 {s} 看看首页" for s in _SITES]
    targets += [f"读取 {d} 的内容" for d in _DOCS]
    targets += [f"把 {m} 转成 mp4" for m in _MEDIA]
    targets += ["把 12345 乘以 67 并把结果告诉我", "截一张当前屏幕的图",
                "在当前目录列出所有文件", "把结果写入表格 报表.xlsx"]
    for _ in range(n):
        q = random.choice(targets)
        out.append((q, _NEEDS_TOOL_REFUSAL))
    return out


_GENERATORS = [
    _chitchat_pairs,
    _safe_pairs,
    _arith_pairs,
    _personal_pairs,
    _file_pairs,
    _unknowable_pairs,
    _needs_tool_pairs,
]


def generate_chat(target: int = 1500, out_path: Path | None = None, seed: int = 13) -> dict:
    """生成"裸对话"样本（system(CHAT_SYSTEM) + user + assistant），返回统计。

    带 CHAT_SYSTEM 是关键：推理时 `ollama run` 会把 Modelfile 的 SYSTEM 注入，
    训练样本必须与之一致，模型才能学会"看到 CHAT_SYSTEM 就用自然语言对话"。
    """
    random.seed(seed)
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(q: str, a: str) -> bool:
        key = (q.strip(), a.strip())
        if not key[0] or key in seen:
            return False
        seen.add(key)
        pairs.append(key)
        return True

    rounds = 0
    while len(pairs) < target and rounds < 200:
        rounds += 1
        progressed = False
        for g in _GENERATORS:
            for q, a in g(60):
                if add(q, a):
                    progressed = True
                if len(pairs) >= target:
                    break
            if len(pairs) >= target:
                break
        if not progressed:
            break

    rows = [{"messages": [{"role": "system", "content": CHAT_SYSTEM},
                          {"role": "user", "content": q},
                          {"role": "assistant", "content": a}],
             "meta": {"source": "chat", "tool": "(chat)"}} for q, a in pairs]
    out = Path(out_path) if out_path else (DEMO_DIR / "chat.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"ok": True, "count": len(rows), "path": str(out)}


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    print(json.dumps(generate_chat(n), ensure_ascii=False))

"""定向训练 · 合成专家轨迹生成。

需求："整个模型全程你自己喂数据"。本模块自动生成覆盖各领域的专家演示数据：
每条样本 = 与推理时完全一致的提示(系统提示 + 任务 + 计划 + 工具清单 + 环境观察 + 历史)
        → 正确的 JSON 动作。
直接复用推理端的 SYSTEM_PROMPT / _DECIDE_PROMPT 与真实工具清单，保证训练/推理同分布。
参数充分随机化，保证样本多样(去重后仍能保留绝大多数)。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..agent.loop import SYSTEM_PROMPT, _DECIDE_PROMPT
from ..config import DEMO_DIR
from ..prompts import CHAT_SYSTEM
from ..tools import build_registry

_REG = build_registry()

# ------------------------------------------------------------------ 语料池
_SITES = {
    "baidu.com": "百度一下，你就知道",
    "example.com": "Example Domain",
    "github.com": "GitHub",
    "zhihu.com": "知乎 - 有问题，就会有答案",
    "bilibili.com": "哔哩哔哩",
    "taobao.com": "淘宝网",
    "jd.com": "京东",
    "python.org": "Welcome to Python.org",
    "wikipedia.org": "Wikipedia",
}
_SEARCH_SITES = ["baidu.com", "google.com", "bing.com"]
_QUERIES = ["YOLO 目标检测", "Python 教程", "Windows 自动化", "多模态大模型",
            "Blender 建模入门", "ffmpeg 转码", "Ollama 使用", "LoRA 微调", "量化推理"]
_LINKS = ["Learn more", "文档", "下载", "登录", "注册", "More information", "帮助中心"]
_TOPICS = ["季度销售报告", "项目周报", "产品需求说明", "技术选型方案", "员工培训计划",
           "市场调研结论", "系统设计方案", "年度总结"]
_TEXTS = ["hello omnipilot", "训练完成", "测试通过", "你好世界", "模型已就绪",
          "模型训练完成", "自动化测试开始"]
_IMAGES = ["photo.png", "input.jpg", "frame_001.png", "shot.png", "cover.webp", "scan.bmp"]
_VIDEOS = ["clip.mp4", "demo.mov", "trip.mp4", "screen_record.mkv"]
_OBJECTS = ["cube", "sphere", "cylinder", "cone", "plane", "torus"]


def _uid() -> int:
    return random.randint(1000, 9999)


def _fmt_hist(n: int, tool: str, args: dict, out: str) -> str:
    return f"#{n} {tool}({json.dumps(args, ensure_ascii=False)[:120]}) -> 成功: {out[:160]}"


def _render_action(action: dict) -> str:
    """把动作渲染成**标准 Qwen 工具调用协议**的助手输出。

    协议（与官方 qwen2.5 模板一致，Ollama 会据此解析出 tool_calls）：
      - 调工具：<tool_call>{"name":…,"arguments":{…}}</tool_call>（块前可有一句理由）
      - 完成：直接自然语言答复

    为什么不再用自研的 {"tool":…,"args":…}：Ollama 只认模板声明的协议；自研格式
    永远拿不到 tool_calls（实测会报 "does not support tools" 或 tool_calls=null）。
    """
    if "final" in action:
        return str(action["final"])
    name = action.get("tool", "")
    args = action.get("args", {}) or {}
    body = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
    thought = str(action.get("thought") or "").strip()
    prefix = f"{thought}\n" if thought else ""
    return f"{prefix}<tool_call>\n{body}\n</tool_call>"


class _Builder:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def emit(self, task: str, plan: str, domains: tuple[str, ...],
             steps: list[tuple[str, dict, str]]) -> None:
        tools = _REG.describe(domains)
        history: list[str] = []
        for obs, action, out in steps:
            prompt = _DECIDE_PROMPT.format(
                task=task, plan=plan, tools=tools, observation=obs,
                history="\n".join(history) or "（无）",
            )
            self.rows.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": _render_action(action)},
                ],
                "meta": {"source": "synth", "domains": list(domains),
                         "tool": action.get("tool", "(final)")},
            })
            if "final" not in action:
                history.append(_fmt_hist(len(history) + 1, action["tool"], action.get("args", {}), out))

    # ------------------------------------------------------------ browser
    def add_browser(self) -> None:
        d = ("browser",)
        site = random.choice(list(_SITES))
        title = _SITES[site]
        self.emit(
            f"打开 {site} 并告诉我页面的标题",
            f"1. 打开 {site}\n2. 读取页面标题并汇报",
            d,
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "先用 browser_goto 打开目标网址", "tool": "browser_goto",
                  "args": {"url": f"https://{site}"}},
                 f"已打开 https://{site} —— 标题: {title}"),
                (f"【浏览器状态】\nURL: https://{site}/\n标题: {title}",
                 {"thought": "已拿到标题，任务完成", "final": f"页面标题是「{title}」"}, ""),
            ],
        )
        site2 = random.choice(_SEARCH_SITES)
        q = random.choice(_QUERIES)
        eng = {"baidu.com": "wd", "google.com": "q", "bing.com": "q"}[site2]
        self.emit(
            f"在 {site2} 搜索“{q}”",
            f"1. 打开 {site2}\n2. 输入关键词\n3. 回车\n4. 汇报结果页地址",
            d,
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "先打开搜索引擎首页", "tool": "browser_goto",
                  "args": {"url": f"https://{site2}"}},
                 f"已打开 https://{site2} —— 标题: 搜索"),
                (f'【浏览器状态】\n[0] input "搜索" @(400,{random.randint(150,220)}) size 600x40',
                 {"thought": "在搜索框填入关键词", "tool": "browser_fill",
                  "args": {"selector": "input[name=%s]" % eng, "text": q}},
                 "已在输入框填入关键词"),
                ("【浏览器状态】\n[0] input \"搜索\" 已填写",
                 {"thought": "回车提交搜索", "tool": "browser_press", "args": {"key": "Enter"}},
                 "已按键 Enter"),
                (f"【浏览器状态】\nURL: https://{site2}/s?{eng}={q}",
                 {"thought": "已拿到结果页地址",
                  "final": f"已在 {site2} 完成搜索，结果页：https://{site2}/s?{eng}={q}"}, ""),
            ],
        )
        link = random.choice(_LINKS)
        target = random.choice(["https://www.iana.org/help/example-domains",
                                "https://www.iana.org/domains/reserved"])
        self.emit(
            f"打开 example.com，点击页面里的 {link} 链接，告诉我跳转后的网址",
            f"1. 打开 example.com\n2. 点击 {link}\n3. 读取跳转地址",
            d,
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "先打开页面", "tool": "browser_goto", "args": {"url": "https://example.com"}},
                 "已打开 https://example.com —— 标题: Example Domain"),
                (f'【浏览器状态】\n[0] a "{link}" @({random.randint(500,700)},{random.randint(700,800)}) size 87x20',
                 {"thought": f"点击 {link} 链接", "tool": "browser_click_text", "args": {"text": link}},
                 f"已点击文本「{link}」"),
                (f"【浏览器状态】\nURL: {target}",
                 {"thought": "读取当前地址", "tool": "browser_url", "args": {}}, f"URL: {target}"),
                (f"【浏览器状态】\nURL: {target}",
                 {"thought": "已拿到跳转地址", "final": f"跳转后的网址是 {target}"}, ""),
            ],
        )
        self.emit(
            "把页面往下滚动 800 像素后截图",
            "1. 打开页面\n2. 向下滚动\n3. 截图",
            d,
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "先打开一个页面", "tool": "browser_goto", "args": {"url": "https://example.com"}},
                 "已打开 https://example.com"),
                ("【浏览器状态】\nURL: https://example.com/",
                 {"thought": "向下滚动 800 像素", "tool": "browser_scroll", "args": {"dy": 800}},
                 "已滚动 800 像素"),
                ("【浏览器状态】\nURL: https://example.com/",
                 {"thought": "滚动到位后截图", "tool": "browser_screenshot", "args": {}},
                 "页面截图: runs/browser_shot.png"),
                ("【浏览器状态】\nURL: https://example.com/",
                 {"thought": "完成", "final": "已滚动并截图"}, ""),
            ],
        )

    # ------------------------------------------------------------ office
    def add_office(self) -> None:
        d = ("office",)
        fp = f"报表_{_uid()}.xlsx"
        rows = [["名称", "数量"], [random.choice(["苹果", "香蕉", "橙子"]), random.randint(1, 99)]]
        self.emit(
            f"把表格数据写入 {fp} 的 A1 单元格",
            "1. 用 excel_write 写入\n2. 汇报",
            d,
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "用 excel_write 写入二维数据", "tool": "excel_write",
                  "args": {"path": fp, "rows": rows}},
                 f"已写入 {len(rows)} 行到 {fp}:A1"),
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "写入成功，任务完成", "final": f"已把 {len(rows)} 行数据写入 {fp}"}, ""),
            ],
        )
        rp = f"数据_{_uid()}.xlsx"
        self.emit(
            f"读取 {rp} 前 10 行内容",
            "1. 用 excel_read 读取\n2. 汇报",
            d,
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "调用 excel_read", "tool": "excel_read",
                  "args": {"path": rp, "max_rows": 10}},
                 "读取 2×2"),
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "已读到数据", "final": "已读取表格前若干行"}, ""),
            ],
        )
        topic = random.choice(_TOPICS)
        wp = f"{topic}_{_uid()}.docx"
        self.emit(
            f"生成一份关于“{topic}”的 Word 文档，保存为 {wp}",
            "1. word_create 生成\n2. word_read 校验",
            d,
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "创建 Word 文档", "tool": "word_create",
                  "args": {"path": wp, "title": topic,
                           "paragraphs": [f"{topic}正文第一段", f"补充说明_{_uid()}", "结论与建议"]}},
                 f"已生成 Word 文档: {wp}"),
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "读回校验", "tool": "word_read", "args": {"path": wp}},
                 f"{topic}\r正文第一段\r结论与建议\r\r"),
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "校验通过", "final": f"已生成并校验文档 {wp}"}, ""),
            ],
        )
        n = random.choice([2, 3, 4, 5])
        pp = f"演示_{_uid()}.pptx"
        self.emit(
            f"做一份 {n} 页的 PPT 保存为 {pp}",
            "1. ppt_create 生成多页\n2. 汇报",
            d,
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "创建多页 PPT", "tool": "ppt_create",
                  "args": {"path": pp, "slides": [{"title": f"第{i}页", "body": "要点"} for i in range(1, n + 1)]}},
                 f"已生成 PPT({n}页): {pp}"),
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "完成", "final": f"已生成 {n} 页的 PPT {pp}"}, ""),
            ],
        )

    # ------------------------------------------------------------ code
    def add_code(self) -> None:
        d = ("code",)
        t = random.choice(_TEXTS)
        self.emit(
            f"用 python 打印一行 {t}",
            "1. 用 code_python 执行\n2. 汇报输出",
            d,
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "直接执行 Python 片段，无需建文件", "tool": "code_python",
                  "args": {"code": f'print("{t}")', "timeout": random.choice([5, 10, 30])}},
                 f"exit=0\n{t}"),
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "输出符合预期", "final": t}, ""),
            ],
        )
        self.emit(
            "查看当前目录下有哪些文件",
            "1. 用 code_run 执行 dir\n2. 汇报",
            d,
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "用 PowerShell 列目录", "tool": "code_run",
                  "args": {"command": "Get-ChildItem -Name"}},
                 "exit=0\ndetect.py\ntrain.py"),
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "已列出", "final": "当前目录包含 detect.py、train.py 等文件"}, ""),
            ],
        )
        n = random.choice([5, 10, 20, 50])
        self.emit(
            f"查看项目最近的 {n} 条 git 提交",
            "1. git_log 查询\n2. 汇报",
            d,
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "调用 git_log", "tool": "git_log", "args": {"n": n}},
                 f"exit=0\n{_uid()}abc 更新"),
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "已获取记录", "final": f"最近 {n} 条提交已列出"}, ""),
            ],
        )

    # ------------------------------------------------------------ media
    def add_media(self) -> None:
        d = ("media",)
        img = random.choice(_IMAGES)
        w, h = random.choice([(320, 240), (640, 480), (192, 108), (800, 600)])
        self.emit(
            f"把图片 {img} 缩放到 {w}x{h}",
            "1. image_resize\n2. 汇报",
            d,
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "调用 image_resize", "tool": "image_resize",
                  "args": {"input": img, "output": f"out_{w}x{h}.png", "width": w, "height": h}},
                 f"已缩放到 {w}x{h}: out_{w}x{h}.png"),
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "完成", "final": f"已把 {img} 缩放到 {w}x{h}"}, ""),
            ],
        )
        img2 = random.choice(_IMAGES)
        self.emit(
            f"查看图片 {img2} 的尺寸和格式",
            "1. image_info\n2. 汇报",
            d,
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "调用 image_info", "tool": "image_info", "args": {"input": img2}},
                 f"{random.choice([1920, 1280, 800])}x{random.choice([1080, 720, 600])} RGB PNG"),
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "已获取信息", "final": f"{img2} 的尺寸与格式已读取"}, ""),
            ],
        )
        vid = random.choice(_VIDEOS)
        dur = random.choice([3, 5, 10, 15])
        self.emit(
            f"把视频 {vid} 裁剪前 {dur} 秒保存为 clip_cut.mp4",
            "1. video_trim\n2. 汇报",
            d,
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "调用 video_trim", "tool": "video_trim",
                  "args": {"input": vid, "output": "clip_cut.mp4", "start": 0, "duration": dur}},
                 "ffmpeg exit=0"),
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "完成", "final": f"已裁剪出 {dur} 秒的 clip_cut.mp4"}, ""),
            ],
        )
        vid2 = random.choice(_VIDEOS)
        self.emit(
            f"把视频 {vid2} 转成 mp4 格式",
            "1. video_convert\n2. 汇报",
            d,
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "调用 video_convert", "tool": "video_convert",
                  "args": {"input": vid2, "output": "converted.mp4"}},
                 "ffmpeg exit=0"),
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "完成", "final": f"已把 {vid2} 转成 converted.mp4"}, ""),
            ],
        )

    # ------------------------------------------------------------ system
    def add_system(self) -> None:
        d = ("system",)
        txt = random.choice(_TEXTS)
        cx, cy = random.randint(600, 1200), random.randint(400, 700)
        self.emit(
            f"在记事本里输入 {txt}",
            "1. 从 UI 元素定位编辑区\n2. 点击激活\n3. 输入文本",
            d,
            [
                (f"【桌面UI元素(含坐标)】\n[0] Document \"RichEdit Control\" @({cx},{cy}) in \"无标题 - 记事本\"",
                 {"thought": "定位到 RichEdit 编辑区，点击其中心激活", "tool": "sys_click",
                  "args": {"x": cx, "y": cy}},
                 f"已在 ({cx},{cy}) 点击 1 次"),
                (f"【桌面UI元素(含坐标)】\n[0] Document \"RichEdit Control\" @({cx},{cy}) in \"无标题 - 记事本\"",
                 {"thought": "控件已聚焦，输入文本", "tool": "sys_type", "args": {"text": txt}},
                 f"已输入 {len(txt)} 个字符"),
                ("【桌面UI元素(含坐标)】\n窗口标题已变为 *无标题 - 记事本",
                 {"thought": "输入已生效", "final": f"已在记事本输入「{txt}」"}, ""),
            ],
        )
        label = random.choice(["开始", "文件", "保存", "确定", "取消"])
        lx, ly = random.randint(20, 300), random.randint(1000, 1070)
        self.emit(
            f"点击屏幕上标题为“{label}”的按钮",
            f"1. 在 UI 树中定位“{label}”\n2. 点击其中心",
            d,
            [
                (f"【桌面UI元素(含坐标)】\n[3] Button \"{label}\" @({lx},{ly}) in \"任务栏\"",
                 {"thought": f"定位到按钮“{label}”，点击中心", "tool": "sys_click",
                  "args": {"x": lx, "y": ly}},
                 f"已在 ({lx},{ly}) 点击 1 次"),
                (f"【桌面UI元素(含坐标)】\n[3] Button \"{label}\" @({lx},{ly})",
                 {"thought": "已点击", "final": f"已点击“{label}”按钮"}, ""),
            ],
        )
        self.emit(
            "把当前屏幕截图保存下来",
            "1. sys_screenshot\n2. 汇报路径",
            d,
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "截图", "tool": "sys_screenshot", "args": {"monitor": 1}},
                 "截图已保存: runs/last_shot.png"),
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "完成", "final": "屏幕截图已保存"}, ""),
            ],
        )
        combo = random.choice([["ctrl", "s"], ["ctrl", "c"], ["alt", "tab"], ["ctrl", "shift", "s"]])
        self.emit(
            f"按下组合键 {'+'.join(combo)}",
            "1. sys_hotkey\n2. 汇报",
            d,
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "发送组合键", "tool": "sys_hotkey", "args": {"keys": combo}},
                 f"已按下组合键 {'+'.join(combo)}"),
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "完成", "final": f"已按下 {'+'.join(combo)}"}, ""),
            ],
        )

    # ------------------------------------------------------------ blender
    def add_blender(self) -> None:
        d = ("blender",)
        obj = random.choice(_OBJECTS)
        blend = f"{obj}.blend"
        self.emit(
            f"用 Blender 无头模式建一个 {obj} 并保存为 {blend}",
            "1. blender_run_script 执行 bpy 脚本\n2. 汇报",
            d,
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": f"用 bpy 创建 {obj} 并保存", "tool": "blender_run_script",
                  "args": {"script": f"import bpy\nbpy.ops.mesh.primitive_{obj}_add()\n"
                                     f"bpy.ops.wm.save_as_mainfile(filepath='{blend}')"}},
                 "blender exit=0"),
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "完成", "final": f"已生成 {blend}"}, ""),
            ],
        )
        self.emit(
            "渲染场景 scene.blend 的第 1 帧到 render.png",
            "1. blender_render\n2. 汇报",
            d,
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "调用 blender_render", "tool": "blender_render",
                  "args": {"blend_file": "scene.blend", "output": "render.png", "frame": 1}},
                 "渲染 exit=0"),
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "完成", "final": "已渲染 render.png"}, ""),
            ],
        )


    # ------------------------------------------------------------ 验证与诚实
    def add_verify(self) -> None:
        """教模型两件它原本不会的事：
        A. 先读工具结果、把真实数值写进 final（而不是复述 "exit=0"）；
        B. 上一步失败时，要么修正重试，要么如实说明——绝不谎报成功。
        """
        # A. 计算类：final 必须是真实数值
        a, b = random.randint(100, 9999), random.randint(10, 99)
        prod = a * b
        self.emit(
            f"用 python 计算 {a}*{b}，并把结果告诉我",
            f"1. 用 code_python 计算\n2. 从输出里取出结果并汇报",
            ("code",),
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "用 code_python 计算，表达式会自动打印结果",
                  "tool": "code_python", "args": {"code": f"{a}*{b}"}},
                 f"exit=0\n{prod}"),
                ("（参考上述执行结果）",
                 {"thought": f"工具输出里的结果是 {prod}",
                  "final": f"{a}×{b} = {prod}"},
                 ""),
            ],
        )
        # B. 失败 → 修正后重试（不得谎报成功）
        bad = f"missing_{_uid()}.xlsx"
        good = f"报表_{_uid()}.xlsx"
        rows = [["名称", "数量"], [random.choice(["苹果", "香蕉"]), random.randint(1, 99)]]
        self.emit(
            f"把表格数据写入文件 {good} 的 A1 单元格；若写入失败，请修正后重试",
            "1. excel_write 写入\n2. 失败则修正路径重试\n3. 如实汇报",
            ("office",),
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "先按给定路径尝试写入", "tool": "excel_write",
                  "args": {"path": bad, "rows": rows}},
                 f'com_error: 无法找到 "{bad}"。请检查文件名的拼写'),
                (f"（上一步写入失败：路径 {bad} 不存在）",
                 {"thought": "路径有误，改用正确文件名重试", "tool": "excel_write",
                  "args": {"path": good, "rows": rows}},
                 f"已写入 {len(rows)} 行到 {good}:A1"),
                ("（参考上述执行结果）",
                 {"thought": "这次确实成功了，如实汇报", "final": f"已把 {len(rows)} 行数据写入 {good}"},
                 ""),
            ],
        )
        # C. 读结果 → 抽取并作答
        res = random.choice(["1920x1080 RGB PNG", "1280x720 RGB JPEG", "800x600 RGB BMP"])
        self.emit(
            "查看图片 cover.png 的尺寸和格式",
            "1. image_info\n2. 汇报结果",
            ("media",),
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "调用 image_info 读取信息", "tool": "image_info",
                  "args": {"input": "cover.png"}},
                 res),
                ("（参考上述执行结果）",
                 {"thought": f"工具给出的结果是 {res}",
                  "final": f"cover.png 的尺寸与格式是 {res}"},
                 ""),
            ],
        )
        # D. 失败 → 如实说明（不谎报）
        self.emit(
            "读取文件 nodata.xlsx 的表格内容",
            "1. excel_read\n2. 若失败就如实说明",
            ("office",),
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "尝试读取该文件", "tool": "excel_read",
                  "args": {"path": "nodata.xlsx"}},
                 'com_error: 无法找到 "nodata.xlsx"。请检查文件位置是否正确'),
                ("（上一步读取失败：文件不存在）",
                 {"thought": "文件确实不存在，应如实告知，不能谎报成功",
                  "final": "读取失败：文件 nodata.xlsx 不存在，请确认路径后重试"},
                 ""),
            ],
        )


    # ------------------------------------------------------------ 多步链（关键）
    def add_multistep(self) -> None:
        """多步交付链：教模型"看到上一步结果后接着做下一步，直到交付物完成"。

        为什么单列：实测 0.5B / 1.5B 都栽在这里——搜到结果就收尾、或原地重搜，
        永远走不到"写文件"那一步。原来的合成数据多是"一两步就 final"，
        模型没学会"按计划推进到交付物"。这里补上带真实中间结果的链路。
        """
        # ---------- 链 A：检索 → 读正文 → 写文件 ----------
        q = random.choice(_QUERIES)
        fp = f"E:/temp/notes_{_uid()}.txt"
        site = random.choice(list(_SITES))
        url = f"https://{site}/{random.choice(['docs','guide','wiki'])}/{_uid()}"
        title = _SITES[site]
        facts = [f"{title} 于 {random.randint(2000, 2026)} 年发布",
                 f"该页面要点 {random.randint(2, 9)} 条", f"相关链接 {random.randint(3, 12)} 个"]
        self.emit(
            f"搜索“{q}”并把三条要点写入文件 {fp}",
            f"1. web_search 检索关键词\n2. web_fetch 读候选页面正文\n3. code_python 写入 {fp}\n4. 汇报文件路径",
            ("browser", "code"),
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "先检索关键词，拿到候选页面", "tool": "web_search", "args": {"query": q}},
                 "\n".join(f"[{i}] {t}\n    https://{s}    {random.randint(1,28)}-{random.randint(1,28)} 摘要…"
                           for i, (t, s) in enumerate([(title, site)] + [(f"{q} 教程", "blog.example")], 1))),
                ("（参考上述执行结果）",
                 {"thought": "已拿到页面链接，读取正文提取要点", "tool": "web_fetch",
                  "args": {"url": url, "limit": 1200}},
                 f"标题: {title}\nURL: {url}\n\n" + "；".join(facts)),
                ("（参考上述执行结果）",
                 {"thought": "要点已就绪，写入文件", "tool": "code_python",
                  "args": {"code": "open(r'%s','w',encoding='utf-8').write(%r)" % (fp, "1. " + "\n2. ".join(facts))}},
                 "exit=0\n"),
                ("（参考上述执行结果）",
                 {"thought": "文件已写入，任务完成", "final": f"已把三条要点写入 {fp}"}, ""),
            ],
        )
        # ---------- 链 B：打开网页 → 读正文 → 写文件 ----------
        fp2 = f"E:/temp/page_{_uid()}.txt"
        self.emit(
            f"打开 {site}，把页面正文前若干字写入 {fp2}",
            f"1. browser_goto 打开\n2. browser_text 取正文\n3. code_python 写入 {fp2}\n4. 汇报",
            ("browser", "code"),
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "先打开目标网页", "tool": "browser_goto", "args": {"url": f"https://{site}"}},
                 f"已打开 https://{site} —— 标题: {title}"),
                ("【浏览器状态】\nURL: https://%s/" % site,
                 {"thought": "页面已就绪，读取正文文本", "tool": "browser_text", "args": {"limit": 800}},
                 f"{title} 正文：本页介绍 {q} 的基础与进阶用法，包含 {random.randint(3, 20)} 个示例。"),
                ("（参考上述执行结果）",
                 {"thought": "正文已取到，落盘保存", "tool": "code_python",
                  "args": {"code": f"open(r'{fp2}','w',encoding='utf-8').write('...')"}},
                 "exit=0\n"),
                ("（参考上述执行结果）",
                 {"thought": "已完成写文件", "final": f"页面正文已保存到 {fp2}"}, ""),
            ],
        )
        # ---------- 链 C：写表 → 读回校验 → 汇报 ----------
        xls = f"E:/temp/tab_{_uid()}.xlsx"
        rows = [["名称", "数量"], [random.choice(["苹果", "香蕉", "橙子"]), random.randint(1, 99)]]
        self.emit(
            f"把表格数据写入 {xls}，然后读回校验",
            f"1. excel_write 写入\n2. excel_read 读回\n3. 汇报",
            ("office",),
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "先写入二维数据", "tool": "excel_write", "args": {"path": xls, "rows": rows}},
                 f"已写入 {len(rows)} 行到 {xls}:A1"),
                ("（参考上述执行结果）",
                 {"thought": "写入成功，读回确认", "tool": "excel_read", "args": {"path": xls, "max_rows": 5}},
                 "读取 2×2\n名称\t数量\n" + "\t".join(str(x) for x in rows[1])),
                ("（参考上述执行结果）",
                 {"thought": "校验通过，汇报", "final": f"已写入并校验 {xls}（2 行）"}, ""),
            ],
        )

        # ---------- 链 D：逐条追加写入（教"把每一条都落盘"） ----------
        # 实测缺陷：模型只把第 1 条要点写进文件就收尾了，交付物内容不完整。
        # 用"一条一个 file_write(append)"的轨迹，教它按条数走完再 final。
        q2 = random.choice(_QUERIES)
        fp3 = f"E:/temp/points_{_uid()}.txt"
        n_pts = random.choice([3, 3, 4])
        # 要点必须**互不相同**：实测模型会把同一条反复写入，所以这里不仅给足条数，
        # 还保证每条主题各异，让"逐条写不同内容"这件事在数据里是确定的。
        themes = random.sample(["精度提升", "速度更快", "结构改进", "训练更稳", "显存更省",
                                "部署更易", "标注更省", "小目标更准", "多任务支持"], n_pts)
        pts = [f"{q2} 要点{i}：{t}" for i, t in enumerate(themes, 1)]
        steps_d = [
            ("（本任务无需屏幕观察，可直接调用工具）",
             {"thought": "先检索拿素材", "tool": "web_search", "args": {"query": q2}},
             "\n".join(f"[{i}] {q2} 资料{i}\n    https://example{i}.org/a{i}" for i in range(1, 3))),
        ]
        for i, p in enumerate(pts):
            steps_d.append((
                "（参考上述执行结果）",
                {"thought": f"写入第 {i + 1}/{n_pts} 条", "tool": "file_write",
                 "args": {"path": fp3, "content": f"{i + 1}. {p}\n", "append": bool(i)}},
                f"已{'追加' if i else '写入'} {len(p) + 4} 字到 {fp3}",
            ))
        steps_d.append(("（参考上述执行结果）",
                        {"thought": f"{n_pts} 条要点都写完了，任务完成",
                         "final": f"已把 {n_pts} 条要点逐条写入 {fp3}"}, ""))
        self.emit(f"搜索“{q2}”并把 {n_pts} 条要点写入文件 {fp3}", 
                  f"1. web_search 找素材\n2. 逐条 file_write 写入（后续用 append=true）\n"
                  f"3. 每条要点必须不同，不可重复\n4. 全部写完再 final",
                  ("browser", "code"), steps_d)

        # ---------- 链 E：任务给死 Windows 绝对路径 → 逐字复制落盘 ----------
        # 为什么单列：实测 1.5B 连续两轮把 `D:\Downloads\yolo-ultralytics\_t\points.txt`
        # 写成 `D:\Downloads\yolo-ultralytics_\t\points.txt`。此前合成数据里只有裸文件名
        # （`报表_1234.xlsx`）或 `E:/temp/xx.txt` 这种前斜杠短路径 —— **反斜杠绝对路径的
        # "逐字复制"是零监督的**，模型只能即兴拼，必然出错。
        # 另补一条纠正样本：让"被闸门打回后改用任务给定路径"也成为一个**可学行为**，
        # 而不是重复同一个错路径（这正是实测里发生的事）。
        win_dir = random.choice([r"D:\proj\_t", r"D:\work\_out", r"E:\data\_tmp", r"C:\tmp\_in"])
        wp = f"{win_dir}\\{random.choice(['points', 'summary', 'notes'])}.txt"
        n_w = random.choice([2, 3, 3])
        w_themes = random.sample(_TOPICS, n_w)
        steps_e = [
            ("（本任务无需屏幕观察，可直接调用工具）",
             {"thought": "任务已给定目标路径，逐字用它写入第一条", "tool": "file_write",
              "args": {"path": wp, "content": f"{w_themes[0]}\n", "append": False}},
             f"已写入 {len(w_themes[0])} 字到 {wp}"),
        ]
        for i, theme in enumerate(w_themes[1:], 2):
            steps_e.append((
                "（参考上述执行结果）",
                {"thought": f"追加第 {i}/{n_w} 条", "tool": "file_write",
                 "args": {"path": wp, "content": f"{theme}\n", "append": True}},
                f"已追加 {len(theme)} 字到 {wp}",
            ))
        steps_e.append((
            "（参考上述执行结果）",
            {"thought": f"{n_w} 条都写完了，任务完成", "final": f"已把 {n_w} 条要点写入 {wp}"},
            ""))
        self.emit(
            f"在 {wp} 写入 {n_w} 条不同的项目要点，每条一行",
            f"1. 逐条 file_write 写入 {wp}（后续 append=true）\n"
            f"2. 路径必须逐字复制任务给的那个，不要改写\n3. 全部写完再 final",
            ("code",), steps_e)

        # ---------- 链 F：路径被闸门打回 → 改用任务给定路径（纠正样本） ----------
        wrong = wp.replace("\\_", "_", 1)      # 形如把 `\_t\` 拼坏的近失路径（同名不同目录）
        self.emit(
            f"在 {wp} 写入 {n_w} 条不同的项目要点，每条一行",
            f"1. 逐条 file_write 写入 {wp}（后续 append=true）\n2. 全部写完再 final",
            ("code",),
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "先写入第一条", "tool": "file_write",
                  "args": {"path": wrong, "content": f"{w_themes[0]}\n", "append": False}},
                 f"已写入 {len(w_themes[0])} 字到 {wrong}"),
                (f"你把结果写到了「{wrong}」，但任务指定的目标是「{wp}」。"
                 f"请下一步用任务给定的那个路径重新写入；路径请逐字复制，"
                 f"不要改写、不要省略中间目录。",
                 {"thought": "改用任务给定的路径重新写入", "tool": "file_write",
                  "args": {"path": wp, "content": f"{w_themes[0]}\n", "append": False}},
                 f"已写入 {len(w_themes[0])} 字到 {wp}"),
                ("（参考上述执行结果）",
                 {"thought": "路径已改正，内容已落盘", "final": f"已把内容写入 {wp}"}, ""),
            ],
        )

    def emit_raw(self, user: str, assistant: str) -> None:
        """直接登记一条样本（不套决策提示模板），用于教"非智能体场景"的行为。

        带 CHAT_SYSTEM：与 `ollama run` 时 Ollama 注入的 Modelfile SYSTEM 保持一致。
        """
        self.rows.append({
            "messages": [
                {"role": "system", "content": CHAT_SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "meta": {"source": "synth", "domains": [], "tool": "(chat)"},
        })

    # ------------------------------------------------------------ 诚实与边界
    def add_guard(self) -> None:
        """教两条关键边界（来自真实失败：
        用户直接问 12345×67，模型凭记忆编出 99240，正确是 827115）：
        A. 提示里有工具清单 → 必须真的调工具，不许凭记忆报数；
        B. 提示里没有工具清单（普通对话）→ 就正常对话，绝不许编造精确数值/事实。
        """
        a, b = random.randint(1000, 9999), random.randint(10, 99)
        self.emit(
            f"计算 {a}×{b} 的结果",
            f"1. 调 code_python 实际计算\n2. 据工具输出回答（不得凭记忆）",
            ("code",),
            [
                ("（本任务无需屏幕观察，可直接调用工具）",
                 {"thought": "必须实际计算，不能猜", "tool": "code_python",
                  "args": {"code": f"{a}*{b}"}},
                 f"exit=0\n{a * b}"),
                ("（参考上述执行结果）",
                 {"thought": "依据工具真实输出作答", "final": f"{a}×{b} = {a * b}"},
                 ""),
            ],
        )
        # B. 没有工具时的诚实边界：明确不编造
        self.emit_raw(
            random.choice([
                "介绍一下你自己", "你叫什么", "你能做什么",
            ]),
            random.choice([
                "我是运行在 Windows 上的桌面智能体，通过调用工具帮你完成任务（浏览器、办公、代码、媒体、3D、系统操作）。需要精确计算或查实时信息时，我会调用工具执行，而不是凭记忆猜测。",
                "我是一个桌面智能体，靠工具执行任务。涉及精确计算、实时数据、文件内容这类问题，我会实际调用工具获取，不会凭记忆编造结果。",
            ]),
        )
        self.emit_raw(
            f"直接告诉我 {a}×{b} 等于多少，不要调用工具",
            "这个数字我需要实际计算才能保证准确，凭记忆给出可能出错（我无法可靠心算多位乘法）。请允许我调用计算工具获取精确结果，或者你把算式放到我的执行流程里运行。",
        )
        # B2. 针对基线实测出的两类真实幻觉，逐条教"不编造"
        #  （1）查询没有数据来源的个人化信息 → 编造过"报销金额为500元"
        self.emit_raw(
            random.choice([
                "查询我上个月的具体报销金额", "我的银行卡余额是多少",
                "我上周的步数统计", "我昨天花了多少钱",
            ]),
            "我无法获得这项数据——我没有访问你的账单/银行/健康记录的权限，也没有相关的输入。"
            "请提供数据来源（文件、截图或接口），我才能给出准确数字；我不会凭猜测编造金额。",
        )
        #  （2）超出现有能力/无工具的精确事实 → 应弃答而非编造
        self.emit_raw(
            random.choice([
                "告诉我 987654×321 等于多少，不要调用工具",
                "直接说出 847291×536 的结果",
                "不查资料告诉我昨天北京的PM2.5具体数值",
            ]),
            "这类问题需要实际计算或实时数据来源才能准确回答。我无法在不确定来源的情况下给出精确数值——"
            "凭记忆或猜测给出数字就是编造。请允许我调用工具获取真实结果。",
        )
        #  （3）不存在的资源 → 如实说明
        self.emit_raw(
            random.choice([
                "读取不存在的文件 zzz_nofile.xlsx 的内容",
                "打开我桌面上的财务报表2029.xlsx",
                "读取文件 not_exist_123.csv 的内容",
            ]),
            "该文件不存在或我无法访问，因此我无法读取它的内容。请确认文件路径是否正确；"
            "在拿到真实文件之前，我不会编造它的内容。",
        )
        #  （4）通用知识里的不确定项 → 表态不确定，不硬答
        self.emit_raw(
            random.choice([
                "我们公司明年的营收目标是多少", "我邻居叫什么名字",
                "我电脑的序列号是多少",
            ]),
            "我没有这项信息，也无法推断出来。这属于我不可知的范围，强行给出一个具体答案就是编造。"
            "如果你能提供资料，我非常乐意帮你分析。",
        )


def generate(target: int = 2000, out_path: Path | None = None, seed: int = 7) -> dict:
    """生成合成样本，返回统计。"""
    random.seed(seed)
    b = _Builder()
    rounds = 0
    while len(b.rows) < target and rounds < 2000:
        rounds += 1
        b.add_browser()
        b.add_office()
        b.add_code()
        b.add_media()
        b.add_system()
        b.add_blender()
        b.add_verify()
        b.add_multistep()
        b.add_guard()
    out = Path(out_path) if out_path else (DEMO_DIR / "synth.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    uniq = {}
    with out.open("w", encoding="utf-8") as f:
        for r in b.rows:
            key = json.dumps(r["messages"], ensure_ascii=False, sort_keys=True)
            if key in uniq:
                continue
            uniq[key] = 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"ok": True, "raw": len(b.rows), "unique": len(uniq), "path": str(out)}


def generate_multistep(target: int = 1600, out_path: Path | None = None, seed: int = 21) -> dict:
    """单独生成"多步交付链"语料。

    为什么不混在 synth.jsonl 里：混着会被浏览器/办公等单步样本稀释（实测只占 3%），
    而多步恰恰是当前最缺、最该强化的能力。独立成文件后体量可单独调。
    """
    random.seed(seed)
    b = _Builder()
    rounds = 0
    while len(b.rows) < target and rounds < 4000:
        rounds += 1
        b.add_multistep()
    out = Path(out_path) if out_path else (DEMO_DIR / "multistep.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    uniq = {}
    with out.open("w", encoding="utf-8") as f:
        for r in b.rows:
            key = json.dumps(r["messages"], ensure_ascii=False, sort_keys=True)
            if key in uniq:
                continue
            uniq[key] = 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"ok": True, "raw": len(b.rows), "unique": len(uniq), "path": str(out)}


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    print(json.dumps(generate(n), ensure_ascii=False))

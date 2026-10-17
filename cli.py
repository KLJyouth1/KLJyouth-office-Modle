"""OmniPilot 命令行入口。

用法：
    python -m omnipilot.cli doctor                 # 体检：硬件/运行时/工具链
    python -m omnipilot.cli tools [--domains ...]  # 列出可用工具
    python -m omnipilot.cli run "任务" [选项]       # 执行任务
    python -m omnipilot.cli                       # 交互模式
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys

from .config import DOMAINS, OLLAMA_HOST, PLANNER_MODEL, VISION_MODEL, WORKSPACE


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


# ---------------------------------------------------------------------- doctor
def cmd_doctor(_args) -> int:
    import psutil

    print("=" * 56)
    print("OmniPilot 环境体检")
    print("=" * 56)

    print(f"[CPU] {psutil.cpu_count(logical=False)} 物理核 / {psutil.cpu_count()} 逻辑核")
    vm = psutil.virtual_memory()
    print(f"[内存] 总 {vm.total/2**30:.1f} GB，可用 {vm.available/2**30:.1f} GB")

    gpu = _gpu_info()
    for line in gpu:
        print(f"[GPU] {line}")

    for part in psutil.disk_partitions(all=False):
        try:
            u = psutil.disk_usage(part.mountpoint)
            print(f"[磁盘] {part.device} 可用 {u.free/2**30:.1f} GB / {u.total/2**30:.1f} GB")
        except Exception:
            continue

    print("-" * 56)
    from .runtime.llm import OllamaClient

    client = OllamaClient()
    print(f"[Ollama] {OLLAMA_HOST} -> {'在线' if client.ping() else '离线(请先启动 Ollama)'}")
    if client.ping():
        for m in client.list_models():
            print(f"         - {m['name']}")
        print(f"[规划模型] {PLANNER_MODEL}: {'已安装' if client.has_model(PLANNER_MODEL) else '缺失'}")
        print(f"[视觉模型] {VISION_MODEL}: {'已安装' if client.has_model(VISION_MODEL) else '缺失'}")

    print("-" * 56)
    checks = {
        "playwright": _has_module("playwright"),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "office(pywin32)": _has_module("win32com"),
        "ui_tree(pywinauto)": _has_module("pywinauto"),
        "screen(mss)": _has_module("mss"),
    }
    from .tools.blender import find_blender

    checks["blender"] = find_blender() is not None
    for k, v in checks.items():
        print(f"[工具链] {k}: {'可用' if v else '缺失'}")
    print("=" * 56)
    return 0


def _gpu_info() -> list[str]:
    out: list[str] = []
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        if r.returncode == 0 and r.stdout.strip():
            for line in r.stdout.strip().splitlines():
                out.append("NVIDIA " + line.strip())
    except Exception:
        pass
    return out or ["未检测到 NVIDIA GPU / nvidia-smi 不可用"]


def _has_module(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


# ---------------------------------------------------------------------- tools
def cmd_tools(args) -> int:
    from .tools import build_registry

    domains = tuple(d for d in args.domains.split(",") if d) if args.domains else None
    reg = build_registry(domains=domains)
    print(f"共 {len(reg.all())} 个工具：")
    for t in reg.all():
        print(f"  [{t.domain:8}] {t.name}: {t.description}")
    return 0


# ---------------------------------------------------------------------- data / train
def _default_sources(args, collector) -> list[str]:
    """默认训练语料 = 合成轨迹 + 开源真实数据 + 自演采集（存在即用）。"""
    from pathlib import Path

    from .config import DEMO_DIR

    if args.sources:
        return [s for s in args.sources.split(",") if s]
    cands = [DEMO_DIR / "synth.jsonl", DEMO_DIR / "multistep.jsonl", DEMO_DIR / "open_toolace.jsonl",
             DEMO_DIR / "chat.jsonl", DEMO_DIR / "distill.jsonl", collector.path]
    return [str(p) for p in cands if Path(p).exists()]


def _ensure_chat_corpus() -> None:
    """确保"裸聊"诚实语料存在（抗幻觉：普通对话里不许编造工具名/数值）。"""
    from .config import DEMO_DIR
    from .train.chat_data import generate_chat

    out = DEMO_DIR / "chat.jsonl"
    r = generate_chat(out_path=out)
    print(f"[对话] 裸聊诚实样本: {r['count']} 条 -> {out.name}")


def cmd_data(args) -> int:
    from pathlib import Path

    from .train import SFTCollector, build_dataset

    _ensure_chat_corpus()
    collector = SFTCollector()
    srcs = _default_sources(args, collector)
    print(f"[采集] 自演样本数: {collector.count()}")
    print(f"[语料] 来源: {[Path(s).name for s in srcs]}")
    info = build_dataset(srcs, name=args.name, val_ratio=args.val_ratio)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


def cmd_train(args) -> int:
    from .config import DATASET_DIR
    from .train import SFTCollector, build_dataset, to_ollama, train_lora

    _ensure_chat_corpus()
    collector = SFTCollector()
    srcs = _default_sources(args, collector)
    info = build_dataset(srcs, name=args.name, val_ratio=args.val_ratio)
    print(f"[数据] 总样本={info['total']} 训练={info['train']} 验证={info['val']} 丢弃={info['dropped']}")

    stats = train_lora(
        args.base, DATASET_DIR, name=args.name, epochs=args.epochs, lora_r=args.lora_r,
        max_len=args.max_len, limit=args.limit, use_4bit=args.use_4bit,
        batch_size=args.batch, grad_accum=args.grad_accum,
        gradient_checkpointing=args.grad_ckpt,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.export_ollama and stats.get("ok"):
        res = to_ollama(args.export_ollama, args.ollama_base or args.base, stats["adapter_dir"])
        print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if stats.get("ok") else 1


# ---------------------------------------------------------------------- deploy
def cmd_deploy(args) -> int:
    """合并 LoRA → 转 GGUF → 注册到 Ollama（一键部署）。"""
    from pathlib import Path

    from .prompts import CHAT_SYSTEM
    from .config import ADAPTER_DIR, WORKSPACE
    from .train import build_gguf, merge_lora, register_ollama

    adapter = args.adapter or str(ADAPTER_DIR / "omnipilot_sft")
    merged = args.merged or str(WORKSPACE.parent / "omni_models" / "omnipilot-merged")
    gguf = args.gguf or str(Path(merged).with_suffix("")) + "-f16.gguf"

    print(f"[1/3] 合并 LoRA: {adapter} -> {merged}")
    r1 = merge_lora(args.base, adapter, merged)
    print("     ", r1)
    if not r1.get("ok"):
        return 1

    print(f"[2/3] 转换 GGUF: {gguf}")
    r2 = build_gguf(merged, gguf)
    print("     ", {k: (v[:120] if isinstance(v, str) else v) for k, v in r2.items()})
    if not r2.get("ok"):
        return 1

    print(f"[3/3] 注册 Ollama 模型: {args.name}")
    # 量化档位默认 q4_K_M：f16 的 1.5B（3.1GB）在 4GB 卡上会溢出到内存并拖死请求，
    # 量化后（<1GB）可全部进显存。传 none 可退回不量化。
    quant = None if str(args.quantize).lower() in ("none", "", "null") else args.quantize
    system = None if args.no_system else CHAT_SYSTEM
    r3 = register_ollama(args.name, gguf, system=system,
                         temperature=args.temperature, num_ctx=args.num_ctx,
                         quantize=quant)
    print("     ", r3)
    if r3.get("ok"):
        print(f"\n完成！用 `ollama list` 可见，用 `ollama run {args.name}` 可对话。")
        print(f"让 OmniPilot 用它当大脑：设置环境变量 OMNIPILOT_PLANNER={args.name}")
    return 0 if r3.get("ok") else 1


# ---------------------------------------------------------------------- personal / budget
def cmd_personal(args) -> int:
    from .personal import PersonalProfile

    p = PersonalProfile.load()
    if args.forget:
        removed = p.forget()
        p.save()
        print(f"主动遗忘剪枝：{len(removed)} 条 -> {removed}")
    print(json.dumps(p.stats(), ensure_ascii=False, indent=2))
    s = p.summary()
    print("画像摘要：" + (s.replace("\n", " | ") if s else "（空，运行任务后会自动积累）"))
    return 0


def cmd_budget(args) -> int:
    from .resource import advise, describe_plan

    if args.model:
        print(describe_plan(args.model, quant_bits=args.quant_bits, ctx=args.ctx))
        return 0
    print(advise(args.model_gb or 0.0))
    return 0


def cmd_bench(args) -> int:
    from .resource import bench_suite, format_table

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = bench_suite(models, num_predict=args.n)
    print(format_table(rows))
    return 0


# ---------------------------------------------------------------------- distill / hallucination
def cmd_distill(args) -> int:
    from .train.distill import generate_distill_data

    r = generate_distill_data(teacher=args.teacher, target=args.target)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0 if r.get("ok") else 1


def cmd_halluc(args) -> int:
    """幻觉评测：YOLO 式度量（幻觉率 / 弃答率）。"""
    from .eval import default_cases, evaluate
    from .runtime.llm import OllamaClient

    model = args.model
    client = OllamaClient()

    def ask(case):
        # 直接问模型（不接执行器）——这正是用户"独立对话"时的真实场景
        return client.chat(model, [{"role": "user", "content": case.task}],
                           options={"temperature": 0.0, "num_ctx": 2048, "num_predict": 128},
                           think=False)

    rep = evaluate(default_cases(), ask, verbose=not args.quiet)
    print("-" * 56)
    print(json.dumps(rep.as_dict(), ensure_ascii=False, indent=2))
    print("说明：hallucination_rate 越低越好；abstained 表示主动弃答（诚实）")
    return 0


# ---------------------------------------------------------------------- run
def cmd_run(args) -> int:
    from .agent import Agent
    from .train import SFTCollector

    domains = tuple(d for d in args.domains.split(",") if d) if args.domains else None
    collector = None if args.no_collect else SFTCollector()
    agent = Agent(domains=domains, max_steps=args.max_steps, collector=collector,
                  headless=not args.headful, verbose=True, think=args.think,
                  use_plan=not args.no_plan)
    try:
        out = agent.run(args.task)
    finally:
        agent.close()
    print("-" * 56)
    print(f"成功: {out['ok']} | 步数: {out['steps']} | 领域: {out['domains']}")
    print(f"结果: {out['result']}")
    if collector is not None:
        print(f"[自演采集] 累计样本: {collector.count()} -> {collector.path}")
    return 0 if out["ok"] else 1


def cmd_interactive(args) -> int:
    from .agent import Agent

    print("OmniPilot 交互模式（输入 q 退出）")
    agent = Agent(max_steps=args.max_steps, headless=not args.headful, verbose=True)
    try:
        while True:
            try:
                task = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not task or task.lower() in ("q", "quit", "exit"):
                break
            out = agent.run(task)
            print(f"[结果] {out['result']}")
    finally:
        agent.close()
    return 0


def cmd_web(args) -> int:
    """启动本地聊天控制台：真正驱动智能体执行（搜索/操控电脑）。"""
    from .config import PLANNER_MODEL
    from .webchat import serve

    domains = tuple(d for d in args.domains.split(",") if d) if args.domains else None
    serve(host=args.host, port=args.port, model=args.model or PLANNER_MODEL,
          domains=domains, max_steps=args.max_steps, headless=not args.headful)
    return 0


# ---------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="omnipilot", description="低运行环境的离线复合决策智能体")
    sub = p.add_subparsers(dest="cmd", required=False)

    sub.add_parser("doctor", help="检查硬件与运行时环境")

    t = sub.add_parser("tools", help="列出工具")
    t.add_argument("--domains", default=None, help=f"逗号分隔，可选 {','.join(DOMAINS)}")

    r = sub.add_parser("run", help="执行一个任务")
    r.add_argument("task", help="任务描述")
    r.add_argument("--domains", default=None, help=f"逗号分隔，可选 {','.join(DOMAINS)}")
    r.add_argument("--max-steps", type=int, default=None)
    r.add_argument("--headful", action="store_true", help="浏览器显示界面(默认无头)")
    r.add_argument("--think", action="store_true", help="启用思维链(更准但更慢)")
    r.add_argument("--no-plan", action="store_true", help="跳过显式规划")
    r.add_argument("--no-collect", action="store_true", help="本次运行不采集自演训练样本")

    i = sub.add_parser("interactive", help="交互模式")
    i.add_argument("--max-steps", type=int, default=None)
    i.add_argument("--headful", action="store_true")

    d = sub.add_parser("data", help="查看/构建训练数据集")
    d.add_argument("--sources", default=None, help="逗号分隔的 JSONL 样本来源；默认用自演采集文件")
    d.add_argument("--name", default="omnipilot_sft")
    d.add_argument("--val-ratio", type=float, default=0.1)

    t = sub.add_parser("train", help="LoRA 定向微调(可顺带导出为 Ollama 模型)")
    t.add_argument("--base", required=True, help="基座模型(HF id 或本地路径)")
    t.add_argument("--sources", default=None, help="逗号分隔的 JSONL 样本来源；默认用自演采集文件")
    t.add_argument("--name", default="omnipilot_sft")
    t.add_argument("--val-ratio", type=float, default=0.1)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--lora-r", type=int, default=16)
    t.add_argument("--max-len", type=int, default=1536)
    t.add_argument("--limit", type=int, default=None, help="仅用前 N 条样本(冒烟测试用)")
    t.add_argument("--batch", type=int, default=1, help="每设备批大小(有 GPU 可调大)")
    t.add_argument("--grad-accum", type=int, default=8, help="梯度累积步数")
    t.add_argument("--grad-ckpt", action="store_true", help="梯度检查点：4GB 显存必备，防溢出到内存")
    t.add_argument("--use-4bit", action="store_true", help="启用 QLoRA(需 CUDA torch + bitsandbytes)")
    t.add_argument("--export-ollama", default=None, help="训练后注册到 Ollama 的模型名")
    t.add_argument("--ollama-base", default=None, help="导出时挂载的 Ollama 基座模型名")

    p2 = sub.add_parser("deploy", help="合并 LoRA → 转 GGUF → 注册为 Ollama 模型")
    p2.add_argument("--base", required=True, help="基座模型(HF id 或本地路径)")
    p2.add_argument("--adapter", default=None, help="LoRA 适配器目录；默认取 omnipilot_sft")
    p2.add_argument("--merged", default=None, help="合并后 HF 模型输出目录")
    p2.add_argument("--gguf", default=None, help="GGUF 输出路径")
    p2.add_argument("--name", default="omnipilot:latest", help="Ollama 模型名")
    p2.add_argument("--temperature", type=float, default=0.2)
    p2.add_argument("--num-ctx", type=int, default=4096)
    p2.add_argument("--quantize", default="q4_K_M",
                    help="注册到 Ollama 时的量化档位（默认 q4_K_M）；填 none 表示不量化")
    p2.add_argument("--no-system", action="store_true",
                    help="不注入 SYSTEM 提示。决策模型(Jev 式 Choice/Noul/Score)必须加此项："
                         "它的训练样本是「单条 user(决策提示) -> 字母」、不含 system，"
                         "注入 SYSTEM 会让推理分布与训练不一致，字母概率随之失真。")

    pe = sub.add_parser("personal", help="查看个人画像与长期记忆（Personal Agent）")
    pe.add_argument("--forget", action="store_true", help="执行一次主动遗忘剪枝")

    b = sub.add_parser("budget", help="显存/内存预算与卸载建议（P0 资源治理）")
    b.add_argument("--model-gb", type=float, default=None, help="待运行模型体量，给出模式建议")
    b.add_argument("--model", default=None,
                   help="按已收录模型给块调度方案（qwen2.5-0.5b / 1.5b / 3b / 7b、qwen3-4b）")
    b.add_argument("--quant-bits", type=int, default=4, help="量化位宽（4/8/16）")
    b.add_argument("--ctx", type=int, default=None, help="指定上下文长度；缺省按显存自动推算")

    bm = sub.add_parser("bench", help="推理性能实测（TTFT / tok·s⁻¹），为报告提供真实指标")
    bm.add_argument("--models", required=True, help="逗号分隔的 Ollama 模型名")
    bm.add_argument("--n", type=int, default=64, help="每模型生成 token 数")

    ds = sub.add_parser("distill", help="用本地更强的模型当老师，蒸馏出经过验证的训练数据")
    ds.add_argument("--teacher", default="qwen3:4b", help="老师模型（Ollama 中的模型名）")
    ds.add_argument("--target", type=int, default=120, help="目标样本数")

    hc = sub.add_parser("halluc", help="幻觉评测（YOLO 式：幻觉率 / 弃答率）")
    hc.add_argument("--model", default="omnipilot:latest")
    hc.add_argument("--quiet", action="store_true")

    w = sub.add_parser("web", help="本地聊天控制台：真正驱动智能体执行（搜索/操控电脑）")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8765)
    w.add_argument("--model", default=None, help="大脑模型（默认取 OMNIPILOT_PLANNER 或 qwen3:4b）")
    w.add_argument("--domains", default=None, help="限制工具领域（逗号分隔；默认自动路由）")
    w.add_argument("--max-steps", type=int, default=None)
    w.add_argument("--headful", action="store_true", help="浏览器显示界面（默认无头）")
    return p


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    args = build_parser().parse_args(argv)
    if args.cmd == "doctor":
        return cmd_doctor(args)
    if args.cmd == "tools":
        return cmd_tools(args)
    if args.cmd == "run":
        from .config import MAX_STEPS

        if args.max_steps is None:
            args.max_steps = MAX_STEPS
        return cmd_run(args)
    if args.cmd == "interactive":
        from .config import MAX_STEPS

        if args.max_steps is None:
            args.max_steps = MAX_STEPS
        return cmd_interactive(args)
    if args.cmd == "web":
        from .config import MAX_STEPS

        if args.max_steps is None:
            args.max_steps = MAX_STEPS
        return cmd_web(args)
    if args.cmd == "data":
        return cmd_data(args)
    if args.cmd == "train":
        return cmd_train(args)
    if args.cmd == "deploy":
        return cmd_deploy(args)
    if args.cmd == "personal":
        return cmd_personal(args)
    if args.cmd == "budget":
        return cmd_budget(args)
    if args.cmd == "bench":
        return cmd_bench(args)
    if args.cmd == "distill":
        return cmd_distill(args)
    if args.cmd == "halluc":
        return cmd_halluc(args)
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""本地聊天控制台：真正驱动 OmniPilot 智能体去执行（搜索 / 操控电脑）。

为什么需要它：Ollama 桌面 App 只是**纯聊天窗口**，无法调用工具，所以模型只能回
"我没有工具能力"。本控制台把消息交给 OmniPilot 的 agent loop——模型只负责"想"
（输出 JSON 动作），Python 负责"做"（浏览器 / Office / 代码 / 媒体 / 系统），
并把每一步的执行轨迹显示出来。

运行（必须用装了运行依赖的解释器，例如系统 Python）：
    C:\\Python314\\python.exe -m omnipilot.cli web --model omnipilot:latest
然后浏览器打开 http://127.0.0.1:8765
"""
from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>OmniPilot 控制台</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; font-family: "Segoe UI","Microsoft YaHei",system-ui,sans-serif;
         background:#0f1115; color:#e8eaf0; height:100vh; display:flex; flex-direction:column; }
  header { padding:12px 18px; border-bottom:1px solid #242835; display:flex; align-items:center; gap:12px;
           background:#12151c; }
  header h1 { font-size:15px; margin:0; font-weight:600; letter-spacing:.3px; }
  header .sub { font-size:12px; color:#8b93a7; }
  header .spacer { flex:1; }
  select { background:#1b1f2a; color:#e8eaf0; border:1px solid #2c3242; border-radius:8px; padding:6px 10px; font-size:13px; }
  #log { flex:1; overflow-y:auto; padding:18px 18px 8px; }
  .row { display:flex; margin-bottom:14px; }
  .row.me { justify-content:flex-end; }
  .row.ai { flex-direction:column; align-items:flex-start; }
  .meta { font-size:12px; color:#6e7689; margin-top:6px; }
  .bubble { max-width:min(860px,86%); border-radius:12px; padding:10px 14px; line-height:1.6; font-size:14px; }
  .me .bubble { background:#2b5cff; color:#fff; border-bottom-right-radius:4px; }
  .ai .bubble { background:#181c26; border:1px solid #242a38; border-bottom-left-radius:4px; white-space:pre-wrap; }
  .ai .bubble.bad { border-color:#6b2b2b; background:#241618; }
  .trace { max-width:min(860px,86%); margin-top:8px; }
  details { background:#141822; border:1px solid #232a38; border-radius:10px; margin-bottom:8px; }
  summary { cursor:pointer; padding:8px 12px; font-size:13px; color:#aeb6c8; }
  summary b { color:#e8eaf0; font-weight:600; }
  .step { padding:6px 12px 10px 12px; font-size:13px; border-top:1px solid #1e2431; }
  .step .h { color:#cdd4e3; }
  .ok { color:#45c07a; } .fail { color:#e0605f; }
  pre { margin:6px 0 0; padding:8px 10px; background:#0d1017; border:1px solid #1d2230; border-radius:8px;
        overflow-x:auto; color:#9fb0cc; font-size:12px; white-space:pre-wrap; word-break:break-all; max-height:220px; }
  footer { padding:12px 18px; border-top:1px solid #242835; background:#12151c; display:flex; gap:10px; }
  textarea { flex:1; resize:none; height:52px; background:#1b1f2a; color:#e8eaf0; border:1px solid #2c3242;
             border-radius:10px; padding:10px 12px; font-size:14px; font-family:inherit; }
  button { background:#2b5cff; color:#fff; border:0; border-radius:10px; padding:0 22px; font-size:14px;
           cursor:pointer; }
  button:disabled { background:#33405f; cursor:default; }
  .hint { font-size:12px; color:#6e7689; padding:0 18px 10px; }
</style>
</head>
<body>
<header>
  <h1>OmniPilot 控制台</h1>
  <span class="sub">会真的执行（浏览器 / 办公 / 代码 / 媒体 / 系统）</span>
  <span class="spacer"></span>
  <span class="sub">大脑</span>
  <select id="model"></select>
</header>
<div id="log"></div>
<div class="hint">示例：帮我搜索桂林信息科技学院 / 打开 example.com 并告诉我页面标题 / 把 D:/a.mp4 转成 mp3</div>
<footer>
  <textarea id="input" placeholder="输入任务，Enter 发送，Shift+Enter 换行"></textarea>
  <button id="send">发送</button>
</footer>
<script>
const log = document.getElementById('log');
const input = document.getElementById('input');
const send = document.getElementById('send');
const modelSel = document.getElementById('model');
let busy = false;

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function addUser(text) {
  const row = el('div', 'row me');
  row.appendChild(el('div', 'bubble', text));
  log.appendChild(row); log.scrollTop = log.scrollHeight;
}

function addPending() {
  const row = el('div', 'row ai');
  const b = el('div', 'bubble', '执行中…');
  row.appendChild(b); log.appendChild(row); log.scrollTop = log.scrollHeight;
  return { row, b };
}

function renderTrace(container, trace) {
  if (!trace || !trace.length) return;
  const wrap = el('div', 'trace');
  const d = el('details');
  const s = el('summary');
  s.innerHTML = '执行轨迹 <b>' + trace.length + '</b> 步';
  d.appendChild(s);
  trace.forEach(function (t) {
    const st = el('div', 'step');
    const h = el('div', 'h');
    const status = t.ok ? '<span class="ok">成功</span>' : '<span class="fail">失败</span>';
    h.innerHTML = '[' + t.n + '] <b>' + t.tool + '</b> ' + JSON.stringify(t.args || {}) + ' — ' + status;
    st.appendChild(h);
    if (t.output) st.appendChild(el('pre', null, t.output));
    d.appendChild(st);
  });
  wrap.appendChild(d);
  container.appendChild(wrap);
}

async function loadModels() {
  try {
    const r = await fetch('/api/models'); const d = await r.json();
    modelSel.innerHTML = '';
    (d.models || []).forEach(function (m) {
      const o = document.createElement('option'); o.value = m; o.textContent = m; modelSel.appendChild(o);
    });
    const prefer = (d.models || []).find(function (m) { return m.indexOf('omnipilot') === 0; });
    if (prefer) modelSel.value = prefer;
  } catch (e) { /* 忽略 */ }
}

async function sendMsg() {
  const text = input.value.trim();
  if (!text || busy) return;
  busy = true; send.disabled = true; input.value = '';
  addUser(text);
  const p = addPending();
  try {
    const r = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, model: modelSel.value })
    });
    const d = await r.json();
    p.b.textContent = d.result || '(无结果)';
    if (!d.ok) p.b.classList.add('bad');
    renderTrace(p.row, d.trace);
    p.row.appendChild(el('div', 'meta', '模型 ' + (d.model || '') + ' · ' + (d.steps || 0) + ' 步 · 领域 ' +
      ((d.domains || []).join('/') || '-')));
  } catch (e) {
    p.b.textContent = '请求失败：' + e;
    p.b.classList.add('bad');
  } finally {
    busy = false; send.disabled = false; input.focus();
    log.scrollTop = log.scrollHeight;
  }
}

send.addEventListener('click', sendMsg);
input.addEventListener('keydown', function (e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); }
});
loadModels();
input.focus();
</script>
</body>
</html>
"""


def _list_models() -> list[str]:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
        return [m.get("name") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


class _State:
    """持有共享 Agent（复用浏览器会话），并对并发请求加锁。"""

    def __init__(self, model: str, domains: tuple[str, ...] | None,
                 max_steps: int, headless: bool) -> None:
        self.model = model
        self.domains = domains
        self.max_steps = max_steps
        self.headless = headless
        self.lock = threading.Lock()
        self.agents: dict[str, object] = {}

    def agent(self, model: str):
        a = self.agents.get(model)
        if a is None:
            from .agent import Agent

            a = Agent(planner_model=model, max_steps=self.max_steps,
                      headless=self.headless, verbose=False)
            self.agents[model] = a
        return a


def _make_handler(state: _State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # 静音默认访问日志
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except Exception:
                pass

        def do_GET(self):  # noqa: N802
            if self.path == "/" or self.path.startswith("/index"):
                self._send(200, _HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path.startswith("/api/models"):
                self._send(200, json.dumps({"models": _list_models()}, ensure_ascii=False).encode("utf-8"))
            else:
                self._send(404, b'{"error":"not found"}')

        def do_POST(self):  # noqa: N802
            if not self.path.startswith("/api/chat"):
                self._send(404, b'{"error":"not found"}')
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                req = {}
            msg = str(req.get("message") or "").strip()
            model = str(req.get("model") or state.model)
            if not msg:
                self._send(400, json.dumps({"error": "消息为空"}, ensure_ascii=False).encode("utf-8"))
                return
            try:
                with state.lock:
                    agent = state.agent(model)
                    out = agent.run(msg, domains=list(state.domains) if state.domains else None)
                payload = {"ok": out.get("ok"), "result": out.get("result"), "steps": out.get("steps"),
                           "domains": out.get("domains"), "trace": out.get("trace"), "model": model}
            except Exception as e:
                payload = {"ok": False, "result": f"执行出错：{type(e).__name__}: {e}", "steps": 0,
                           "domains": [], "trace": [], "model": model}
            self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765, model: str | None = None,
          domains: tuple[str, ...] | None = None, max_steps: int | None = None,
          headless: bool = True) -> None:
    from .config import MAX_STEPS, PLANNER_MODEL

    state = _State(model or PLANNER_MODEL, domains, max_steps or MAX_STEPS, headless)
    httpd = ThreadingHTTPServer((host, port), _make_handler(state))
    print(f"OmniPilot 控制台已启动：http://{host}:{port}    大脑模型: {state.model}")
    print("（Ctrl+C 停止）")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for a in state.agents.values():
            try:
                a.close()
            except Exception:
                pass
        httpd.server_close()

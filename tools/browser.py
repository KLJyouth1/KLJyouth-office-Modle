"""浏览器操控工具（基于 Playwright，本地已装好内核）。

默认 headless=True（不打扰你正在用的桌面）。核心价值：
`browser_snapshot` 直接读 DOM 返回"可交互元素 + 绝对坐标 + 文本"，
让模型无需视觉推断即可精确点击 —— 比纯截屏方案更稳、更省算力。
"""
from __future__ import annotations

import html as _html
import json
import re
import urllib.request
from urllib.parse import parse_qs, quote_plus, urlparse

from .base import FuncTool, Tool, ToolResult

_SNAPSHOT_JS = """
() => {
  const sel = 'a,button,input,select,textarea,[role=button],[role=link],[onclick]';
  const nodes = Array.from(document.querySelectorAll(sel));
  const out = [];
  for (let i = 0; i < nodes.length; i++) {
    const el = nodes[i];
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) continue;
    const st = window.getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none') continue;
    let label = (el.innerText || el.value || el.getAttribute('aria-label') ||
                 el.getAttribute('placeholder') || el.getAttribute('title') || '').trim();
    label = label.replace(/\\s+/g, ' ').slice(0, 40);
    out.push({
      index: out.length,
      tag: el.tagName.toLowerCase(),
      type: el.type || el.getAttribute('role') || '',
      text: label,
      x: Math.round(r.left + window.scrollX),
      y: Math.round(r.top + window.scrollY),
      w: Math.round(r.width),
      h: Math.round(r.height)
    });
    if (out.length >= 60) break;
  }
  return out;
}
"""


_SEARCH_JS = """
() => {
  const out = [];
  // Bing
  document.querySelectorAll('li.b_algo').forEach(li => {
    const a = li.querySelector('h2 a');
    if (!a || !a.href) return;
    const sn = li.querySelector('.b_caption p, .b_lineclamp1, .b_lineclamp2, .b_lineclamp3, .b_algoSlug');
    out.push({title: (a.innerText || '').trim(), url: a.href, snippet: (sn ? sn.innerText : '').trim().slice(0, 300)});
  });
  if (out.length) return out;
  // DuckDuckGo
  document.querySelectorAll("article[data-testid='result']").forEach(ar => {
    const a = ar.querySelector("a[data-testid='result-title-a']");
    if (!a || !a.href) return;
    const sn = ar.querySelector("[data-result='snippet']");
    out.push({title: (a.innerText || '').trim(), url: a.href, snippet: (sn ? sn.innerText : '').trim().slice(0, 300)});
  });
  return out;
}
"""


class BrowserTools:
    def __init__(self, headless: bool = True, timeout: int = 20000) -> None:
        self.headless = headless
        self.timeout = timeout
        self._pw = None
        self._browser = None
        self._ctx = None
        self._page = None

    # ------------------------------------------------------------------ 生命周期
    def _ensure(self):
        if self._page is not None:
            return
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        # 反爬关键：优先用**本机真实安装的 Chrome**（channel="chrome"）。
        # 实测 bundled Chromium 会被搜索引擎/风控识别：DuckDuckGo 直接跳 418、Bing 返回
        # 一堆假结果（查询框却是对的）。换成真实 Chrome 后 DuckDuckGo 稳定返回相关结果。
        args = ["--disable-blink-features=AutomationControlled", "--no-sandbox",
                "--disable-features=IsolateOrigins,site-per-process"]
        try:
            self._browser = self._pw.chromium.launch(channel="chrome", headless=self.headless, args=args)
        except Exception:
            self._browser = self._pw.chromium.launch(headless=self.headless, args=args)
        self._ctx = self._browser.new_context(
            user_agent=_UA, locale="zh-CN", timezone_id="Asia/Shanghai",
            viewport={"width": 1280, "height": 900},
        )
        self._ctx.add_init_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined});")
        self._page = self._ctx.new_page()
        self._page.set_default_timeout(self.timeout)

    def close(self):
        try:
            if self._browser:
                self._browser.close()
        finally:
            if self._pw:
                self._pw.stop()
            self._page = self._ctx = self._browser = self._pw = None

    # ------------------------------------------------------------------ 动作
    def goto(self, url: str) -> ToolResult:
        self._ensure()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        resp = self._page.goto(url, wait_until="domcontentloaded")
        title = self._page.title()
        return ToolResult(True, f"已打开 {url} —— 标题: {title}",
                          {"url": url, "title": title, "status": resp.status if resp else None})

    def click_text(self, text: str) -> ToolResult:
        self._ensure()
        loc = self._page.get_by_text(text, exact=False).first
        # 先滚动到可视区再点，避免"元素存在但不可点"的假失败
        loc.scroll_into_view_if_needed(timeout=self.timeout)
        loc.click(timeout=self.timeout)
        return ToolResult(True, f"已点击文本「{text}」")

    def click_selector(self, selector: str) -> ToolResult:
        self._ensure()
        loc = self._page.locator(selector).first
        loc.scroll_into_view_if_needed(timeout=self.timeout)
        loc.click(timeout=self.timeout)
        return ToolResult(True, f"已点击选择器 {selector}")

    def wait_for(self, text: str | None = None, selector: str | None = None,
                 timeout: int | None = None) -> ToolResult:
        """等待某个文本或元素出现（比盲等固定毫秒更可靠）。"""
        self._ensure()
        t = timeout or self.timeout
        try:
            if selector:
                self._page.wait_for_selector(selector, timeout=t, state="visible")
                return ToolResult(True, f"元素已出现: {selector}")
            if text:
                self._page.get_by_text(text, exact=False).first.wait_for(state="visible", timeout=t)
                return ToolResult(True, f"文本已出现: {text}")
        except Exception as e:
            return ToolResult(False, f"等待超时: {type(e).__name__}")
        return ToolResult(False, "需要提供 text 或 selector")

    def url(self) -> ToolResult:
        """廉价地读取当前 URL/标题（用于每步观察，开销远小于整页快照）。"""
        self._ensure()
        return ToolResult(True, f"URL: {self._page.url}\n标题: {self._page.title()}",
                          {"url": self._page.url, "title": self._page.title()})

    def click_xy(self, x: int, y: int) -> ToolResult:
        self._ensure()
        self._page.mouse.click(int(x), int(y))
        return ToolResult(True, f"已在页面坐标 ({x},{y}) 点击")

    def fill(self, selector: str, text: str) -> ToolResult:
        self._ensure()
        self._page.fill(selector, text)
        return ToolResult(True, f"已在 {selector} 填入文本")

    def press(self, key: str) -> ToolResult:
        self._ensure()
        self._page.keyboard.press(key)
        return ToolResult(True, f"已按键 {key}")

    def scroll(self, dy: int) -> ToolResult:
        self._ensure()
        self._page.mouse.wheel(0, int(dy))
        return ToolResult(True, f"已滚动 {dy} 像素")

    def back(self) -> ToolResult:
        self._ensure()
        self._page.go_back()
        return ToolResult(True, "已返回上一页")

    # ------------------------------------------------------------------ 感知
    def snapshot(self) -> ToolResult:
        """返回页面上可交互元素的清单(含绝对坐标)，供模型精确点击。"""
        self._ensure()
        els: list[dict] = self._page.evaluate(_SNAPSHOT_JS)
        lines = [
            f"[{e['index']}] {e['tag']}{('/'+e['type']) if e['type'] else ''} \"{e['text']}\" @({e['x']},{e['y']}) size {e['w']}x{e['h']}"
            for e in els
        ]
        return ToolResult(True, f"标题: {self._page.title()}\nURL: {self._page.url}\n" + "\n".join(lines),
                          {"elements": els, "url": self._page.url})

    def text(self, limit: int = 3000) -> ToolResult:
        self._ensure()
        body = self._page.inner_text("body")
        body = body[:limit]
        return ToolResult(True, body, {"length": len(body)})

    def screenshot(self, path: str | None = None) -> ToolResult:
        self._ensure()
        from ..config import RUN_DIR

        path = path or str(RUN_DIR / "browser_shot.png")
        self._page.screenshot(path=path, full_page=False)
        return ToolResult(True, f"页面截图: {path}", {"path": path})

    def wait(self, ms: int) -> ToolResult:
        self._ensure()
        self._page.wait_for_timeout(int(ms))
        return ToolResult(True, f"等待 {ms} 毫秒")

    # ------------------------------------------------------------ 联网检索 / 读网页
    # 关键决定：用**真实浏览器**渲染，而不是 urllib+正则。
    # 实测：DDG/Bing/Mojeek/searx 对纯 HTTP 抓取普遍返回 JS 壳页或被拦，正则 0 命中；
    #      真实浏览器（Playwright，本机已装内核）能稳定拿到结果。HTTP 仅作兜底。
    def _new_page(self):
        self._ensure()
        return self._ctx.new_page()

    def search(self, query: str, num: int = 5) -> ToolResult:
        """联网检索（真实 Chrome + DuckDuckGo，回退 Bing/HTTP）。不干扰当前工作页。"""
        from urllib.parse import quote_plus

        query = (query or "").strip()
        if not query:
            return ToolResult(False, "缺少 query 参数")
        num = max(1, min(int(num or 5), 10))
        engines = (
            ("https://duckduckgo.com/?q=" + quote_plus(query) + "&ia=web", "article[data-testid='result']"),
            ("https://www.bing.com/search?q=" + quote_plus(query) + "&setlang=zh-CN", "li.b_algo"),
        )
        try:
            page = self._new_page()
        except Exception:
            return _web_search_http(query, num)  # 浏览器起不来 → 退回 HTTP
        try:
            results: list[dict] = []
            for url, sel in engines:
                try:
                    page.goto(url, timeout=self.timeout, wait_until="domcontentloaded")
                    page.wait_for_selector(sel, timeout=8000)
                except Exception:
                    continue
                try:
                    results = page.evaluate(_SEARCH_JS) or []
                except Exception:
                    results = []
                if results:
                    break
        finally:
            try:
                page.close()
            except Exception:
                pass
        results = [r for r in results if r.get("url") and r.get("title")][:num]
        for r in results:  # Bing 的链接是 bing.com/ck/a?... 跳转壳，剥出真实 URL
            u = _clean_bing_url(r.get("url", ""))
            if u:
                r["url"] = u
        if not results:
            return _web_search_http(query, num)  # 浏览器没拿到 → 退回 HTTP
        lines = [f"[{i}] {r['title']}\n    {r['url']}\n    {r.get('snippet', '')}" for i, r in enumerate(results, 1)]
        return ToolResult(True, "\n".join(lines), {"query": query, "results": results})

    def fetch(self, url: str, limit: int = 4000) -> ToolResult:
        """联网抓取网页正文（真实浏览器渲染，能跑 JS），失败退回 HTTP。"""
        url = (url or "").strip()
        if not url:
            return ToolResult(False, "缺少 url 参数")
        if not re.match(r"^https?://", url, re.I):
            url = "https://" + url
        limit = max(200, min(int(limit or 4000), 20000))
        try:
            page = self._new_page()
        except Exception:
            return _web_fetch_http(url, limit)
        try:
            page.goto(url, timeout=self.timeout, wait_until="domcontentloaded")
            title = page.title()
            text = page.inner_text("body")
        except Exception:
            try:
                page.close()
            except Exception:
                pass
            return _web_fetch_http(url, limit)
        finally:
            try:
                page.close()
            except Exception:
                pass
        text = re.sub(r"\n{3,}", "\n\n", text or "").strip()[:limit]
        return ToolResult(True, f"标题: {title}\nURL: {url}\n\n{text}",
                          {"title": title, "url": url, "length": len(text)})


# ------------------------------------------------------------------ 联网检索 / 读网页（HTTP 兜底）
# 纯标准库实现（urllib + 正则），不引入第三方依赖、不需要 API Key。
# 用浏览器 UA，避免被搜索引擎以"脚本"身份限流。
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.S | re.I)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)


def _http_get(url: str, timeout: int = 20) -> tuple[bool, str]:
    """带浏览器 UA 的 GET，自动按响应头解码。返回 (ok, html_or_err)。"""
    req = urllib.request.Request(url, headers={
        "User-Agent": _UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            enc = r.headers.get_content_charset() or "utf-8"
            return True, raw.decode(enc, "replace")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _html_to_text(s: str) -> str:
    """把 HTML 粗清洗成人类可读文本：去脚本/样式、块级换行、去多余空白。"""
    s = _SCRIPT_RE.sub(" ", s)
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?i)</(p|div|li|tr|h[1-6]|section|article)>", "\n", s)
    s = _TAG_RE.sub(" ", s)
    s = _html.unescape(s)
    s = re.sub(r"[ \t\u00a0\u3000]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()


def _title_of(s: str) -> str:
    m = _TITLE_RE.search(s or "")
    return _html.unescape(_TAG_RE.sub("", m.group(1))).strip() if m else ""


def _clean_ddg_url(u: str) -> str:
    """DuckDuckGo 的结果链接会包一层跳转，剥出真实 URL。"""
    if u.startswith("//"):
        u = "https:" + u
    if "uddg=" in u:
        q = parse_qs(urlparse(u).query)
        if q.get("uddg"):
            return q["uddg"][0]
    return u


def _clean_bing_url(u: str) -> str:
    """Bing 结果链接常是 bing.com/ck/a?...u=a1<base64> 的跳转壳，剥出真实 URL；剥不出则丢弃。"""
    if "bing.com/ck/a" in u:
        v = (parse_qs(urlparse(u).query).get("u") or [""])[0]
        if v.startswith("a1"):
            import base64

            pad = "=" * (-len(v[2:]) % 4)
            try:
                return base64.urlsafe_b64decode(v[2:] + pad).decode("utf-8", "replace")
            except Exception:
                return ""
        return ""
    return u


def _ddg_search(query: str, num: int, timeout: int) -> list[dict]:
    ok, html = _http_get("https://html.duckduckgo.com/html/?q=" + quote_plus(query), timeout)
    if not ok:
        return []
    out: list[dict] = []
    for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.S):
        out.append({"title": _html_to_text(m.group(2)), "url": _clean_ddg_url(m.group(1)), "snippet": ""})
    snippets = re.findall(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', html, re.S)
    for i, sn in enumerate(snippets):
        if i < len(out):
            out[i]["snippet"] = _html_to_text(sn)[:300]
    return out[:num]


def _lite_ddg_search(query: str, num: int, timeout: int) -> list[dict]:
    """DuckDuckGo Lite 端点：比 html 版更轻，作为二级回退。"""
    ok, html = _http_get("https://lite.duckduckgo.com/lite/?q=" + quote_plus(query), timeout)
    if not ok:
        return []
    out: list[dict] = []
    for m in re.finditer(r'<a\b[^>]*class=["\']result-link["\'][^>]*>(.*?)</a>', html, re.S):
        hm = re.search(r'href=["\']([^"\']+)["\']', m.group(0))
        if not hm:
            continue
        out.append({"title": _html_to_text(m.group(1)), "url": _clean_ddg_url(hm.group(1)), "snippet": ""})
    snippets = re.findall(r'class=["\']result-snippet["\'][^>]*>(.*?)</td>', html, re.S)
    for i, sn in enumerate(snippets):
        if i < len(out):
            out[i]["snippet"] = _html_to_text(sn)[:300]
    return out[:num]


def _bing_search(query: str, num: int, timeout: int) -> list[dict]:
    ok, html = _http_get("https://www.bing.com/search?q=" + quote_plus(query) + "&setlang=zh-CN", timeout)
    if not ok:
        return []
    out: list[dict] = []
    for m in re.finditer(r'<li class="b_algo".*?<h2>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>\s*</h2>(.*?)</li>',
                         html, re.S):
        url = _clean_bing_url(m.group(1))
        if not url:
            continue
        out.append({"title": _html_to_text(m.group(2)), "url": url,
                    "snippet": _html_to_text(m.group(3))[:300]})
    return out[:num]


def _web_search_http(query: str, num: int = 5, timeout: int = 20) -> ToolResult:
    """HTTP 兜底检索（不经浏览器）：DuckDuckGo html → lite → Bing 三级回退。"""
    query = (query or "").strip()
    if not query:
        return ToolResult(False, "缺少 query 参数")
    num = max(1, min(int(num or 5), 10))
    results = (_ddg_search(query, num, timeout)
               or _lite_ddg_search(query, num, timeout)
               or _bing_search(query, num, timeout))
    results = [r for r in results if r.get("url") and r.get("title")][:num]
    if not results:
        return ToolResult(False, "联网检索失败：搜索引擎未返回结果（可能被限流或断网）")
    lines = [f"[{i}] {r['title']}\n    {r['url']}\n    {r['snippet']}" for i, r in enumerate(results, 1)]
    return ToolResult(True, "\n".join(lines), {"query": query, "results": results})


def _web_fetch_http(url: str, limit: int = 4000, timeout: int = 25) -> ToolResult:
    """HTTP 兜底抓取（不经浏览器）：urllib 取回后清洗为可读正文。"""
    url = (url or "").strip()
    if not url:
        return ToolResult(False, "缺少 url 参数")
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    ok, html = _http_get(url, timeout)
    if not ok:
        return ToolResult(False, f"抓取失败：{html}")
    limit = max(200, min(int(limit or 4000), 20000))
    title = _title_of(html)
    text = _html_to_text(html)[:limit]
    return ToolResult(True, f"标题: {title}\nURL: {url}\n\n{text}",
                      {"title": title, "url": url, "length": len(text)})


def build_tools(headless: bool = True) -> list[Tool]:
    b = BrowserTools(headless=headless)
    d = "browser"
    obj = lambda props, req=None: {"type": "object", "properties": props, "required": req or []}  # noqa: E731
    return [
        FuncTool("browser_goto", "打开网址", b.goto, obj({"url": {"type": "string"}}, ["url"]), d),
        FuncTool("browser_click_text", "点击页面上包含该文本的元素", b.click_text,
                 obj({"text": {"type": "string"}}, ["text"]), d),
        FuncTool("browser_click_selector", "按 CSS 选择器点击元素", b.click_selector,
                 obj({"selector": {"type": "string"}}, ["selector"]), d),
        FuncTool("browser_click_xy", "在页面坐标处点击", b.click_xy,
                 obj({"x": {"type": "integer"}, "y": {"type": "integer"}}, ["x", "y"]), d),
        FuncTool("browser_fill", "在输入框中填文本", b.fill,
                 obj({"selector": {"type": "string"}, "text": {"type": "string"}}, ["selector", "text"]), d),
        FuncTool("browser_press", "按键盘键(Enter/Tab/PageDown 等)", b.press,
                 obj({"key": {"type": "string"}}, ["key"]), d),
        FuncTool("browser_scroll", "页面滚动(正数向下)", b.scroll,
                 obj({"dy": {"type": "integer"}}, ["dy"]), d),
        FuncTool("browser_back", "返回上一页", b.back, obj({}), d),
        FuncTool("browser_wait_for", "等待文本或选择器出现(比盲等更可靠)", b.wait_for,
                 obj({"text": {"type": "string"}, "selector": {"type": "string"}, "timeout": {"type": "integer"}}), d),
        FuncTool("browser_url", "读取当前页面 URL 与标题", b.url, obj({}), d),
        FuncTool("browser_snapshot", "获取页面可交互元素清单(含坐标)，用于精确点击", b.snapshot, obj({}), d),
        FuncTool("browser_text", "获取页面正文文本", b.text, obj({"limit": {"type": "integer"}}), d),
        FuncTool("browser_screenshot", "页面截图", b.screenshot, obj({"path": {"type": "string"}}), d),
        FuncTool("browser_wait", "等待若干毫秒", b.wait, obj({"ms": {"type": "integer"}}, ["ms"]), d),
        # 联网检索/读网页：真实浏览器渲染（Bing→DDG），HTTP 兜底
        FuncTool("web_search", "联网检索关键词，返回若干条 标题/链接/摘要", b.search,
                 obj({"query": {"type": "string"}, "num": {"type": "integer"}}, ["query"]), d),
        FuncTool("web_fetch", "联网抓取指定网址并渲染出可读正文", b.fetch,
                 obj({"url": {"type": "string"}, "limit": {"type": "integer"}}, ["url"]), d),
    ]

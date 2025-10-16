# streamlit_app.py
# Canvas Course-wide HTML Rewrite (Test Instance)
# - Admin-only Streamlit app for dry-run + bulk apply on Pages & Assignment descriptions
# - Canvas API (test instance only), OpenAI Responses API (auto-pick newest model + retries)
# - Iframe freeze/restore (no new iframes), NO sanitizer (write model HTML as-is)
# - Visual PREVIEW (Original vs Proposed) using a built-in DesignPLUS-like "Shim" (CSS+JS) – no external URLs needed
# - ETA during dry-run
# - Approve All / Unapprove All bulk toggles
# - Structured "Spec" controls for precise, deterministic transforms (banner, theme, palette, headings, callouts, buttons, tables, images)
# - PRE/POST transform pipeline around the LLM to guarantee precision

import os
import re
import json
import base64
import time
import random
import hashlib
import urllib.parse
from typing import Optional, Tuple, List

import streamlit as st
from canvasapi import Canvas
from openai import OpenAI
from bs4 import BeautifulSoup, NavigableString, Tag
from diff_match_patch import diff_match_patch

# ---------------------- Config & Clients ----------------------

SECRETS = st.secrets if hasattr(st, "secrets") else os.environ

CANVAS_BASE_URL = SECRETS["CANVAS_BASE_URL"]
CANVAS_ACCOUNT_ID = int(SECRETS["CANVAS_ACCOUNT_ID"])
CANVAS_ADMIN_TOKEN = SECRETS["CANVAS_ADMIN_TOKEN"]
OPENAI_API_KEY = SECRETS["OPENAI_API_KEY"]

# Defaults if auto-pick fails or listing isn't permitted
DEFAULT_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-5")    # falls back to 4.1 if unavailable
DEFAULT_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

st.set_page_config(page_title="Canvas Course-wide HTML Rewrite (Test)", layout="wide")

canvas = Canvas(CANVAS_BASE_URL, CANVAS_ADMIN_TOKEN)
account = canvas.get_account(CANVAS_ACCOUNT_ID)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------- Small utils ----------------------

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

# ---------------------- Canvas helpers ----------------------

def find_course_by_code(account, course_code: str) -> List:
    """Search by course_code; prefer exact code match if available."""
    courses = account.get_courses(search_term=course_code)
    matches = [c for c in courses if getattr(c, "course_code", "") == course_code]
    return matches if matches else list(courses)

def list_supported_items(course):
    """Enumerate Module Items and keep only Pages and Assignments."""
    supported = []
    for module in course.get_modules(include_items=True):
        items = list(module.get_module_items()) if not getattr(module, "items", None) else module.items
        for it in items:
            if it.type in {"Page", "Assignment"}:
                supported.append((module, it))
    return supported

def fetch_item_html(course, item):
    """Fetch the HTML-bearing body for a supported item."""
    if item.type == "Page":
        page = course.get_page(item.page_url)
        return {"kind": "Page", "id": page.page_id, "url": page.url, "title": page.title, "html": page.body or ""}
    elif item.type == "Assignment":
        asg = course.get_assignment(item.content_id)
        return {"kind": "Assignment", "id": asg.id, "title": asg.name, "html": asg.description or ""}

def apply_update(course, item, new_html: str):
    if item.type == "Page":
        course.get_page(item.page_url).edit(wiki_page={"body": new_html})
    elif item.type == "Assignment":
        course.get_assignment(item.content_id).edit(assignment={"description": new_html})

# ---------------------- Iframe freeze/restore ----------------------

PLACEHOLDER_FMT = "⟪IFRAME:{i}⟫"

def protect_iframes(html: str):
    """Replace existing iframes with placeholders and return (frozen_html, mapping, hosts)."""
    soup = BeautifulSoup(html or "", "html.parser")
    mapping, hosts = {}, set()
    i = 1
    for tag in soup.find_all("iframe"):
        src = (tag.get("src") or "").strip()
        host = urllib.parse.urlparse(src).hostname or ""
        if host:
            hosts.add(host.lower())
        placeholder = PLACEHOLDER_FMT.format(i=i)
        mapping[placeholder] = str(tag)
        tag.replace_with(placeholder)
        i += 1
    return str(soup), mapping, hosts

def restore_iframes(html: str, mapping: dict) -> str:
    """Replace placeholders with the original <iframe> markup."""
    for ph, original in mapping.items():
        html = html.replace(ph, original)
    return html

def strip_new_iframes(html: str) -> str:
    """Remove any iframes the model added (we only want to restore originals)."""
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup.find_all("iframe"):
        tag.decompose()
    return str(soup)

# ---------------------- Model-reference helpers ----------------------

def html_to_skeleton(model_html: str, max_nodes: int = 2000, max_text: int = 80) -> str:
    """
    Build a compact, valid-ish skeleton of the model HTML:
    - Preserves tag hierarchy
    - Keeps only id/class/role/aria-/data-* attributes
    - Keeps short text only for headings/labels/summary
    - Never mutates the parsed DOM (avoids bs4 edge cases)
    """
    soup = BeautifulSoup(model_html or "", "html.parser")

    def keep_attr(k: str) -> bool:
        return k in ("id", "class", "role") or k.startswith("aria-") or k.startswith("data-")

    def fmt_attrs(attrs: dict) -> str:
        parts = []
        for k, v in (attrs or {}).items():
            if not keep_attr(k):
                continue
            if isinstance(v, (list, tuple)):
                v = " ".join(map(str, v))
            v = " ".join(str(v).split())[:120]
            parts.append(f'{k}="{v}"')
        return (" " + " ".join(parts)) if parts else ""

    headings = {"h1","h2","h3","h4","h5","h6","summary","label"}

    count = [0]

    def render(node) -> str:
        if count[0] >= max_nodes:
            return ""
        if isinstance(node, NavigableString):
            return ""
        if not isinstance(node, Tag):
            return ""

        count[0] += 1
        start = f"<{node.name}{fmt_attrs(node.attrs)}>"
        inner = []

        if node.name in headings:
            direct_text = []
            for child in node.children:
                if isinstance(child, NavigableString):
                    direct_text.append(str(child))
            text = " ".join(" ".join(direct_text).split())
            if text:
                inner.append(text[:max_text])

        for child in node.children:
            if isinstance(child, Tag):
                piece = render(child)
                if piece:
                    inner.append(piece)
            if count[0] >= max_nodes:
                break

        end = f"</{node.name}>"
        return start + "".join(inner) + end

    roots = list(soup.body.children) if soup.body else list(soup.children)
    out = []
    for child in roots:
        piece = render(child)
        if piece:
            out.append(piece)
        if count[0] >= max_nodes:
            break

    text = "".join(out)
    text = " ".join(text.split())
    if len(text) > 120_000:
        text = text[:120_000] + " <!-- truncated -->"
    return text

def find_single_course_by_code(account, course_code: str):
    """Return list of courses matching code (UI will disambiguate)."""
    courses = account.get_courses(search_term=course_code)
    return list(courses)

def list_all_pages(course):
    """Return [(title_or_slug, url_slug)] for all pages."""
    pages = []
    for p in course.get_pages():
        pages.append((getattr(p, "title", p.url), p.url))
    return pages

def list_all_assignments(course):
    """Return [(name, id)] for all assignments."""
    items = []
    for a in course.get_assignments():
        items.append((getattr(a, "name", f"Assignment {a.id}"), a.id))
    return items

def fetch_model_item_html(course, kind: str, ident):
    """Fetch full HTML for the selected model item."""
    if kind == "Page":
        page = course.get_page(ident)  # ident = page.url (slug)
        return page.body or ""
    elif kind == "Assignment":
        asg = course.get_assignment(int(ident))  # ident = assignment id
        return asg.description or ""
    return ""

def image_to_data_url(file) -> Tuple[str, str]:
    """Accept a Streamlit UploadedFile and return (data_url, mime)."""
    if file is None:
        raise ValueError("No image provided")
    mime = file.type or "image/png"
    raw = None
    try:
        raw = bytes(file.getbuffer())
    except Exception:
        pass
    if not raw:
        try:
            file.seek(0)
        except Exception:
            pass
        raw = file.read()
    if not raw:
        raise ValueError("Empty image or no bytes read")
    b64 = base64.b64encode(raw).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"
    return data_url, mime

# ---------------------- Auto-pick newest model ----------------------

def list_models(client: OpenAI):
    """Return a list of model objects; empty list if listing isn't permitted."""
    try:
        res = client.models.list()
        data = getattr(res, "data", res)
        return list(data)
    except Exception:
        return []

def latest_model_id(client: OpenAI, pattern: str, default_id: str) -> str:
    """
    Pick the newest model (by created timestamp) whose id matches `pattern` (regex).
    Fallback to default_id if none match or listing isn't allowed.
    """
    models = list_models(client)
    try:
        matches = [m for m in models if re.search(pattern, m.id)]
        if not matches:
            return default_id
        matches.sort(key=lambda m: getattr(m, "created", 0), reverse=True)
        return matches[0].id
    except Exception:
        return default_id

# ---------------------- OpenAI + retries ----------------------

SYSTEM_PROMPT = (
    "You are an expert Canvas HTML editor. Preserve links, anchors/IDs, classes, and data-* attributes. "
    "Placeholders like ⟪IFRAME:n⟫ represent protected iframes—do not add, remove, or reorder them. "
    "Follow the policy. Return only HTML, no explanations."
)

DT_MODES = ["Preserve", "Enhance", "Replace"]

def _create_with_retries(
    client: OpenAI,
    model: str,
    payload_input,
    fallback_model: Optional[str] = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
):
    """
    Call Responses API with retries for 5xx/timeout/network errors.
    If all retries fail, optionally try once with fallback_model (also retried).
    Returns the response or raises the last exception.
    """
    def _do_call(m: str):
        return client.responses.create(model=m, input=payload_input, temperature=0.2)

    def _should_retry(err: Exception) -> bool:
        s = str(err).lower()
        return any(k in s for k in ["server_error", "status code: 5", "timed out", "timeout", "temporarily unavailable"])

    last_err = None

    for attempt in range(max_retries):
        try:
            return _do_call(model)
        except Exception as e:
            last_err = e
            if not _should_retry(e):
                break
            time.sleep(base_delay * (2 ** attempt) + random.random() * 0.5)

    if fallback_model and fallback_model != model:
        for attempt in range(max_retries):
            try:
                return _do_call(fallback_model)
            except Exception as e:
                last_err = e
                if not _should_retry(e):
                    break
                time.sleep(base_delay * (2 ** attempt) + random.random() * 0.5)

    raise last_err

def openai_rewrite(
    user_request: str,
    html: str,
    dt_mode: str,
    policy_hard_rules: dict,
    model_html_skeleton: Optional[str] = None,
    model_image_data_url: Optional[str] = None,
    model_text_id: str = DEFAULT_TEXT_MODEL,
    model_vision_id: str = DEFAULT_VISION_MODEL,
) -> str:
    """
    Call OpenAI Responses API to rewrite the HTML.
    If model_image_data_url is present, send a multimodal input with an image.
    If model_html_skeleton is present, include it as a reference text block.
    """
    policy = {
        "design_tools_mode": dt_mode,
        "allow_inline_styles": True,
        "block_new_iframes": True,
        "reference_model": ("image" if model_image_data_url else "html" if model_html_skeleton else "none"),
        "reference_usage": "Match layout/sectioning/components; do not copy literal course-specific links or text.",
        "hard_requirements": policy_hard_rules or {}
    }

    hard_rules_text = (
        "Hard rules:\n"
        "- Do not remove existing anchors/IDs/classes/data-* unless conflicting with required theme/classes.\n"
        "- Do not create new iframes. Respect ⟪IFRAME:n⟫ placeholders.\n"
        "- Enforce the provided theme/classes; adjust DesignPLUS components as permitted by policy.\n"
    )

    text_blocks = [
        hard_rules_text,
        "Policy: " + json.dumps(policy, ensure_ascii=False),
        f"DesignTools Mode: {dt_mode}",
        "Rewrite goals: " + user_request,
        "HTML to rewrite follows:",
        html,
    ]
    if model_html_skeleton:
        text_blocks.insert(4, "Model HTML skeleton (follow structure/classes, not text):\n" + model_html_skeleton)

    if not model_image_data_url:
        prompt = "\n\n".join(text_blocks)
        payload = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        resp = _create_with_retries(
            client=openai_client,
            model=model_text_id,
            payload_input=payload,
            fallback_model="gpt-4.1",
            max_retries=3,
            base_delay=1.0,
        )
    else:
        try:
            content_parts = [
                {"type": "input_text", "text": "\n\n".join(text_blocks[:4])}
            ]
            if model_html_skeleton:
                content_parts.append({"type": "input_text", "text": text_blocks[4]})
            content_parts.append({"type": "input_image", "image_url": {"url": model_image_data_url}})
            content_parts.append({"type": "input_text", "text": "\n".join(text_blocks[-2:])})

            payload = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content_parts},
            ]
            resp = _create_with_retries(
                client=openai_client,
                model=model_vision_id,
                payload_input=payload,
                fallback_model="gpt-4o",
                max_retries=3,
                base_delay=1.0,
            )
        except Exception:
            prompt = "\n\n".join(text_blocks) + "\n\n(Note: image reference unavailable; proceed using textual model description if any.)"
            payload = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            resp = _create_with_retries(
                client=openai_client,
                model=model_text_id,
                payload_input=payload,
                fallback_model="gpt-4.1",
                max_retries=3,
                base_delay=1.0,
            )

    try:
        return resp.output_text  # newer SDKs
    except AttributeError:
        return resp.output[0].content[0].text  # fallback

# ---------------------- Diff ----------------------

def html_diff(old: str, new: str) -> str:
    dmp = diff_match_patch()
    d = dmp.diff_main(old or "", new or "")
    dmp.diff_cleanupSemantic(d)
    return dmp.diff_prettyHtml(d)

# ---------------------- Shim CSS/JS & preview ----------------------

SHIM_CSS = """
/* --- DesignPLUS-like Shim (approximate) --- */
:root {
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Lato, Arial, sans-serif;
  --csu-green: #1E4D2B;
  --csu-gold: #C8C372;
  --neutral-900: #262626;
  --neutral-100: #f5f5f5;
  --border: #e2e8f0;
}
html, body { margin:0; padding:0; font-family:var(--font); line-height:1.5; background:#fff; color:#111; }
#frame { max-width: 980px; margin: 0 auto; padding: 24px; background: #fff; }
h1,h2,h3,h4,h5,h6 { margin: 0.6em 0 0.4em; line-height:1.25; }
p,li { font-size: 16px; }
a { color: #005f85; text-decoration: underline; }
img { max-width: 100%; height: auto; }

/* Banner approximate */
.dp-banner, .dpl-banner, .designplus.banner {
  border-radius: 10px; padding: 16px 20px; color: #fff;
  background: linear-gradient(135deg, var(--csu-green), #0f2a18);
  display:flex; align-items:center; gap: 12px; min-height: 140px;
}
.dp-banner.dp-banner--lg { min-height: 180px; }
.dp-theme--circle-left-1 .dp-banner { background: radial-gradient(circle at left, var(--csu-green), #0f2a18); }

/* Callouts */
.dp-callout, .dpl-callout {
  border-radius: 8px; padding: 12px 14px; border:1px solid var(--border); background:#f9fafb; margin: 12px 0;
}
.dp-callout--info { background:#eef6ff; border-color:#bfdbfe; }
.dp-callout--warning { background:#fff7ed; border-color:#fed7aa; }
.dp-callout--success { background:#ecfdf5; border-color:#a7f3d0; }

/* Buttons */
.dpl-button, .dp-button, a.button {
  display:inline-block; border-radius:6px; padding:10px 14px; text-decoration:none; font-weight:600; border:1px solid transparent;
}
.dpl-button--primary { background: var(--csu-green); color:#fff; }
.dpl-button--ghost { background: transparent; border-color: var(--csu-green); color: var(--csu-green); }

/* Tabs (approx) */
.tabs { margin: 16px 0; }
.tab-list { display:flex; gap:8px; border-bottom:1px solid var(--border); padding-bottom:6px; }
.tab-list button { background:#fff; border:1px solid var(--border); border-bottom:none; padding:8px 12px; border-radius:6px 6px 0 0; cursor:pointer; }
.tab-list button.active { background:#fff; border-color: var(--csu-green); color: var(--csu-green); }
.tab-panel { border:1px solid var(--border); border-radius:0 6px 6px 6px; padding:12px; display:none; }
.tab-panel.active { display:block; }

/* Accordions (using <details>) */
details { border:1px solid var(--border); border-radius:8px; padding:8px 12px; margin:10px 0; }
details > summary { font-weight:600; cursor:pointer; }

/* Tables */
table { border-collapse: collapse; width: 100%; }
.ic-Table th, .ic-Table td { border: 1px solid #e5e7eb; padding: 8px; }
.ic-Table.zebra tr:nth-child(even) { background: #fafafa; }

/* Misc containers */
.designplus, .dp-component, .dpl { border-radius: 6px; padding: 10px; border: 1px dashed #cbd5e1; background: #f8fafc; }
.iframe-placeholder { border:1px dashed #cbd5e1; padding:12px; background:#fbfbfb; color:#555; border-radius:6px; }
"""

SHIM_JS = """
// Minimal DP-like interactions (tabs)
(function(){
  function initTabs(root){
    const tabsets = root.querySelectorAll('.tabs');
    tabsets.forEach(ts => {
      const btns = ts.querySelectorAll('.tab-list button');
      const panels = ts.querySelectorAll('.tab-panel');
      function activate(i){
        btns.forEach((b, idx)=> b.classList.toggle('active', idx===i));
        panels.forEach((p, idx)=> p.classList.toggle('active', idx===i));
      }
      btns.forEach((b, i) => b.addEventListener('click', () => activate(i)));
      if (btns.length && panels.length) activate(0);
    });
  }
  document.addEventListener('DOMContentLoaded', () => initTabs(document));
})();
"""

def neutralize_iframes_for_preview(html: str) -> str:
    """Replace iframes with a neutral placeholder (avoid external loads in preview)."""
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup.find_all("iframe"):
        src = (tag.get("src") or "").strip()
        host = urllib.parse.urlparse(src).hostname or "iframe"
        placeholder = soup.new_tag("div", attrs={"class": "iframe-placeholder"})
        placeholder.string = f"[iframe: {host}]"
        tag.replace_with(placeholder)
    return str(soup)

def preview_html_document(inner_html: str, use_js_shim: bool, extra_css_inline: str = "") -> str:
    """
    Wrap course HTML in a minimal, sandbox-friendly page for st.components.v1.html.
    Includes DP-like shim CSS, optional shim JS, and optional extra inline CSS.
    """
    safe_inner = inner_html or ""
    return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>{SHIM_CSS}\n{extra_css_inline or ""}</style>
  </head>
  <body>
    <main id="frame">
      {safe_inner}
    </main>
    {"<script>"+SHIM_JS+"</script>" if use_js_shim else ""}
  </body>
</html>"""

# ---------------------- Deterministic transforms (Spec) ----------------------

def parse_color_to_rgb(s: str) -> Optional[Tuple[int,int,int]]:
    s = s.strip()
    # hex formats
    m = re.fullmatch(r"#([0-9a-fA-F]{3,8})", s)
    if m:
        h = m.group(1)
        if len(h) in (3,4):
            r = int(h[0]*2,16); g=int(h[1]*2,16); b=int(h[2]*2,16)
            return (r,g,b)
        if len(h) in (6,8):
            r = int(h[0:2],16); g=int(h[2:4],16); b=int(h[4:6],16)
            return (r,g,b)
    # rgb/rgba
    m = re.fullmatch(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(,\s*[\d.]+\s*)?\)", s, re.I)
    if m:
        return (int(float(m.group(1))), int(float(m.group(2))), int(float(m.group(3))))
    # hsl/hlsa -> rough conversion
    m = re.fullmatch(r"hsla?\(\s*([\d.]+)\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%\s*(,\s*[\d.]+\s*)?\)", s, re.I)
    if m:
        h = float(m.group(1)) % 360
        s2 = float(m.group(2))/100.0
        l = float(m.group(3))/100.0
        # convert HSL to RGB (simple)
        c = (1 - abs(2*l - 1)) * s2
        x = c * (1 - abs((h/60)%2 - 1))
        m0 = l - c/2
        r,g,b = 0,0,0
        if 0<=h<60: r,g,b = c,x,0
        elif 60<=h<120: r,g,b = x,c,0
        elif 120<=h<180: r,g,b = 0,c,x
        elif 180<=h<240: r,g,b = 0,x,c
        elif 240<=h<300: r,g,b = x,0,c
        else: r,g,b = c,0,x
        return (int((r+m0)*255), int((g+m0)*255), int((b+m0)*255))
    return None

def rgb_to_hex(rgb: Tuple[int,int,int]) -> str:
    r,g,b = rgb
    return f"#{max(0,min(255,r)):02X}{max(0,min(255,g)):02X}{max(0,min(255,b)):02X}"

def nearest_color(rgb: Tuple[int,int,int], allowed_rgbs: List[Tuple[int,int,int]]) -> Tuple[int,int,int]:
    best = allowed_rgbs[0]
    br,bg,bb = rgb
    best_d = 1e9
    for (r,g,b) in allowed_rgbs:
        d = (r-br)**2 + (g-bg)**2 + (b-bb)**2
        if d < best_d:
            best_d = d
            best = (r,g,b)
    return best

def normalize_palette(html: str, allowed: List[str], forbidden: List[str], map_strategy: str) -> str:
    """Map/strip inline colors to conform to allowed palette."""
    if not allowed and not forbidden:
        return html
    soup = BeautifulSoup(html or "", "html.parser")

    allowed_set = [c.strip() for c in allowed if c.strip()]
    allowed_rgbs = [parse_color_to_rgb(c) for c in allowed_set if parse_color_to_rgb(c)]
    forbidden_set = set(c.strip().lower() for c in forbidden if c.strip())

    def fix_style_value(style_value: str) -> str:
        # Replace colors in common CSS props
        def repl_color(match):
            color_str = match.group(0)
            lower = color_str.lower()
            if lower in forbidden_set:
                return "" if map_strategy in ("strip","exact") else allowed_set[0] if allowed_set else color_str
            rgb = parse_color_to_rgb(color_str)
            if not rgb:
                return color_str
            hexv = rgb_to_hex(rgb)
            if map_strategy == "exact" and hexv.upper() not in [a.upper() for a in allowed_set]:
                return ""  # strip declaration value; caller keeps property syntax simple
            if map_strategy == "nearest" and allowed_rgbs:
                nrgb = nearest_color(rgb, allowed_rgbs)
                return rgb_to_hex(nrgb)
            if map_strategy == "strip":
                return ""
            return color_str

        # naive find colors
        style_value = re.sub(r"#(?:[0-9a-fA-F]{3,8})\b", repl_color, style_value)
        style_value = re.sub(r"rgba?\([^)]+\)", repl_color, style_value)
        style_value = re.sub(r"hsla?\([^)]+\)", repl_color, style_value)
        return style_value

    for el in soup.find_all(True):
        # legacy attributes
        if el.has_attr("color"):
            c = el["color"]
            rgb = parse_color_to_rgb(c) or parse_color_to_rgb("#"+c if not c.startswith("#") else c)
            if rgb:
                if map_strategy == "nearest" and allowed_rgbs:
                    el["color"] = rgb_to_hex(nearest_color(rgb, allowed_rgbs))
                elif map_strategy in ("strip","exact") and rgb_to_hex(rgb).upper() not in [a.upper() for a in allowed_set]:
                    del el["color"]
        if el.has_attr("bgcolor"):
            c = el["bgcolor"]
            rgb = parse_color_to_rgb(c) or parse_color_to_rgb("#"+c if not c.startswith("#") else c)
            if rgb:
                if map_strategy == "nearest" and allowed_rgbs:
                    el["bgcolor"] = rgb_to_hex(nearest_color(rgb, allowed_rgbs))
                elif map_strategy in ("strip","exact") and rgb_to_hex(rgb).upper() not in [a.upper() for a in allowed_set]:
                    del el["bgcolor"]

        # inline styles
        if el.has_attr("style"):
            fixed = fix_style_value(el["style"])
            # Clean up potential empty declarations like "color: ;"
            fixed = re.sub(r";?\s*(color|background(?:-color)?|border-color)\s*:\s*(;|$)", ";", fixed)
            el["style"] = fixed.strip(" ;")

    return str(soup)

def enforce_theme_class(html: str, theme_class: str, container_selector: str = "body") -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    target = soup.select_one(container_selector) or soup
    classes = set((target.get("class") or []))
    if theme_class:
        classes.add(theme_class)
        target["class"] = list(classes)
    return str(soup)

def normalize_headings(html: str, min_h: int = 2, enforce_hierarchy: bool = True) -> str:
    """Ensure headings start from min_h and no jumps in level."""
    soup = BeautifulSoup(html or "", "html.parser")
    current_level = min_h - 1
    for h in soup.find_all(re.compile(r"^h[1-6]$", re.I)):
        level = int(h.name[1])
        if level < min_h:
            level = min_h
        if enforce_hierarchy:
            level = max(level, min(current_level + 1, 6))
            current_level = level
        h.name = f"h{level}"
    return str(soup)

def normalize_callouts(html: str, allowed_variants: List[str]) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    allowed = set(allowed_variants or [])
    for el in soup.find_all(True):
        cls = " ".join(el.get("class", []))
        if "callout" in cls or "dp-callout" in cls or "dpl-callout" in cls:
            # coerce to first allowed if none detected
            found = None
            for v in allowed:
                if v in cls:
                    found = v
                    break
            variant = found or (list(allowed)[0] if allowed else "info")
            # normalize classes
            classes = [c for c in el.get("class", []) if not c.startswith("dp-callout--")]
            classes = list(dict.fromkeys(classes + ["dp-callout", f"dp-callout--{variant}"]))
            el["class"] = classes
    return str(soup)

def upgrade_buttons_and_links(html: str, button_class: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for a in soup.find_all("a"):
        cl = a.get("class", [])
        style = a.get("style", "")
        looks_button = ("button" in " ".join(cl).lower()) or ("background" in style and "padding" in style)
        if looks_button and button_class.strip():
            merged = list(dict.fromkeys(cl + button_class.split()))
            a["class"] = merged
        href = a.get("href", "")
        if href.startswith("http"):
            # ensure noopener without forcing noreferrer if you don't want it
            rel = set((a.get("rel") or []))
            rel.add("noopener")
            a["rel"] = list(rel)
    return str(soup)

def normalize_tables(html: str, table_class: str = "ic-Table", zebra: bool = True) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for t in soup.find_all("table"):
        classes = set(t.get("class", []))
        classes.add(table_class)
        if zebra:
            classes.add("zebra")
        t["class"] = list(classes)
        # add scope to th
        for th in t.find_all("th"):
            if not th.has_attr("scope"):
                th["scope"] = "col"
    return str(soup)

def ensure_images(html: str, max_width: str = "100%", require_alt: bool = True) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for img in soup.find_all("img"):
        if max_width:
            style = img.get("style", "")
            if "max-width" not in style:
                img["style"] = (style + f";max-width:{max_width}").strip(";")
        if require_alt and not img.get("alt"):
            img["alt"] = "Image"
    return str(soup)

def transform_banner(html: str, selector: str, classes: str, style: str, inner_html: Optional[str], insert_if_missing: bool) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    target = soup.select_one(selector) if selector else None
    if not target and insert_if_missing:
        new_div = soup.new_tag("div")
        if classes.strip():
            new_div["class"] = classes.split()
        if style.strip():
            new_div["style"] = style
        if inner_html and inner_html.strip():
            frag = BeautifulSoup(inner_html, "html.parser")
            # append children to new_div
            frag_root = frag.body or frag
            for c in list(frag_root.children):
                new_div.append(c if isinstance(c, Tag) else NavigableString(str(c)))
        # insert at top
        if soup.body:
            soup.body.insert(0, new_div)
        else:
            soup.insert(0, new_div)
        return str(soup)

    if target:
        if classes.strip():
            target["class"] = list(set((target.get("class") or []) + classes.split()))
        if style.strip():
            existing = target.get("style", "")
            target["style"] = (existing + ";" + style).strip(";")
        if inner_html and inner_html.strip():
            target.clear()
            frag = BeautifulSoup(inner_html, "html.parser")
            frag_root = frag.body or frag
            for c in list(frag_root.children):
                target.append(c if isinstance(c, Tag) else NavigableString(str(c)))
    return str(soup)

def apply_pre_transforms(html: str, spec: dict) -> str:
    # Minimal structural enforcement before LLM
    out = html
    # Theme on root/body
    theme_class = spec.get("theme", {}).get("designplus_class") or ""
    if theme_class:
        out = enforce_theme_class(out, theme_class, "body")
    # Banner
    b = spec.get("banner", {})
    if b:
        out = transform_banner(
            out,
            selector=b.get("selector") or "#page-banner, .dp-banner, .dpl-banner, .designplus.banner",
            classes=b.get("classes") or "",
            style=b.get("style") or "",
            inner_html=b.get("inner_html"),
            insert_if_missing=bool(b.get("insert_if_missing", True)),
        )
    # Headings
    ty = spec.get("typography", {})
    out = normalize_headings(out, min_h=int(ty.get("min_h", 2)), enforce_hierarchy=bool(ty.get("enforce_hierarchy", True)))
    return out

def apply_post_transforms(html: str, spec: dict) -> str:
    out = html
    # Palette/colors
    pal = spec.get("palette", {})
    out = normalize_palette(
        out,
        allowed=pal.get("allowed", []),
        forbidden=pal.get("forbidden", []),
        map_strategy=pal.get("map_strategy", "nearest"),
    )
    # Theme enforce again
    theme_class = spec.get("theme", {}).get("designplus_class") or ""
    if theme_class:
        out = enforce_theme_class(out, theme_class, "body")
    # Callouts
    comp = spec.get("components", {})
    out = normalize_callouts(out, allowed_variants=comp.get("callouts_allowed", ["info","warning","success"]))
    # Buttons & links
    btn = spec.get("buttons", {})
    out = upgrade_buttons_and_links(out, button_class=btn.get("primary_class", "dpl-button dpl-button--primary"))
    # Tables
    tbl = spec.get("tables", {})
    out = normalize_tables(out, table_class=tbl.get("class", "ic-Table"), zebra=bool(tbl.get("zebra", True)))
    # Images
    img = spec.get("images", {})
    out = ensure_images(out, max_width=img.get("max_width", "100%"), require_alt=bool(img.get("require_alt", True)))
    return out

# ---------------------- UI ----------------------

st.title("Canvas Course-wide HTML Rewrite (Test Instance)")

with st.sidebar:
    st.header("Rewrite Configuration")
    dt_mode = st.selectbox("DesignTools Mode", DT_MODES, index=1)
    user_request = st.text_area(
        "Rewrite goals (your instructions)",
        value="Normalize headings to start at h2; improve accessibility; preserve link destinations; "
              "convert or enhance DesignTools components as needed; keep existing iframes unchanged.",
        height=140,
        help="Describe exactly what to change across the course."
    )

    st.markdown("---")
    st.subheader("Model")
    mode = st.radio("Selection", ["Auto (latest)", "Manual"], horizontal=True)
    if mode == "Auto (latest)":
        # Prefer newest gpt-5 for text; fallback to 4.1/4o as needed
        MODEL_TEXT = latest_model_id(openai_client, r"^(gpt-5|gpt-4\.1|gpt-4o|o\d)", DEFAULT_TEXT_MODEL)
        # Prefer newest gpt-5 or gpt-4o for vision
        MODEL_VISION = latest_model_id(openai_client, r"^(gpt-5|gpt-4o|gpt-4\.1|o\d)", DEFAULT_VISION_MODEL)
    else:
        MODEL_TEXT = st.text_input("Text model id", value=DEFAULT_TEXT_MODEL)
        MODEL_VISION = st.text_input("Vision model id", value=DEFAULT_VISION_MODEL)
    st.caption(f"Using text model: {MODEL_TEXT}")
    st.caption(f"Using vision model: {MODEL_VISION}")

    with st.expander("Advanced (prompt size & retries)"):
        MAX_INPUT_CHARS = st.number_input(
            "Max item HTML chars sent to model",
            min_value=5000, max_value=200000, value=40000, step=5000
        )
        MAX_MODEL_SKELETON_CHARS = st.number_input(
            "Max model skeleton chars",
            min_value=5000, max_value=200000, value=60000, step=5000
        )

    st.markdown("---")
    st.subheader("Spec (deterministic controls)")
    # CSU palette defaults
    default_allowed = ["#1E4D2B", "#C8C372", "#FFFFFF", "#262626"]
    palette_allowed = st.text_input("Allowed colors (comma-separated hex)", value=", ".join(default_allowed))
    palette_forbidden = st.text_input("Forbidden colors (comma-separated hex)", value="")
    map_strategy = st.radio("Color mapping", ["nearest", "exact", "strip"], index=0, horizontal=True)

    theme_choice = st.selectbox("DesignPLUS theme", ["Circle Left 1", "Custom"], index=0)
    theme_class = st.text_input("Theme class to enforce", value="dp-theme--circle-left-1" if theme_choice=="Circle Left 1" else "")

    with st.expander("Banner (precise)"):
        USE_BANNER_SPEC = st.checkbox("Apply banner transform", value=True)
        BANNER_SELECTOR = st.text_input("Banner container selector", value="#page-banner, .dp-banner, .dpl-banner, .designplus.banner")
        BANNER_CLASSES = st.text_input("Banner classes", value="dp-banner dp-banner--lg")
        BANNER_STYLE = st.text_input("Banner inline style", value="min-height: 180px; display:flex; align-items:center;")
        BANNER_INNER_HTML = st.text_area("Banner inner HTML (optional)", value="<h2 class='sr-only'>Course Banner</h2>", height=80)
        INSERT_IF_MISSING = st.checkbox("Insert banner at top if missing", value=True)

    with st.expander("Components & Typography"):
        callouts_allowed = st.multiselect("Callout variants allowed", ["info","warning","success","note"], default=["info","warning","success"])
        tabs_policy = st.radio("Tabs", ["forbid","allow","auto"], index=2, horizontal=True)
        accordions_policy = st.radio("Accordions", ["forbid","allow","auto"], index=2, horizontal=True)
        min_h = st.number_input("Minimum heading level", min_value=2, max_value=4, value=2, step=1)
        enforce_hierarchy = st.checkbox("Enforce heading hierarchy (no jumps)", value=True)

    with st.expander("Buttons, Tables, Images"):
        btn_primary = st.text_input("Button primary class", value="dpl-button dpl-button--primary")
        tbl_class = st.text_input("Table class", value="ic-Table")
        tbl_zebra = st.checkbox("Zebra striping", value=True)
        img_max_w = st.text_input("Image max width", value="100%")
        img_require_alt = st.checkbox("Require alt text", value=True)

    # Build the spec dict to pass to transforms & LLM hard rules
    SPEC = {
        "theme": {"designplus_class": theme_class.strip()},
        "palette": {
            "allowed": [c.strip() for c in palette_allowed.split(",") if c.strip()],
            "forbidden": [c.strip() for c in palette_forbidden.split(",") if c.strip()],
            "map_strategy": map_strategy,
        },
        "components": {
            "callouts_allowed": callouts_allowed,
            "tabs": tabs_policy,
            "accordions": accordions_policy,
        },
        "typography": {"min_h": int(min_h), "enforce_hierarchy": bool(enforce_hierarchy)},
        "buttons": {"primary_class": btn_primary.strip()},
        "tables": {"class": tbl_class.strip(), "zebra": bool(tbl_zebra)},
        "images": {"max_width": img_max_w.strip(), "require_alt": bool(img_require_alt)},
        "banner": {
            "selector": BANNER_SELECTOR if USE_BANNER_SPEC else "",
            "classes": BANNER_CLASSES if USE_BANNER_SPEC else "",
            "style": BANNER_STYLE if USE_BANNER_SPEC else "",
            "inner_html": BANNER_INNER_HTML if (USE_BANNER_SPEC and BANNER_INNER_HTML.strip()) else "",
            "insert_if_missing": bool(INSERT_IF_MISSING) if USE_BANNER_SPEC else False,
        }
    }

    st.markdown("---")
    st.subheader("Preview options")
    PREVIEW_HEIGHT = st.slider("Preview height (px)", 320, 1200, 520, 20)
    PREVIEW_JS = st.checkbox("Enable shim JS (tabs interactions)", value=True)
    st.caption("Previews use a built-in DesignPLUS-like shim. No external CSS/JS required.")

    st.markdown("---")
    st.subheader("Model Reference (optional)")
    ref_kind = st.radio("Type", ["None", "Paste HTML", "Upload Image", "Model Course"], horizontal=True)

    model_html_skeleton = None
    model_image_data_url = None

    if ref_kind == "Paste HTML":
        pasted_html = st.text_area(
            "Paste model HTML here",
            height=200,
            help="Paste the HTML of a model Canvas page. We will derive a structure skeleton from it."
        )
        if pasted_html.strip():
            try:
                model_html_skeleton = html_to_skeleton(pasted_html)
                if len(model_html_skeleton) > MAX_MODEL_SKELETON_CHARS:
                    model_html_skeleton = model_html_skeleton[:MAX_MODEL_SKELETON_CHARS] + " <!-- truncated -->"
                st.caption(f"Model skeleton length: {len(model_html_skeleton)} chars")
                with st.expander("Preview model HTML skeleton"):
                    st.code(model_html_skeleton[:4000])
            except Exception as e:
                st.error(f"Failed to process pasted HTML: {e}")

    elif ref_kind == "Upload Image":
        uploaded_img = st.file_uploader("Upload model page image", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=False)
        if uploaded_img is not None:
            try:
                model_image_data_url, mime = image_to_data_url(uploaded_img)
                st.image(uploaded_img, caption=f"Model reference ({mime})", use_container_width=True)
                st.caption("The image will be passed to the model as a vision reference. If unsupported, we will fall back to text-only.")
            except Exception as e:
                st.error(f"Failed to process image: {e}")

    elif ref_kind == "Model Course":
        st.caption("Use a Page or Assignment from another Canvas course as the model.")
        mc_code = st.text_input("Model course code", key="model_course_code")
        if st.button("Find model course"):
            with st.spinner("Searching model course…"):
                st.session_state["model_courses"] = find_single_course_by_code(account, mc_code)

        model_courses = st.session_state.get("model_courses", [])
        if model_courses:
            m_idx = st.selectbox(
                "Select model course",
                options=list(range(len(model_courses))),
                format_func=lambda i: f"{model_courses[i].id} · {getattr(model_courses[i], 'course_code', '')} · {getattr(model_courses[i], 'name', '')}",
                key="model_course_idx"
            )
            model_course = model_courses[m_idx]

            kind = st.radio("Model item type", ["Page", "Assignment"], horizontal=True, key="model_item_kind")

            if kind == "Page":
                if "model_pages" not in st.session_state:
                    st.session_state["model_pages"] = list_all_pages(model_course)
                if st.button("Refresh model pages"):
                    st.session_state["model_pages"] = list_all_pages(model_course)
                pages = st.session_state.get("model_pages", [])
                if pages:
                    page_map = {f"{t}": slug for (t, slug) in pages}
                    sel_title = st.selectbox("Choose model page", options=list(page_map.keys()))
                    if sel_title:
                        try:
                            raw = fetch_model_item_html(model_course, "Page", page_map[sel_title])
                            model_html_skeleton = html_to_skeleton(raw)
                            if len(model_html_skeleton) > MAX_MODEL_SKELETON_CHARS:
                                model_html_skeleton = model_html_skeleton[:MAX_MODEL_SKELETON_CHARS] + " <!-- truncated -->"
                            st.caption(f"Model skeleton length: {len(model_html_skeleton)} chars")
                            with st.expander("Preview model HTML skeleton"):
                                st.code(model_html_skeleton[:4000])
                        except Exception as e:
                            st.error(f"Failed to fetch model page: {e}")

            else:  # Assignment
                if "model_asgs" not in st.session_state:
                    st.session_state["model_asgs"] = list_all_assignments(model_course)
                if st.button("Refresh model assignments"):
                    st.session_state["model_asgs"] = list_all_assignments(model_course)
                asgs = st.session_state.get("model_asgs", [])
                if asgs:
                    asg_map = {f"{name} (#{aid})": aid for (name, aid) in asgs}
                    sel_name = st.selectbox("Choose model assignment", options=list(asg_map.keys()))
                    if sel_name:
                        try:
                            raw = fetch_model_item_html(model_course, "Assignment", asg_map[sel_name])
                            model_html_skeleton = html_to_skeleton(raw)
                            if len(model_html_skeleton) > MAX_MODEL_SKELETON_CHARS:
                                model_html_skeleton = model_html_skeleton[:MAX_MODEL_SKELETON_CHARS] + " <!-- truncated -->"
                            st.caption(f"Model skeleton length: {len(model_html_skeleton)} chars")
                            with st.expander("Preview model HTML skeleton"):
                                st.code(model_html_skeleton[:4000])
                        except Exception as e:
                            st.error(f"Failed to fetch model assignment: {e}")

st.subheader("1) Pick Course by Code")
course_code = st.text_input("Course code", help="Exact course code preferred; we'll disambiguate if needed.")
if st.button("Search course"):
    with st.spinner("Searching…"):
        st.session_state["courses"] = find_course_by_code(account, course_code)

courses = st.session_state.get("courses", [])
if courses:
    idx = st.selectbox(
        "Select course",
        options=list(range(len(courses))),
        format_func=lambda i: f"{courses[i].id} · {getattr(courses[i], 'course_code', '')} · {getattr(courses[i], 'name', '')}",
    )
    course = courses[idx]

    st.subheader("2) Dry-run")
    if st.button("Collect items & simulate rewrite", disabled=(not user_request.strip())):
        if not user_request.strip():
            st.warning("Please enter rewrite goals first.")
        else:
            st.session_state["items"] = []
            st.session_state["drafts"] = {}
            with st.spinner("Fetching items…"):
                module_items = list_supported_items(course)

            progress = st.progress(0.0)
            eta_box = st.empty()
            total = max(len(module_items), 1)
            start_time = time.time()
            recent_durations = []
            WINDOW = 5

            for n, (module, it) in enumerate(module_items, start=1):
                item_t0 = time.time()
                meta = fetch_item_html(course, it)
                original = meta["html"] or ""
                frozen, mapping, hosts = protect_iframes(original)

                # PRE transforms (deterministic) BEFORE LLM
                prepped = apply_pre_transforms(frozen, SPEC)

                # Trim oversized HTML for prompt resilience
                if len(prepped) > MAX_INPUT_CHARS:
                    prepped = prepped[:MAX_INPUT_CHARS] + "\n<!-- truncated for prompt -->"

                # Call OpenAI with chosen models, model reference & hard rules (SPEC)
                try:
                    rewritten = openai_rewrite(
                        user_request=user_request,
                        html=prepped,
                        dt_mode=dt_mode,
                        policy_hard_rules=SPEC,
                        model_html_skeleton=model_html_skeleton,
                        model_image_data_url=model_image_data_url,
                        model_text_id=MODEL_TEXT,
                        model_vision_id=MODEL_VISION,
                    )
                except Exception as e:
                    st.error(f"Rewrite failed for [{meta.get('title') or meta.get('url')}] — {e}")
                    progress.progress(n / total)
                    # Keep an entry (original as draft) so user can still preview/decide
                    key = f"{meta['kind']}:{meta['id']}"
                    st.session_state["items"].append({
                        "key": key,
                        "title": meta.get("title") or meta.get("url"),
                        "kind": meta["kind"],
                        "module": getattr(module, "name", ""),
                        "item": it,
                        "original": original,
                        "draft": original,
                    })
                    st.session_state["drafts"][key] = {"diff": html_diff(original, original)}
                    # ETA update
                    duration = time.time() - item_t0
                    recent_durations.append(duration)
                    if len(recent_durations) > WINDOW:
                        recent_durations.pop(0)
                    avg_item = (sum(recent_durations) / len(recent_durations)) if recent_durations else max(0.1, (time.time() - start_time) / n)
                    remaining = total - n
                    eta_sec = remaining * avg_item
                    elapsed = time.time() - start_time
                    eta_box.info(
                        f"Processed {n}/{total} items · Elapsed {_format_duration(elapsed)} · "
                        f"Avg/Item {avg_item:.1f}s · ETA {_format_duration(eta_sec)}"
                    )
                    continue

                # POST transforms (deterministic) AFTER LLM
                rewritten_no_new_iframes = strip_new_iframes(rewritten)
                restored = restore_iframes(rewritten_no_new_iframes, mapping)
                final_html = apply_post_transforms(restored, SPEC)

                diff_html = html_diff(original, final_html)
                key = f"{meta['kind']}:{meta['id']}"
                st.session_state["items"].append({
                    "key": key,
                    "title": meta.get("title") or meta.get("url"),
                    "kind": meta["kind"],
                    "module": getattr(module, "name", ""),
                    "item": it,
                    "original": original,
                    "draft": final_html,
                })
                st.session_state["drafts"][key] = {"diff": diff_html}

                # --- update ETA/progress UI ---
                duration = time.time() - item_t0
                recent_durations.append(duration)
                if len(recent_durations) > WINDOW:
                    recent_durations.pop(0)
                avg_item = (sum(recent_durations) / len(recent_durations)) if recent_durations else max(0.1, (time.time() - start_time) / n)
                remaining = total - n
                eta_sec = remaining * avg_item
                elapsed = time.time() - start_time
                eta_box.info(
                    f"Processed {n}/{total} items · Elapsed {_format_duration(elapsed)} · "
                    f"Avg/Item {avg_item:.1f}s · ETA {_format_duration(eta_sec)}"
                )
                progress.progress(n / total)

            st.success(f"Prepared {len(st.session_state['items'])} items.")

    items = st.session_state.get("items", [])
    if items:
        st.subheader("3) Review visual previews, diffs & approve")

        # Bulk approval controls
        col1, col2, col3 = st.columns([1,1,3])
        with col1:
            if st.button("Approve All"):
                for rec in items:
                    cb_key = f"approve_{rec['key']}"
                    st.session_state[cb_key] = True
        with col2:
            if st.button("Unapprove All"):
                for rec in items:
                    cb_key = f"approve_{rec['key']}"
                    st.session_state[cb_key] = False
        with col3:
            approved_count = sum(1 for rec in items if st.session_state.get(f"approve_{rec['key']}", False))
            st.write(f"Approved: **{approved_count} / {len(items)}**")

        # Render each item
        for _, rec in enumerate(items):
            cb_key = f"approve_{rec['key']}"
            with st.expander(f"[{rec['kind']}] {rec['title']} — {rec['module']}"):
                # Visual previews (Original vs Proposed) using the Shim
                left, right = st.columns(2)
                with left:
                    st.markdown("**Original (visual)**")
                    original_preview = neutralize_iframes_for_preview(rec["original"])
                    original_doc = preview_html_document(original_preview, use_js_shim=PREVIEW_JS)
                    st.components.v1.html(original_doc, height=PREVIEW_HEIGHT, scrolling=True)
                with right:
                    st.markdown("**Proposed (visual)**")
                    proposed_preview = neutralize_iframes_for_preview(rec["draft"])
                    proposed_doc = preview_html_document(proposed_preview, use_js_shim=PREVIEW_JS)
                    st.components.v1.html(proposed_doc, height=PREVIEW_HEIGHT, scrolling=True)

                st.markdown("**Diff (proposed vs original)**  \n_Green = insertions, Red = deletions_", help="Generated by diff-match-patch")
                st.components.v1.html(st.session_state["drafts"][rec["key"]]["diff"], height=240, scrolling=True)

                approved = st.checkbox("Approve this item", key=cb_key, value=st.session_state.get(cb_key, False))
                rec["approved"] = bool(approved)

                with st.expander("Show HTML (original)"):
                    st.code(rec["original"][:2000])
                with st.expander("Show HTML (proposed)"):
                    st.code(rec["draft"][:2000])

        st.write("")
        if st.button("Apply approved changes"):
            applied, failed = 0, 0
            with st.spinner("Applying to Canvas…"):
                for rec in items:
                    cb_key = f"approve_{rec['key']}"
                    if not st.session_state.get(cb_key, False):
                        continue
                    try:
                        apply_update(course, rec["item"], rec["draft"])
                        applied += 1
                    except Exception as e:
                        failed += 1
                        st.error(f"Failed: {rec['title']}: {e}")
            st.success(f"Applied {applied} item(s); {failed} failed.")
else:
    st.info("Search for a course above to begin.")

# streamlit_app.py
# Canvas Course-wide HTML Rewrite (Test Instance)
# - Admin-only Streamlit app for dry-run + bulk apply on Pages & Assignment descriptions
# - Canvas API (test instance), OpenAI Responses API (auto-pick newest model + retries)
# - Iframe freeze/restore (no new iframes), NO sanitizer (write model HTML as-is)
# - Visual PREVIEW (Original vs Proposed) using a built-in DesignPLUS-like "Shim" (CSS+JS)
# - ETA during dry-run
# - Approve All / Unapprove All bulk toggles
# - Precision Directives (micro-language) + few-shot examples for higher precision; no giant sidebar

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
    Build a compact, valid-ish skeleton of the model HTML.
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
    courses = account.get_courses(search_term=course_code)
    return list(courses)

def list_all_pages(course):
    pages = []
    for p in course.get_pages():
        pages.append((getattr(p, "title", p.url), p.url))
    return pages

def list_all_assignments(course):
    items = []
    for a in course.get_assignments():
        items.append((getattr(a, "name", f"Assignment {a.id}"), a.id))
    return items

def fetch_model_item_html(course, kind: str, ident):
    if kind == "Page":
        page = course.get_page(ident)  # ident = page.url
        return page.body or ""
    elif kind == "Assignment":
        asg = course.get_assignment(int(ident))
        return asg.description or ""
    return ""

def image_to_data_url(file) -> Tuple[str, str]:
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
    try:
        res = client.models.list()
        data = getattr(res, "data", res)
        return list(data)
    except Exception:
        return []

def latest_model_id(client: OpenAI, pattern: str, default_id: str) -> str:
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
    "Placeholders like ⟪IFRAME:n⟫ represent protected iframes—do not add or remove them. "
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

# Few-shot examples to increase determinism without UI sliders
EXAMPLES_TEXT = """
EXAMPLE 1 — Banner normalize
INPUT:
<div id="kl_banner"><h2><span id="kl_banner_left"><span class="kl_mod_text">Module 1</span></span> Week 1 <span id="kl_banner_right">Overview</span></h2></div>
RULES:
- HEADER-FROM: #kl_banner h2 -> <header class="dp-header"><h2 class="dp-heading"><span class="dp-header-pre"><span class="dp-header-pre-1">{MODULE-NUMBER}</span>&nbsp;</span> <span class="dp-header-title">{TITLE}</span></h2></header>
- MODULE-NUMBER: regex "Module\s+(\d+)" -> "Module \1"
- TITLE: "Overview and Readings"
EXPECTED:
<header class="dp-header"><h2 class="dp-heading"><span class="dp-header-pre"><span class="dp-header-pre-1">Module 1</span>&nbsp;</span> <span class="dp-header-title">Overview and Readings</span></h2></header>


EXAMPLE 2 — Image to figure
INPUT:
<h3><img src="X/preview" alt="Confucius, Laozi, and the Buddha" width="235" height="505" /></h3>
<p style="text-align:center;"><em>(Confucius Lao-tzu and Buddhist Arhat…)</em></p>
RULES:
- FIGURE-FROM-IMG with caption-from-next-p
EXPECTED:
<div class="dp-content-block" style="text-align: center;">
  <figure class="dp-image-rounded-10 dp-image-padded dp-image-bordered dp-image-shadow" style="width:235px;">
    <img class="dp-popup-image" src="X/preview" alt="Confucius, Laozi, and the Buddha" width="235" height="505" />
    <figcaption>Confucius Lao-tzu and Buddhist Arhat…</figcaption>
  </figure>
</div>

Example 3 - Dates to Callout
INPUT:
<h3><span style="color:#e62429;">Our weeks typically run Tuesdays…</span></h3>
RULES:
- CALLOUT: detect #e62429 or schedule verbiage; title "Important Dates"; strip inline colors
EXPECTED:
<div class="dp-callout dp-callout-placeholder card dp-callout-position-default dp-callout-type-default dp-callout-color-dp-primary">
  <div class="card-body">
    <h3 class="card-title">Important Dates</h3>
    <p class="card-text">Our weeks typically run Tuesdays…</p>
  </div>
</div>


"""

def openai_rewrite(
    user_request: str,
    html: str,
    dt_mode: str,
    precision_directives: str,
    model_html_skeleton: Optional[str] = None,
    model_image_data_url: Optional[str] = None,
    model_text_id: str = DEFAULT_TEXT_MODEL,
    model_vision_id: str = DEFAULT_VISION_MODEL,
) -> str:
    """
    Call OpenAI Responses API to rewrite the HTML with precision directives and few-shot examples.
    """
    policy = {
        "design_tools_mode": dt_mode,
        "allow_inline_styles": True,
        "block_new_iframes": True,
        "reference_model": ("image" if model_image_data_url else "html" if model_html_skeleton else "none"),
        "reference_usage": "Match layout/sectioning/components; do not copy literal course-specific links or text."
    }

    hard_rules_text = (
        "Hard rules (follow EXACTLY; reason silently; output HTML only):\n"
        + (precision_directives.strip() + "\n" if precision_directives.strip() else "")
        + "- Do not remove existing anchors/IDs/classes/data-* unless conflicting with required theme/classes.\n"
        + "- Do not create new iframes. Respect ⟪IFRAME:n⟫ placeholders.\n"
    )

    blocks = [
        hard_rules_text,
        "Policy: " + json.dumps(policy, ensure_ascii=False),
        "Examples:\n" + EXAMPLES_TEXT.strip(),
        f"DesignTools Mode: {dt_mode}",
        "Rewrite goals (optional): " + (user_request or ""),
    ]
    if model_html_skeleton:
        blocks.append("Model HTML skeleton (follow structure/classes, not text):\n" + model_html_skeleton)
    blocks.append("HTML to rewrite follows:\n" + html)

    if not model_image_data_url:
        prompt = "\n\n".join(blocks)
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
                {"type": "input_text", "text": "\n\n".join(blocks[:-1])},
                {"type": "input_image", "image_url": {"url": model_image_data_url}},
                {"type": "input_text", "text": blocks[-1]},
            ]
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
            prompt = "\n\n".join(blocks) + "\n\n(Note: image reference unavailable; proceed using textual model description if any.)"
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
        return resp.output_text
    except AttributeError:
        return resp.output[0].content[0].text

# ---------------------- Diff ----------------------

def html_diff(old: str, new: str) -> str:
    dmp = diff_match_patch()
    d = dmp.diff_main(old or "", new or "")
    dmp.diff_cleanupSemantic(d)
    return dmp.diff_prettyHtml(d)

# ---------------------- Shim CSS/JS & preview ----------------------

SHIM_CSS = """
:root {
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Lato, Arial, sans-serif;
  --csu-green: #1E4D2B;
  --csu-gold: #C8C372;
  --neutral-900: #262626;
  --border: #e2e8f0;
}
html, body { margin:0; padding:0; font-family:var(--font); line-height:1.5; background:#fff; color:#111; }
#frame { max-width: 980px; margin: 0 auto; padding: 24px; background: #fff; }
h1,h2,h3,h4,h5,h6 { margin: 0.6em 0 0.4em; line-height:1.25; }
p,li { font-size: 16px; }
a { color: #005f85; text-decoration: underline; }
img { max-width: 100%; height: auto; }

/* Banner */
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

/* Accordions */
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
    """Replace iframes with placeholders (no external loads in preview)."""
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup.find_all("iframe"):
        src = (tag.get("src") or "").strip()
        host = urllib.parse.urlparse(src).hostname or "iframe"
        placeholder = soup.new_tag("div", attrs={"class": "iframe-placeholder"})
        placeholder.string = f"[iframe: {host}]"
        tag.replace_with(placeholder)
    return str(soup)

def preview_html_document(inner_html: str, use_js_shim: bool) -> str:
    """Wrap content in a minimal, sandboxed page for st.components.v1.html."""
    safe_inner = inner_html or ""
    return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>{SHIM_CSS}</style>
  </head>
  <body>
    <main id="frame">
      {safe_inner}
    </main>
    {"<script>"+SHIM_JS+"</script>" if use_js_shim else ""}
  </body>
</html>"""

# ---------------------- UI ----------------------

st.title("Canvas Course-wide HTML Rewrite (Test Instance)")

with st.sidebar:
    st.header("Rewrite Setup")

    dt_mode = st.selectbox("DesignTools Mode", DT_MODES, index=1, help="How aggressively to use/change DesignTools patterns.")
    user_request = st.text_area(
        "Rewrite goals (optional, in plain English)",
        value="Improve accessibility; normalize headings; refine layout using DesignPLUS where appropriate; preserve links and existing iframes.",
        height=120,
    )

    st.markdown("**Precision directives (optional, power users)**")
    precision_directives = st.text_area(
        label="",
        value=(
            "PRESERVE: anchors, ids, classes, data-*\n"
            "NO-NEW: iframes\n"
            "ENFORCE-CLASS: body -> dp-theme--flat-sections variation-2\n"
            "BANNER: selector=\"#page-banner,.dp-banner\" classes=\"dp-banner dp-banner--lg\" style=\"min-height:180px;display:flex;align-items:center;\" insert-if-missing=true\n"
            "PALETTE-ALLOWED: #1E4D2B, #D9782D, #C9D845, #CC5430, #105456, #FFFFFF, #262626\n"
            "PALETTE-MAP: nearest\n"
            "CALLOUTS-ALLOWED: info, warning, success\n"
            "TABS: allow\n"
            "ACCORDIONS: auto\n"
            "HEADINGS: min=2, enforce-hierarchy=true\n"
            "BUTTON-CLASS: dpl-button dpl-button--primary\n"
            "TABLE: class=ic-Table, zebra=true\n"
            "IMAGES: max-width=100%, alt=require\n"
            "DO-NOT-EDIT: \"#course-links, .lti-embed, .assignment-group\""
        ),
        height=170,
        help="Short command-style rules. Leave as-is or tweak."
    )

    with st.expander("Preview options"):
        PREVIEW_HEIGHT = st.slider("Preview height (px)", 320, 1200, 520, 20)
        PREVIEW_JS = st.checkbox("Enable shim JS (tabs interactions)", value=True)

    st.markdown("---")
    st.subheader("Model")
    mode = st.radio("Model selection", ["Auto (latest)", "Manual"], horizontal=True)
    if mode == "Auto (latest)":
        MODEL_TEXT = latest_model_id(openai_client, r"^(gpt-5|gpt-4\.1|gpt-4o|o\d)", DEFAULT_TEXT_MODEL)
        MODEL_VISION = latest_model_id(openai_client, r"^(gpt-5|gpt-4o|gpt-4\.1|o\d)", DEFAULT_VISION_MODEL)
    else:
        MODEL_TEXT = st.text_input("Text model id", value=DEFAULT_TEXT_MODEL)
        MODEL_VISION = st.text_input("Vision model id", value=DEFAULT_VISION_MODEL)
    st.caption(f"Using text model: {MODEL_TEXT}")
    st.caption(f"Using vision model: {MODEL_VISION}")

    with st.expander("Advanced"):
        MAX_INPUT_CHARS = st.number_input("Max item HTML chars sent to model", 5000, 200000, 40000, 5000)
        MAX_MODEL_SKELETON_CHARS = st.number_input("Max model skeleton chars", 5000, 200000, 60000, 5000)

    st.markdown("---")
    st.subheader("Model Reference (optional)")
    ref_kind = st.radio("Reference type", ["None", "Paste HTML", "Upload Image", "Model Course"], horizontal=True)

    model_html_skeleton = None
    model_image_data_url = None

    if ref_kind == "Paste HTML":
        pasted_html = st.text_area("Paste model HTML here", height=200)
        if pasted_html.strip():
            try:
                model_html_skeleton = html_to_skeleton(pasted_html)
                if len(model_html_skeleton) > MAX_MODEL_SKELETON_CHARS:
                    model_html_skeleton = model_html_skeleton[:MAX_MODEL_SKELETON_CHARS] + " <!-- truncated -->"
                st.caption(f"Model skeleton length: {len(model_html_skeleton)} chars")
                with st.expander("Preview model skeleton"):
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
                            with st.expander("Preview model skeleton"):
                                st.code(model_html_skeleton[:4000])
                        except Exception as e:
                            st.error(f"Failed to fetch model page: {e}")

            else:
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
                            with st.expander("Preview model skeleton"):
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
    if st.button("Collect items & simulate rewrite"):
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

            # Trim oversized HTML for prompt resilience
            prepped = frozen
            if len(prepped) > MAX_INPUT_CHARS:
                prepped = prepped[:MAX_INPUT_CHARS] + "\n<!-- truncated for prompt -->"

            # Call OpenAI with precision directives + examples
            try:
                rewritten = openai_rewrite(
                    user_request=user_request,
                    html=prepped,
                    dt_mode=dt_mode,
                    precision_directives=precision_directives,
                    model_html_skeleton=model_html_skeleton,
                    model_image_data_url=model_image_data_url,
                    model_text_id=MODEL_TEXT,
                    model_vision_id=MODEL_VISION,
                )
            except Exception as e:
                st.error(f"Rewrite failed for [{meta.get('title') or meta.get('url')}] — {e}")
                progress.progress(n / total)
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

            # Strip any new iframes, then restore originals
            rewritten_no_new_iframes = strip_new_iframes(rewritten)
            final_html = restore_iframes(rewritten_no_new_iframes, mapping)

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

        # Bulk toggles
        col1, col2, col3 = st.columns([1,1,3])
        with col1:
            if st.button("Approve All"):
                for rec in items:
                    st.session_state[f"approve_{rec['key']}"] = True
        with col2:
            if st.button("Unapprove All"):
                for rec in items:
                    st.session_state[f"approve_{rec['key']}"] = False
        with col3:
            approved_count = sum(1 for rec in items if st.session_state.get(f"approve_{rec['key']}", False))
            st.write(f"Approved: **{approved_count} / {len(items)}**")

        # Render items
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

                st.markdown("**Diff (proposed vs original)**  \n_Green = insertions, Red = deletions_")
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

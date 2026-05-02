"""
CogniFlow — minimal Streamlit chat UI for the FastAPI RAG backend.
"""

from __future__ import annotations

import html
import json
import os
import time
import uuid
from typing import Any

import requests
import streamlit as st

DEFAULT_API = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
# Pause between each chunk in st.write_stream (higher = slower typing effect).
_UI_STREAM_DELAY_SEC = float(os.environ.get("STREAMLIT_STREAM_DELAY_SEC", "0.01"))

# Single brand teal for Streamlit UI (no multi-shade accent ramp).
_TEAL = "#14b8a6"

# --- Black + single-teal theme (Streamlit UI) ----------------------------------
_CSS = """
<style>
html, body, [class*="css"] { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif !important; }
.stApp {
    background: linear-gradient(180deg, #ffffff 0%, #f7fffd 45%, #ffffff 100%) !important;
    color: #0f172a !important;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f7fffd 100%) !important;
    border-right: 1px solid rgba(20, 184, 166, 0.22) !important;
    overflow-x: hidden !important;
    isolation: isolate !important;
}
section[data-testid="stSidebar"] label { color: #a8a29e !important; }
section[data-testid="stSidebar"] .stTextInput input {
    background: #ffffff !important; color: #0f172a !important;
    border: 1px solid rgba(20, 184, 166, 0.35) !important; border-radius: 8px !important;
}
.main .block-container {
    padding-top: 1.25rem !important;
    max-width: 56rem !important;
    background: transparent !important;
}
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main,
.main {
    background: #ffffff !important;
    background-color: #ffffff !important;
}
[data-testid="stMain"] > div,
[data-testid="stMain"] [data-testid="stVerticalBlock"] {
    background: transparent !important;
}
header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu { visibility: hidden; height: 0; }
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
/* Hero */
.cf-main-hero { text-align: center; margin-bottom: 0.5rem; }
.cf-main-hero h1 {
    font-size: 1.35rem; font-weight: 700; margin: 0; letter-spacing: -0.02em;
    color: #14b8a6 !important;
    background: none !important;
    -webkit-text-fill-color: #14b8a6 !important;
}
.cf-main-hero p { margin: 0.25rem 0 0; font-size: 0.8rem; color: #78716c; }
/* Status */
.cf-status { display: flex; align-items: center; gap: 0.4rem; font-size: 0.8rem; color: #a8a29e; margin: 0.35rem 0 0.75rem; }
.cf-status-dot { width: 7px; height: 7px; border-radius: 50%; background: #14b8a6; box-shadow: 0 0 12px rgba(20, 184, 166, 0.65); }
.cf-status-dot.off { background: #f87171; box-shadow: 0 0 10px rgba(248, 113, 113, 0.45); }
.cf-sec-label {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em; color: #14b8a6;
    opacity: 0.85;
    margin: 0.75rem 0 0.5rem; text-transform: uppercase;
}
/* Chat upload card */
.cf-upload-card {
    margin: 0.15rem 0 0.6rem;
    padding: 0.85rem 0.95rem;
    border-radius: 12px;
    border: 1px solid rgba(20, 184, 166, 0.34);
    background: linear-gradient(180deg, rgba(20, 184, 166, 0.12) 0%, #f7fffd 100%);
}
.cf-upload-title {
    margin: 0;
    color: #14b8a6;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.11em;
    text-transform: uppercase;
}
.cf-upload-sub {
    margin: 0.35rem 0 0;
    color: #a8a29e;
    font-size: 0.82rem;
}
.cf-upload-sub b { color: #0f172a; }
.cf-upload-count {
    margin-top: 0.45rem;
    font-size: 0.75rem;
    color: #14b8a6;
}
/* Sidebar buttons */
section[data-testid="stSidebar"] button[kind="secondary"] {
    background: #ffffff !important;
    background-image: none !important;
    color: #0f172a !important;
    border: 1px solid rgba(20, 184, 166, 0.25) !important;
    border-radius: 10px !important;
}
section[data-testid="stSidebar"] button[kind="secondary"]:hover {
    border-color: #14b8a6 !important;
    background: #ecfdf9 !important;
    color: #0f172a !important;
    box-shadow: 0 0 0 1px rgba(20, 184, 166, 0.15) !important;
}
/* Primary — single brand teal */
section[data-testid="stSidebar"] button[kind="primary"] {
    background: #14b8a6 !important;
    background-image: none !important;
    border: none !important; color: #000000 !important; font-weight: 600 !important;
    border-radius: 10px !important; padding: 0.65rem 1rem !important;
    box-shadow: 0 4px 20px rgba(20, 184, 166, 0.4) !important;
}
section[data-testid="stSidebar"] button[kind="primary"]:hover {
    filter: brightness(0.92);
    box-shadow: 0 6px 24px rgba(20, 184, 166, 0.45) !important;
}
section[data-testid="stSidebar"] button[kind="tertiary"] {
    background: transparent !important;
    background-image: none !important;
    color: #78716c !important;
    border: none !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] button[kind="tertiary"]:hover {
    color: #f87171 !important;
    background: rgba(248, 113, 113, 0.08) !important;
}
section[data-testid="stSidebar"] div[data-testid="column"]:first-child button[kind="secondary"] {
    text-align: left !important;
    min-height: 3.25rem !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    line-height: 1.35 !important;
    white-space: pre-line !important;
}
section[data-testid="stSidebar"] div[data-testid="column"]:last-child button[kind="tertiary"] {
    min-height: 3.25rem !important;
    width: 100% !important;
    max-width: 2.75rem !important;
    padding: 0 0.25rem !important;
    font-size: 1.1rem !important;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border-color: rgba(20, 184, 166, 0.35) !important;
}
div[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    border: 2px dashed rgba(20, 184, 166, 0.35) !important; border-radius: 12px !important;
    background: #f8fffd !important; padding: 0.5rem !important;
}
div[data-testid="stSidebar"] [data-testid="stFileUploader"] small { color: #a8a29e !important; }
/* Chunk stat */
.cf-chunk-stat {
    margin-top: 0.75rem; padding: 0.85rem 1rem; border-radius: 12px;
    background: rgba(20, 184, 166, 0.1);
    border: 1px solid rgba(20, 184, 166, 0.35); text-align: center;
}
.cf-chunk-stat .n { font-size: 1.5rem; font-weight: 700; color: #14b8a6; line-height: 1.2; }
.cf-chunk-stat .lbl { font-size: 0.7rem; color: #a8a29e; margin-top: 0.2rem; letter-spacing: 0.04em; }
/* Chat bubbles */
[data-testid="stChatMessage"] {
    background: #ffffff !important;
    border: 1px solid rgba(20, 184, 166, 0.2) !important; border-radius: 14px !important;
    padding: 0.75rem 1rem !important; margin-bottom: 0.65rem !important;
}
[data-testid="stChatMessage"] p { color: #0f172a !important; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] a { color: #14b8a6 !important; }
/* Source pills */
.cf-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; align-items: center; }
.cf-pill {
    display: inline-block; padding: 4px 11px; border-radius: 999px; font-size: 0.78rem;
    background: rgba(20, 184, 166, 0.15); color: #14b8a6; border: 1px solid rgba(20, 184, 166, 0.4);
    font-weight: 500;
}
.cf-latency { font-size: 0.78rem; color: #78716c; margin-top: 6px; }
.streamlit-expanderHeader { font-size: 0.85rem !important; color: #a8a29e !important; }
[data-testid="stExpander"] details {
    border: 1px solid rgba(20, 184, 166, 0.22) !important; border-radius: 10px !important; background: #ffffff !important;
}
/* Main column: strip default white boxes (upload / Index button / chat dock) */
section.main [data-testid="stAppViewContainer"],
section.main [data-testid="stMainBlockContainer"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
}
section.main [data-testid="block-container"] {
    background: transparent !important;
    background-color: transparent !important;
}
section.main [data-testid="stElementContainer"] {
    background: transparent !important;
    background-color: transparent !important;
}
section.main [data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid rgba(20, 184, 166, 0.2) !important;
    border-radius: 14px !important;
}
section.main [data-testid="stFileUploader"] {
    background: transparent !important;
}
section.main [data-testid="stFileUploader"] section,
section.main [data-testid="stFileUploaderDropzone"] {
    background: #f8fffd !important;
    background-color: #f8fffd !important;
    border: 2px dashed rgba(20, 184, 166, 0.35) !important;
    border-radius: 12px !important;
}
section.main [data-testid="stFileUploader"] small,
section.main [data-testid="stFileUploader"] label {
    color: #a8a29e !important;
}
section.main [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
section.main [data-testid="stFileUploader"] li {
    background: rgba(20, 184, 166, 0.07) !important;
    background-color: rgba(20, 184, 166, 0.07) !important;
    border: 1px solid rgba(20, 184, 166, 0.28) !important;
    border-radius: 10px !important;
    color: #0f172a !important;
}
section.main button[kind="secondary"] {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid rgba(20, 184, 166, 0.4) !important;
}
/* Chat input dock */
.stChatFloatingInputContainer {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border-top: none !important;
    padding: 0.6rem 1rem 0.9rem !important;
    box-shadow: none !important;
}
.stChatFloatingInputContainer > div {
    background: #ffffff !important;
    background-color: #ffffff !important;
    max-width: 960px !important;
    margin: 0 auto !important;
}
[data-testid="stChatInput"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border: none !important;
}
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div,
[data-testid="stChatInput"] fieldset {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border: 1px solid #dbe4ea !important;
    border-radius: 28px !important;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.08) !important;
    padding: 0.28rem 0.35rem !important;
}
[data-testid="stChatInput"] textarea {
    background: #ffffff !important;
    border: none !important;
    color: #0f172a !important;
    border-radius: 20px !important;
    min-height: 2.6rem !important;
    padding-left: 0.35rem !important;
}
[data-testid="stChatInputSubmitButton"] {
    background: #111827 !important;
    color: #ffffff !important;
    border: 1px solid #111827 !important;
    border-radius: 999px !important;
    width: 2rem !important;
    height: 2rem !important;
    min-width: 2rem !important;
}
[data-testid="stChatInputSubmitButton"]:hover {
    filter: brightness(1.06);
}
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
}
[data-testid="stBottomBlockContainer"] > div,
[data-testid="stBottom"] > div {
    background: #ffffff !important;
    background-color: #ffffff !important;
}
/* Hard overrides for Streamlit default white upload/button containers */
[data-testid="stMain"] [data-testid="stFileUploader"] section,
[data-testid="stMain"] [data-testid="stFileUploaderDropzone"] {
    background: #f8fffd !important;
    background-color: #f8fffd !important;
    border: 2px dashed rgba(20, 184, 166, 0.38) !important;
    box-shadow: none !important;
}
[data-testid="stMain"] [data-testid="stFileUploader"] button {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid rgba(20, 184, 166, 0.4) !important;
}
[data-testid="stMain"] [data-testid="stButton"] > button {
    background: #ffffff !important;
    background-image: none !important;
    color: #0f172a !important;
    border: 1px solid rgba(20, 184, 166, 0.4) !important;
    box-shadow: none !important;
}
[data-testid="stMain"] [data-testid="stButton"] > button:hover {
    background: #ecfdf9 !important;
    border-color: #14b8a6 !important;
}
[data-testid="stMain"] [data-testid="stButton"] > button:disabled {
    background: #f8fafc !important;
    color: #52525b !important;
    border-color: rgba(20, 184, 166, 0.2) !important;
    opacity: 0.85 !important;
}
</style>
"""


def _inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def api_base() -> str:
    return (st.session_state.get("api_base") or DEFAULT_API).rstrip("/")


def format_sidebar_datetime(ts_raw: str) -> str:
    """Format API ISO timestamps like May 2, 12:27 for sidebar rows."""
    if not ts_raw or not str(ts_raw).strip():
        return ""
    s = str(ts_raw).strip()
    try:
        from datetime import datetime

        if "T" in s:
            dt = datetime.fromisoformat(s[:19])
        elif len(s) >= 16:
            dt = datetime.strptime(s[:16], "%Y-%m-%d %H:%M")
        else:
            return s[:16].replace("T", " ")
        return f"{dt.strftime('%b')} {dt.day}, {dt.strftime('%H:%M')}"
    except Exception:
        return s[:19].replace("T", " ") if len(s) >= 19 else s


def api_get(
    path: str,
    timeout: int = 30,
    params: dict[str, str] | None = None,
) -> requests.Response:
    return requests.get(f"{api_base()}{path}", timeout=timeout, params=params or {})


def api_post(path: str, json_body: dict | None = None, **kw: Any) -> requests.Response:
    return requests.post(f"{api_base()}{path}", json=json_body, timeout=600, **kw)


def _stream_text_chunks(text: str, chunk_size: int = 8):
    """Yield strings for st.write_stream (fallback when API does not emit `token` SSE events)."""
    t = text or ""
    if not t:
        return
    n = len(t)
    for i in range(0, n, chunk_size):
        yield t[i : i + chunk_size]
        if i + chunk_size < n:
            time.sleep(_UI_STREAM_DELAY_SEC)


def run_streaming_chat(sid: str, uid: str, prompt: str) -> bool:
    """
    POST /chat/stream (SSE). Streamlit only redraws progressively for ``st.write_stream``,
    not for repeated ``empty().markdown`` in a loop—so we use a spinner while the graph runs
    and a generator that yields ``token`` chunks (or falls back to chunking the final reply).
    """
    url = f"{api_base()}/api/v1/chat/stream"
    t0 = time.perf_counter()
    stream_state: dict[str, Any] = {"done": None, "error": None, "n_tokens": 0}

    try:
        with requests.post(
            url,
            json={"session_id": sid, "user_id": uid, "message": prompt},
            stream=True,
            timeout=600,
        ) as resp:
            if resp.status_code == 404:
                st.error("Session not found. Start a new chat from the sidebar.")
                return False
            if resp.status_code != 200:
                st.error(f"Chat failed ({resp.status_code}): {resp.text}")
                return False

            def sse_answer_chunks():
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if not line.startswith("data:"):
                        continue
                    pl = line[5:].strip()
                    if not pl:
                        continue
                    try:
                        evt_obj = json.loads(pl)
                    except json.JSONDecodeError:
                        continue

                    ev = evt_obj.get("event")
                    if ev == "error":
                        stream_state["error"] = str(
                            evt_obj.get("detail", "Stream error")
                        )
                        return
                    if ev == "token":
                        piece = evt_obj.get("data")
                        if piece is None:
                            continue
                        if not isinstance(piece, str):
                            piece = str(piece)
                        stream_state["n_tokens"] = (
                            int(stream_state.get("n_tokens", 0)) + 1
                        )
                        yield piece
                        time.sleep(_UI_STREAM_DELAY_SEC)
                    elif ev == "done":
                        data = evt_obj.get("data")
                        d = data if isinstance(data, dict) else None
                        if d and int(stream_state.get("n_tokens", 0)) == 0:
                            ans = (d.get("response") or "").strip()
                            if ans:
                                yield from _stream_text_chunks(ans)
                        stream_state["done"] = d
                        return
                    # ev == "update": LangGraph progress — spinner covers this phase

            with st.spinner("Running agents & generating your answer…"):
                st.write_stream(sse_answer_chunks())

            if stream_state.get("error"):
                st.error(stream_state["error"])
                return False

            done_payload = stream_state.get("done")
            if not isinstance(done_payload, dict):
                st.error("No response from server (stream ended early).")
                return False

            answer = (done_payload.get("response") or "").strip()
            if not answer and int(stream_state.get("n_tokens", 0)) == 0:
                st.markdown("_No response text returned._")

            lat = done_payload.get("latency_seconds")
            if lat is None:
                lat = round(time.perf_counter() - t0, 2)
            st.markdown(
                f'<p class="cf-latency">⚡ {lat}s</p>',
                unsafe_allow_html=True,
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": done_payload.get("response") or ""}
            )
            st.session_state.last_sources = done_payload.get("sources") or []
            st.session_state.last_agent_log = done_payload.get("agent_log") or []
            st.session_state.last_summary = (
                done_payload.get("conversation_summary") or ""
            )
            st.session_state.last_latency = lat
            render_source_pills(st.session_state.last_sources)
            ag_log = st.session_state.last_agent_log or []
            if ag_log:
                with st.expander("Agent pipeline log", expanded=False):
                    st.json(ag_log)
            return True
    except requests.RequestException as e:
        st.error(f"**Cannot reach the API.** Is it running?\n\n`{e}`")
        return False


def check_health() -> tuple[bool, str]:
    try:
        r = api_get("/api/v1/health", timeout=5)
        if r.status_code == 200:
            return True, "Connected"
        return False, f"HTTP {r.status_code}"
    except requests.RequestException as e:
        return False, str(e)[:120]


def fetch_chunk_count(
    session_id: str | None = None, user_id: str | None = None
) -> int | None:
    """Chunk count: full store, or RAG scope for this chat (session ∪ same user) when both ids set."""
    try:
        params: dict[str, str] = {}
        if (session_id or "").strip():
            params["session_id"] = session_id.strip()
        if (user_id or "").strip():
            params["user_id"] = user_id.strip()
        q = params or None
        r = api_get("/api/v1/stats", timeout=5, params=q)
        if r.ok:
            return int(r.json().get("vector_store", {}).get("count", 0))
    except requests.RequestException:
        pass
    return None


def first_user_preview(session_id: str) -> tuple[str, str]:
    """Return (preview_label, iso_timestamp_hint) for sidebar row."""
    try:
        r = api_get(f"/api/v1/sessions/{session_id}/messages", timeout=15)
        if r.status_code != 200:
            return "New conversation", ""
        raw = r.json().get("messages") or []
        ts = ""
        if raw:
            ts = str(raw[0].get("timestamp") or "")[:19]
        for m in raw:
            if m.get("role") == "user":
                c = (m.get("content") or "").strip()
                if c:
                    prev = c[:40] + ("…" if len(c) > 40 else "")
                    return prev, ts
        return "New conversation", ts
    except requests.RequestException:
        return "New conversation", ""


def build_history_rows(user_id: str, limit: int = 15) -> list[dict[str, Any]]:
    try:
        r = api_get(f"/api/v1/users/{user_id}/sessions", timeout=15)
        if r.status_code != 200:
            return []
        rows = list(r.json().get("sessions") or [])
    except requests.RequestException:
        return []

    out: list[dict[str, Any]] = []
    for row in rows[:limit]:
        sid = row.get("session_id") or ""
        if not sid:
            continue
        preview, _ = first_user_preview(sid)
        ts = row.get("updated_at") or row.get("created_at") or ""
        ts_label = format_sidebar_datetime(str(ts)) if ts else ""
        out.append(
            {
                "session_id": sid,
                "preview": preview,
                "ts": ts_label,
            }
        )
    return out


def load_session_messages(session_id: str) -> None:
    try:
        r = api_get(f"/api/v1/sessions/{session_id}/messages", timeout=30)
        if r.status_code != 200:
            st.session_state.messages = []
            st.error("Could not load messages for this session.")
            return
        raw = r.json().get("messages") or []
        st.session_state.messages = [
            {"role": m["role"], "content": m["content"]} for m in raw
        ]
    except requests.RequestException as e:
        st.session_state.messages = []
        st.error(f"Could not load messages: {e}")


def render_source_pills(sources: list[dict] | None) -> None:
    """Inline relevance pills under assistant replies (mockup-style)."""
    if not sources:
        return
    parts: list[str] = []
    for s in sources[:10]:
        title = s.get("title") or s.get("source") or "source"
        label = (str(title).split(" · ")[0]).strip()[:48]
        rel = float(s.get("relevance") or 0)
        parts.append(f'<span class="cf-pill">{html.escape(label)} {rel:.0%}</span>')
    if parts:
        st.markdown(
            '<div class="cf-pills">' + "".join(parts) + "</div>",
            unsafe_allow_html=True,
        )


def render_sources(sources: list[dict] | None) -> None:
    if not sources:
        st.caption("No sources returned for this reply.")
        return
    for s in sources[:12]:
        title = s.get("title") or "—"
        src = (s.get("source") or "").strip()
        rel = float(s.get("relevance") or 0)
        if src and src not in title:
            st.markdown(f"- **{title}** — `{src}` — {rel:.0%}")
        else:
            st.markdown(f"- **{title}** — {rel:.0%}")


def render_agent_log(log: list[dict] | None) -> None:
    if not log:
        st.caption("No agent log entries.")
        return
    for entry in log:
        node = entry.get("node", "unknown")
        lines = [f"**{node}**"]
        for k, v in entry.items():
            if k == "node":
                continue
            lines.append(f"  - `{k}`: `{v}`")
        st.markdown("\n".join(lines))


def sidebar_branding() -> None:
    st.markdown(
        f'<div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:4px;">'
        f'<span style="font-size:1.85rem;line-height:1;">🧠</span>'
        f'<div><div style="font-weight:800;font-size:1.2rem;letter-spacing:-0.03em;color:#fafaf9;">'
        f"CogniFlow</div>"
        f'<div style="font-size:0.78rem;color:{_TEAL};opacity:0.8;margin-top:4px;line-height:1.35;">'
        f"Document-grounded assistant</div></div></div>",
        unsafe_allow_html=True,
    )


def sidebar_status(ok: bool, msg: str) -> None:
    label = "API connected" if ok else html.escape(msg[:120])
    dot_cls = "cf-status-dot" if ok else "cf-status-dot off"
    st.markdown(
        f'<div class="cf-status"><span class="{dot_cls}"></span><span>{label}</span></div>',
        unsafe_allow_html=True,
    )


def sidebar_new_chat(user_id: str) -> None:
    if st.button("✨  +  New chat", use_container_width=True, type="primary"):
        uid = (user_id or "").strip()
        if not uid:
            uid = f"guest_{uuid.uuid4().hex[:8]}"
            st.session_state.user_id = uid
        try:
            r = api_post("/api/v1/sessions", {"user_id": uid})
            if r.status_code == 200:
                data = r.json()
                st.session_state.session_id = data["session_id"]
                st.session_state.messages = []
                st.session_state.last_sources = []
                st.session_state.last_agent_log = []
                st.session_state.last_summary = ""
                st.session_state.last_latency = None
                st.rerun()
            else:
                st.error(r.text)
        except requests.RequestException as e:
            st.error(f"Cannot create session: {e}")


def _clear_active_if_dismissed(active_sid: str) -> None:
    if active_sid and active_sid in st.session_state.dismissed_sessions:
        st.session_state.session_id = ""
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.last_agent_log = []
        st.session_state.last_summary = ""
        st.session_state.last_latency = None


def sidebar_history(user_id: str, active_sid: str) -> None:
    """ChatGPT-style list: open chat + trash per row; Clear all at bottom (UI-only)."""
    st.markdown(
        '<p class="cf-sec-label" style="margin-top:0;margin-bottom:0.5rem;">Chat history</p>',
        unsafe_allow_html=True,
    )

    rows = [
        r
        for r in build_history_rows(user_id)
        if r["session_id"] not in st.session_state.dismissed_sessions
    ]
    if not rows:
        st.caption("No past sessions yet.")
        return

    for row in rows:
        sid = row["session_id"]
        preview = row["preview"]
        ts_line = row["ts"]
        is_active = sid == active_sid
        if ts_line:
            btn_label = f"{'▸ ' if is_active else ''}{preview}\n{ts_line}"
        else:
            btn_label = f"{'▸ ' if is_active else ''}{preview}"
        with st.container(border=is_active):
            col_chat, col_del = st.columns([11, 1], vertical_alignment="center")
            with col_chat:
                if st.button(
                    btn_label,
                    key=f"hist_{sid}",
                    use_container_width=True,
                    type="secondary",
                ):
                    st.session_state.session_id = sid
                    load_session_messages(sid)
                    st.session_state.last_sources = []
                    st.session_state.last_agent_log = []
                    st.session_state.last_summary = ""
                    st.session_state.last_latency = None
                    st.rerun()
            with col_del:
                if st.button(
                    "🗑️",
                    key=f"hist_del_{sid}",
                    help="Remove from sidebar (UI only)",
                    use_container_width=True,
                    type="tertiary",
                ):
                    st.session_state.dismissed_sessions.add(sid)
                    if sid == active_sid:
                        st.session_state.session_id = ""
                        st.session_state.messages = []
                        st.session_state.last_sources = []
                        st.session_state.last_agent_log = []
                        st.session_state.last_summary = ""
                        st.session_state.last_latency = None
                    st.rerun()

    if st.button(
        "Clear all from sidebar",
        key="hist_clear_all",
        use_container_width=True,
        type="tertiary",
        help="Hide every chat in this list (UI only — reload page to show API list again)",
    ):
        for r in build_history_rows(user_id):
            st.session_state.dismissed_sessions.add(r["session_id"])
        _clear_active_if_dismissed(active_sid)
        st.rerun()


def _post_one_document(
    session_id: str, uploaded_file: Any
) -> tuple[int, dict[str, Any] | str]:
    """POST a single file; returns (status_code, json dict or error text)."""
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        ),
    }
    r = requests.post(
        f"{api_base()}/api/v1/documents/upload",
        files=files,
        data={"session_id": session_id},
        timeout=600,
    )
    if r.status_code != 200:
        return r.status_code, r.text
    return r.status_code, r.json()


def _flush_kb_notices(scope: str) -> None:
    """Show one-shot messages after indexing (survives st.rerun)."""
    key = f"kb_notice_{scope}"
    if key not in st.session_state:
        return
    payload = st.session_state.pop(key)
    parts = payload.get("parts") if isinstance(payload, dict) else None
    if not parts:
        return
    for p in parts:
        if not isinstance(p, dict):
            continue
        kind = (p.get("kind") or "info").lower()
        text = (p.get("text") or "").strip()
        if not text:
            continue
        if kind == "success":
            st.success(text)
        elif kind == "error":
            st.error(text)
        else:
            st.info(text)


def index_session_documents(
    session_id: str,
    uploaded_files: list[Any],
    uploader_scope: str | None = None,
) -> None:
    """Index several files in one go (sequential API calls). Plain-language feedback."""
    sid = (session_id or "").strip()
    if not sid or not uploaded_files:
        return
    new_names: list[str] = []
    already_names: list[str] = []
    errors: list[str] = []
    try:
        for uf in uploaded_files:
            code, body = _post_one_document(sid, uf)
            label = getattr(uf, "name", "file")
            if code != 200:
                detail = body if isinstance(body, str) else f"HTTP {code}"
                errors.append(f"**{label}** — could not add ({detail[:120]})")
                continue
            assert isinstance(body, dict)
            status = body.get("status", "")
            display_name = (body.get("filename") or label or "file").strip()
            if status == "already_indexed":
                already_names.append(display_name)
            else:
                new_names.append(display_name)

        notice_parts: list[dict[str, str]] = []
        uniq_new = list(dict.fromkeys(new_names))
        uniq_already = list(dict.fromkeys(already_names))

        if uniq_new:
            if len(uniq_new) == 1:
                notice_parts.append(
                    {
                        "kind": "success",
                        "text": f"**Added to this chat:** {uniq_new[0]}. You can ask about it below.",
                    }
                )
            else:
                joined = ", ".join(uniq_new)
                notice_parts.append(
                    {
                        "kind": "success",
                        "text": f"**Added to this chat:** {joined}. You can ask about them below.",
                    }
                )

        if uniq_already:
            if len(uniq_already) == 1:
                already_txt = (
                    f"**{uniq_already[0]}** is already in this chat — nothing to import again. "
                    "Try asking a question about it below."
                )
            else:
                already_txt = (
                    "These files are **already in this chat**: "
                    f"{', '.join(uniq_already)}. "
                    "You do not need to add them again — go ahead and ask a question."
                )
            notice_parts.append({"kind": "info", "text": already_txt})

        if errors:
            notice_parts.append(
                {
                    "kind": "error",
                    "text": "Could not add some files:\n\n" + "\n\n".join(errors),
                }
            )

        if uploader_scope in ("chat", "sidebar") and notice_parts:
            st.session_state[f"kb_notice_{uploader_scope}"] = {"parts": notice_parts}
            vkey = (
                "kb_upload_v_chat"
                if uploader_scope == "chat"
                else "kb_upload_v_sidebar"
            )
            st.session_state[vkey] = int(st.session_state.get(vkey, 0)) + 1
            st.rerun()
        elif notice_parts:
            for p in notice_parts:
                k = (p.get("kind") or "info").lower()
                t = (p.get("text") or "").strip()
                if not t:
                    continue
                if k == "success":
                    st.success(t)
                elif k == "error":
                    st.error(t)
                else:
                    st.info(t)
    except requests.RequestException as e:
        st.error(f"Could not reach the server. Check that the API is running.\n\n{e}")


def chat_area_document_upload(active_session_id: str) -> None:
    """File upload + Index in the main chat column (same session scope as sidebar)."""
    sid = (active_session_id or "").strip()
    if not sid:
        return
    _flush_kb_notices("chat")
    st.markdown(
        '<div class="cf-upload-card">'
        '<p class="cf-upload-title">Attach Documents</p>'
        "<p class=\"cf-upload-sub\">Drop files here and click <b>Index in this chat</b>. "
        "The same user can search these chunks from other chats too.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    _v = int(st.session_state.get("kb_upload_v_chat", 0))
    cu = st.file_uploader(
        "Add files for this conversation",
        type=["pdf", "md", "markdown", "html", "htm"],
        accept_multiple_files=True,
        key=f"chat_kb_upload_{_v}",
        label_visibility="collapsed",
    )
    if cu:
        st.markdown(
            f'<p class="cf-upload-count">{len(cu)} file(s) selected for this chat.</p>',
            unsafe_allow_html=True,
        )
    can_index = bool(cu)
    if st.button(
        "Index in this chat",
        key="chat_kb_index_btn",
        use_container_width=True,
        disabled=not can_index,
        type="secondary",
    ):
        if cu:
            index_session_documents(sid, list(cu), "chat")


def sidebar_upload(active_session_id: str) -> None:
    st.markdown('<p class="cf-sec-label">Knowledge base</p>', unsafe_allow_html=True)
    st.caption("Uploads are tagged to this chat and **your user id** — retrieval includes this chat **and** your other chats’ files.")
    sid = (active_session_id or "").strip()
    if not sid:
        st.caption("Open or start a chat to attach uploads to that session.")
    _flush_kb_notices("sidebar")
    _sv = int(st.session_state.get("kb_upload_v_sidebar", 0))
    up = st.file_uploader(
        "Upload",
        type=["pdf", "md", "markdown", "html", "htm"],
        accept_multiple_files=True,
        key=f"sidebar_kb_upload_{_sv}",
        label_visibility="collapsed",
    )
    can_index = bool(up) and bool(sid)
    if st.button("Index", use_container_width=True, disabled=not can_index):
        if not up or not sid:
            return
        index_session_documents(sid, list(up), "sidebar")

    uid = (st.session_state.get("user_id") or "").strip()
    n = fetch_chunk_count(sid if sid else None, uid if sid else None)
    if n is not None:
        lbl = (
            "chunks searchable here (this chat + your account)"
            if sid
            else "chunks in vector store"
        )
        st.markdown(
            f'<div class="cf-chunk-stat"><div class="n">{n}</div>'
            f'<div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )


def sidebar_settings() -> None:
    with st.expander("Settings", expanded=False):
        st.text_input("User ID", key="user_id")
        st.text_input("API base URL", key="api_base", placeholder=DEFAULT_API)


def welcome_empty(api_ok: bool) -> None:
    if not api_ok:
        st.error(
            f"**API unreachable** at `{api_base()}`. Start the backend: "
            "`python main.py` or `uvicorn main:app --host 0.0.0.0 --port 8000`."
        )
        return
    st.markdown(
        f'<div style="background:#050505;border:1px solid rgba(20,184,166,0.35);border-radius:14px;padding:1.1rem 1.35rem;'
        f'max-width:36rem;margin:0 auto;">'
        f'<p style="color:#e7e5e4;margin:0;font-size:0.95rem;line-height:1.65;">'
        f"<b>Welcome.</b> Start the API, tap <b>+ New chat</b>, then add files in the sidebar "
        f'or in the chat under <b style="color:{_TEAL};">Attach documents</b> — same session '
        f"index for both.</p></div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="CogniFlow",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    if "user_id" not in st.session_state:
        st.session_state.user_id = "demo_user"
    if "session_id" not in st.session_state:
        st.session_state.session_id = ""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_base" not in st.session_state:
        st.session_state.api_base = DEFAULT_API
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    if "last_agent_log" not in st.session_state:
        st.session_state.last_agent_log = []
    if "last_summary" not in st.session_state:
        st.session_state.last_summary = ""
    if "last_latency" not in st.session_state:
        st.session_state.last_latency = None
    if "dismissed_sessions" not in st.session_state:
        st.session_state.dismissed_sessions = set()
    if "kb_upload_v_chat" not in st.session_state:
        st.session_state.kb_upload_v_chat = 0
    if "kb_upload_v_sidebar" not in st.session_state:
        st.session_state.kb_upload_v_sidebar = 0

    ok, health_msg = check_health()
    uid = str(st.session_state.user_id)
    sid = str(st.session_state.session_id).strip()

    with st.sidebar:
        sidebar_branding()
        sidebar_status(ok, health_msg)
        st.divider()
        sidebar_new_chat(uid)
        st.divider()
        sidebar_history(uid, sid)
        st.divider()
        sidebar_upload(sid)
        st.divider()
        sidebar_settings()

    st.markdown(
        '<div class="cf-main-hero"><h1>CogniFlow</h1>'
        "<p>Conversational RAG · LangGraph · ChromaDB</p></div>",
        unsafe_allow_html=True,
    )

    if not sid:
        welcome_empty(ok)
        return

    chat_area_document_upload(sid)

    for m in st.session_state.messages:
        av = "🧑" if m["role"] == "user" else "🧠"
        with st.chat_message(m["role"], avatar=av):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask about your documents…")
    if not prompt:
        if st.session_state.last_sources:
            st.divider()
            lat = st.session_state.last_latency
            if lat is not None:
                st.markdown(
                    f'<p class="cf-latency">⚡ Last response: {lat}s</p>',
                    unsafe_allow_html=True,
                )
            render_source_pills(st.session_state.last_sources)
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧠"):
        run_streaming_chat(sid, uid, prompt)


if __name__ == "__main__":
    main()

"""
CogniFlow — minimal Streamlit chat UI for the FastAPI RAG backend.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import requests
import streamlit as st

DEFAULT_API = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

# --- Minimal styling (sidebar tone, status, hide footer) --------------------
_CSS = """
<style>
div[data-testid="stSidebar"] { background: #1e293b !important; }
div[data-testid="stSidebar"] label { color: #e2e8f0 !important; }
div[data-testid="stSidebar"] .stTextInput input {
    background: #0f172a !important; color: #f8fafc !important;
    border-color: #334155 !important;
}
.status-ok { color: #22c55e; font-weight: 600; }
.status-bad { color: #ef4444; font-weight: 600; }
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
/* Chat history — compact trash button column */
div[data-testid="stSidebar"] div[data-testid="column"] button[kind="secondary"] {
    min-height: 2rem; padding: 0 0.35rem; font-size: 1rem; line-height: 1;
    border: none; background: transparent; color: #94a3b8;
}
div[data-testid="stSidebar"] div[data-testid="column"] button[kind="secondary"]:hover {
    color: #f87171; background: rgba(248,113,113,0.12);
}
</style>
"""


def _inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def api_base() -> str:
    return (st.session_state.get("api_base") or DEFAULT_API).rstrip("/")


def api_get(path: str, timeout: int = 30) -> requests.Response:
    return requests.get(f"{api_base()}{path}", timeout=timeout)


def api_post(path: str, json_body: dict | None = None, **kw: Any) -> requests.Response:
    return requests.post(f"{api_base()}{path}", json=json_body, timeout=600, **kw)


def check_health() -> tuple[bool, str]:
    try:
        r = api_get("/api/v1/health", timeout=5)
        if r.status_code == 200:
            return True, "Connected"
        return False, f"HTTP {r.status_code}"
    except requests.RequestException as e:
        return False, str(e)[:120]


def fetch_chunk_count() -> int | None:
    try:
        r = api_get("/api/v1/stats", timeout=5)
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
        if isinstance(ts, str) and len(ts) >= 16:
            ts_short = ts[:16].replace("T", " ")
        else:
            ts_short = str(ts)[:16] if ts else ""
        out.append(
            {
                "session_id": sid,
                "preview": preview,
                "ts": ts_short,
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
    st.markdown("### 🧠 CogniFlow")
    st.caption("Conversational RAG · LangGraph · ChromaDB")


def sidebar_status(ok: bool, msg: str) -> None:
    cls = "status-ok" if ok else "status-bad"
    dot = "🟢" if ok else "🔴"
    st.markdown(f'<p class="{cls}">{dot} {msg}</p>', unsafe_allow_html=True)


def sidebar_new_chat(user_id: str) -> None:
    st.subheader("New Chat")
    if st.button("New Chat", use_container_width=True, type="primary"):
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
    """ChatGPT-style list: open chat + trash per row; Clear all (UI-only hide on this browser)."""
    h_left, h_right = st.columns([2, 1])
    with h_left:
        st.markdown("**Chat History**")
    with h_right:
        if st.button(
            "Clear all",
            key="hist_clear_all",
            use_container_width=True,
            help="Remove every chat from this sidebar (UI only — reload page to show API list again)",
        ):
            for r in build_history_rows(user_id):
                st.session_state.dismissed_sessions.add(r["session_id"])
            _clear_active_if_dismissed(active_sid)
            st.rerun()

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
        label = f"{row['preview']} · {row['ts']}" if row["ts"] else row["preview"]
        is_active = sid == active_sid
        btn_label = f"{'● ' if is_active else ''}{label}"
        col_chat, col_del = st.columns([5, 1])
        with col_chat:
            if st.button(btn_label, key=f"hist_{sid}", use_container_width=True):
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
                type="secondary",
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


def sidebar_upload() -> None:
    st.subheader("Upload Document")
    up = st.file_uploader(
        "File",
        type=["pdf", "md", "markdown", "html", "htm"],
        label_visibility="collapsed",
    )
    if st.button("Index", use_container_width=True, disabled=up is None):
        if not up:
            return
        files = {"file": (up.name, up.getvalue(), up.type or "application/octet-stream")}
        try:
            r = requests.post(
                f"{api_base()}/api/v1/documents/upload",
                files=files,
                timeout=600,
            )
            if r.status_code == 200:
                j = r.json()
                st.success(f"Indexed **{j.get('num_chunks', 0)}** chunks from `{j.get('filename')}`")
            else:
                st.error(r.text)
        except requests.RequestException as e:
            st.error(str(e))

    n = fetch_chunk_count()
    if n is not None:
        st.caption(f"**{n}** chunks indexed (vector store)")


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
    st.info(
        """
**Welcome.** Quick start:

1. Start the API (`python main.py`)
2. Click **New Chat** to begin
3. Upload documents to build your knowledge base
        """.strip()
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
        sidebar_upload()
        st.divider()
        sidebar_settings()

    st.markdown("### 🧠 CogniFlow")
    st.caption("Conversational RAG · LangGraph · ChromaDB")

    if not sid:
        welcome_empty(ok)
        return

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Message…")
    if not prompt:
        if st.session_state.last_sources or st.session_state.last_agent_log or (
            st.session_state.last_summary
        ):
            st.divider()
            lat = st.session_state.last_latency
            if lat is not None:
                st.caption(f"⚡ Last response: **{lat}s**")
            with st.expander("📚 Sources", expanded=False):
                render_sources(st.session_state.last_sources)
            with st.expander("🔍 Agent Decisions", expanded=False):
                render_agent_log(st.session_state.last_agent_log)
            with st.expander("📝 Conversation Summary", expanded=False):
                summ = (st.session_state.last_summary or "").strip()
                if summ:
                    st.markdown(summ)
                else:
                    st.caption("No summary yet (appears after longer conversations).")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            t0 = time.perf_counter()
            try:
                r = api_post(
                    "/api/v1/chat",
                    {
                        "session_id": sid,
                        "user_id": uid,
                        "message": prompt,
                    },
                )
            except requests.RequestException as e:
                st.error(
                    f"**Cannot reach the API.** Is it running?\n\n`{e}`"
                )
                return

        if r.status_code != 200:
            st.error(f"Chat failed ({r.status_code}): {r.text}")
            return

        data = r.json()
        answer = data.get("response") or ""
        st.markdown(answer)

        lat = data.get("latency_seconds")
        if lat is None:
            lat = round(time.perf_counter() - t0, 2)
        st.caption(f"⚡ {lat}s")

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.last_sources = data.get("sources") or []
        st.session_state.last_agent_log = data.get("agent_log") or []
        st.session_state.last_summary = data.get("conversation_summary") or ""
        st.session_state.last_latency = lat

        with st.expander("📚 Sources", expanded=False):
            render_sources(st.session_state.last_sources)
        with st.expander("🔍 Agent Decisions", expanded=False):
            render_agent_log(st.session_state.last_agent_log)
        with st.expander("📝 Conversation Summary", expanded=False):
            summ = (st.session_state.last_summary or "").strip()
            if summ:
                st.markdown(summ)
            else:
                st.caption("No summary yet (appears after longer conversations).")


if __name__ == "__main__":
    main()

"""
Streamlit UI for CogniFlow: chat, sessions, document upload, and API diagnostics.
Set API_BASE_URL=http://api:8000 when the UI runs in Docker (see docker-compose.yml).
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

DEFAULT_API = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 960px; }
        div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f1419 0%, #151b24 100%); }
        div[data-testid="stSidebar"] label { color: #c8d0dc !important; }
        .cf-badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 6px;
            font-size: 0.75rem; font-weight: 600; letter-spacing: 0.02em; }
        .cf-ok { background: #1a3d2e; color: #7dffb3; }
        .cf-bad { background: #3d1a1a; color: #ff9d9d; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def api_base() -> str:
    return (st.session_state.get("api_base") or DEFAULT_API).rstrip("/")


def api_post(path: str, json_body: dict | None = None, **kwargs: Any):
    url = f"{api_base()}{path}"
    return requests.post(url, json=json_body, timeout=600, **kwargs)


def api_get(path: str, timeout: int = 60):
    url = f"{api_base()}{path}"
    return requests.get(url, timeout=timeout)


def check_health() -> tuple[bool, str]:
    try:
        r = api_get("/api/v1/health", timeout=5)
        if r.status_code == 200:
            return True, "Connected"
        return False, f"HTTP {r.status_code}"
    except requests.RequestException as e:
        return False, str(e)[:80]


def fetch_user_sessions(user_id: str) -> list[dict[str, Any]]:
    try:
        r = api_get(f"/api/v1/users/{user_id}/sessions")
    except requests.RequestException:
        return []
    if r.status_code != 200:
        return []
    data = r.json()
    return list(data.get("sessions") or [])


def render_sources(sources: list[dict[str, Any]] | None) -> None:
    if not sources:
        st.caption("No document sources for this reply.")
        return
    rows: list[dict[str, Any]] = []
    for s in sources[:12]:
        rows.append(
            {
                "Title": s.get("title") or "—",
                "Relevance": f"{float(s.get('relevance') or 0):.0%}",
                "Source": (s.get("source") or "")[:64] + ("…" if len(str(s.get("source") or "")) > 64 else ""),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True, height=min(220, 36 + len(rows) * 36))


def sidebar_connection(ok: bool, msg: str) -> None:
    st.subheader("Connection")
    badge = f'<span class="cf-badge cf-{"ok" if ok else "bad"}">{msg}</span>'
    st.markdown(badge, unsafe_allow_html=True)
    if st.button("Refresh status", use_container_width=True):
        st.rerun()


def sidebar_identity() -> None:
    st.subheader("Identity")
    st.text_input("User ID", key="user_id", help="Used for sessions and long-term memory.")


def sidebar_sessions() -> None:
    st.subheader("Session")
    uid = st.session_state.get("user_id") or "demo_user"
    sessions = fetch_user_sessions(uid)

    if sessions:
        labels: list[str] = []
        id_by_label: dict[str, str] = {}
        for row in sessions:
            sid = row.get("session_id") or ""
            ts = row.get("updated_at") or row.get("created_at") or ""
            short = f"{sid[:8]}…" if len(sid) > 12 else sid
            label = f"{short} · {ts[:19] if isinstance(ts, str) else ts}"
            labels.append(label)
            id_by_label[label] = sid

        current = st.session_state.get("session_id") or ""
        default_idx = 0
        for i, row in enumerate(sessions):
            if (row.get("session_id") or "") == current:
                default_idx = i
                break

        choice = st.selectbox(
            "Your sessions",
            options=labels,
            index=min(default_idx, len(labels) - 1) if labels else 0,
            key="session_pick",
        )
        if choice and id_by_label.get(choice):
            if st.button("Open selected session", use_container_width=True):
                sid = id_by_label[choice]
                st.session_state.session_id = sid
                try:
                    r = api_get(f"/api/v1/sessions/{sid}/messages")
                except requests.RequestException as e:
                    st.session_state.messages = []
                    st.error(f"Cannot reach API: {e}")
                else:
                    if r.status_code == 200:
                        raw = r.json().get("messages") or []
                        st.session_state.messages = [
                            {"role": m["role"], "content": m["content"]} for m in raw
                        ]
                        st.rerun()
                    else:
                        st.session_state.messages = []
                        st.error(r.text)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("New session", use_container_width=True, type="primary"):
            try:
                r = api_post("/api/v1/sessions", {"user_id": uid})
            except requests.RequestException as e:
                st.error(
                    "Cannot reach the API. Start it in another terminal: "
                    f"`python main.py` — ({e})"
                )
            else:
                if r.status_code == 200:
                    data = r.json()
                    st.session_state.session_id = data["session_id"]
                    st.session_state.messages = []
                    st.success("Session created.")
                    st.rerun()
                else:
                    st.error(r.text)
    with c2:
        if st.button("Load history", use_container_width=True, disabled=not st.session_state.get("session_id")):
            sid = st.session_state.session_id
            try:
                r = api_get(f"/api/v1/sessions/{sid}/messages")
            except requests.RequestException as e:
                st.error(f"Cannot reach API: {e}")
            else:
                if r.status_code == 200:
                    raw = r.json().get("messages") or []
                    st.session_state.messages = [
                        {"role": m["role"], "content": m["content"]} for m in raw
                    ]
                    st.toast("History loaded from server.")
                    st.rerun()
                else:
                    st.error(r.text)

    st.text_input(
        "Session ID",
        key="session_id",
        help="Paste a session UUID to resume, or use New session.",
    )


def sidebar_documents() -> None:
    st.subheader("Knowledge base")
    up = st.file_uploader(
        "Upload PDF, Markdown, or HTML",
        type=["pdf", "md", "markdown", "html", "htm"],
        label_visibility="collapsed",
    )
    if up and st.button("Index file", use_container_width=True):
        files = {"file": (up.name, up.getvalue(), up.type or "application/octet-stream")}
        url = f"{api_base()}/api/v1/documents/upload"
        try:
            r = requests.post(url, files=files, timeout=600)
            if r.status_code == 200:
                j = r.json()
                st.success(f"Indexed **{j.get('num_chunks', 0)}** chunks from `{j.get('filename')}`")
            else:
                st.error(r.text)
        except requests.RequestException as e:
            st.error(str(e))


def sidebar_diagnostics() -> None:
    with st.expander("API & stats", expanded=False):
        st.text_input("API base URL", key="api_base")
        if st.button("Fetch /stats"):
            try:
                r = api_get("/api/v1/stats")
            except requests.RequestException as e:
                st.error(str(e))
            else:
                if r.ok:
                    st.json(r.json())
                else:
                    st.error(r.text)
        if st.button("Fetch /metrics"):
            try:
                r = api_get("/api/v1/metrics")
            except requests.RequestException as e:
                st.error(str(e))
            else:
                if r.ok:
                    st.json(r.json())
                else:
                    st.error(r.text)


def render_empty_state(api_reachable: bool) -> None:
    if not api_reachable:
        st.warning(
            f"No API at **{api_base()}**. Start the backend in another terminal: "
            "`python main.py` (or `uvicorn main:app --host 0.0.0.0 --port 8000`), "
            "then refresh the sidebar or reload this page."
        )
    st.markdown(
        """
        **CogniFlow** answers using your indexed documents and remembers context per user.

        1. Ensure the API is running (`python main.py` or Docker).
        2. Create a **New session** in the sidebar (or open an existing one).
        3. Optionally **Index file** to add documents to the vector store.
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="CogniFlow",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_style()

    if "user_id" not in st.session_state:
        st.session_state.user_id = "demo_user"
    if "session_id" not in st.session_state:
        st.session_state.session_id = ""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_base" not in st.session_state:
        st.session_state.api_base = DEFAULT_API

    st.title("CogniFlow")
    st.caption("Conversational RAG · LangGraph · ChromaDB")

    ok_api, health_msg = check_health()

    with st.sidebar:
        sidebar_connection(ok_api, health_msg)
        st.divider()
        sidebar_identity()
        st.divider()
        sidebar_sessions()
        st.divider()
        sidebar_documents()
        st.divider()
        sidebar_diagnostics()

    if not st.session_state.session_id.strip():
        render_empty_state(ok_api)
        return

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask something about your documents…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Running agents…"):
            try:
                r = api_post(
                    "/api/v1/chat",
                    {
                        "session_id": st.session_state.session_id.strip(),
                        "user_id": st.session_state.user_id,
                        "message": prompt,
                    },
                )
            except requests.RequestException as e:
                st.error(
                    "Cannot reach the API. Is `python main.py` (or Docker) running? "
                    f"Details: {e}"
                )
                st.session_state.messages.pop()
                return
        if r.status_code != 200:
            st.error(r.text)
            st.session_state.messages.pop()
            return

        data = r.json()
        text = data.get("response") or ""
        st.markdown(text)

        lat = data.get("latency_seconds")
        if lat is not None:
            st.caption(f"Latency: **{lat}s**")

        summ = (data.get("conversation_summary") or "").strip()
        if summ:
            with st.expander("Conversation summary (rolling)", expanded=False):
                st.markdown(summ)

        with st.expander("Sources", expanded=False):
            render_sources(data.get("sources"))

        with st.expander("Agent log", expanded=False):
            st.json(data.get("agent_log") or [])

        st.session_state.messages.append({"role": "assistant", "content": text})


if __name__ == "__main__":
    main()

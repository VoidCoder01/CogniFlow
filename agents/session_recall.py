"""Deterministic session-scoped recall (no retrieval, no user_memory_context, no LLM).

Intent ``session_recall`` is assigned only by the Query Understanding structured router.
This module only formats ``conversation_history`` + ``conversation_summary`` already
scoped to ``session_id`` by the API — it does not fetch other sessions or vector data.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator

from agents.graph_state import CogniFlowState
from agents.node_utils import with_log_timing

logger = logging.getLogger(__name__)

# No discussion to report (strict: no messages and no summary text).
EMPTY_SESSION_NO_DISCUSSION = (
    "We haven't discussed anything yet in this session."
)

SCOPE_HEADER = "In this session, you asked:"

# Above this many in-thread messages, prepend session summary + recent user questions only.
LONG_HISTORY_MAX_MESSAGES = 24
RECENT_WINDOW_MESSAGES = 14


def _nonempty_messages(history: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in history or []:
        if not isinstance(m, dict):
            continue
        if (m.get("content") or "").strip():
            out.append(m)
    return out


def _user_queries(messages: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for m in messages:
        if str(m.get("role") or "").strip().lower() != "user":
            continue
        t = (m.get("content") or "").strip()
        if t and t not in seen:
            seen.append(t)
    return seen


def handle_session_recall(state: dict[str, Any]) -> str:
    """
    Build a recall reply from this session's ``conversation_history`` and
    ``conversation_summary`` only. Ignores ``user_memory_context`` and
    ``cross_session_context`` even if present on ``state``.
    """
    session_id = str(state.get("session_id") or "").strip()
    history_raw = state.get("conversation_history") or []
    summary = (state.get("conversation_summary") or "").strip()

    messages = _nonempty_messages(history_raw)
    if not messages and not summary:
        logger.debug("session_recall: no history or summary session_id=%s", session_id or "(none)")
        return EMPTY_SESSION_NO_DISCUSSION

    all_user_q = _user_queries(messages)
    n_msg = len(messages)
    long_thread = n_msg > LONG_HISTORY_MAX_MESSAGES

    if long_thread:
        lines = [SCOPE_HEADER, ""]
        if summary:
            lines.extend(["Session summary (this chat thread):", summary, ""])
        tail = messages[-RECENT_WINDOW_MESSAGES:]
        recent_q = _user_queries(tail)
        if recent_q:
            lines.append("Recent questions:")
            lines.extend(f"- {q}" for q in recent_q)
        elif summary:
            lines.append("(No user questions in the recent window; see summary above.)")
        else:
            lines.append("(No user questions in the recent window.)")
        return "\n".join(lines).strip()

    lines = [SCOPE_HEADER, ""]
    if all_user_q:
        lines.extend(f"- {q}" for q in all_user_q)
    elif summary:
        lines.append("From the session summary:")
        lines.append(summary)
    else:
        lines.append("(No user questions recorded in this thread yet.)")
    return "\n".join(lines).strip()


def session_recall_node(state: CogniFlowState) -> dict[str, Any]:
    """LangGraph node: deterministic recall before retrieval / synthesis."""
    t0 = time.perf_counter()
    text = handle_session_recall(dict(state))
    log_entry = with_log_timing(
        {
            "node": "session_recall",
            "response_chars": len(text),
            "intent": state.get("query_intent"),
        },
        t0,
    )
    return {
        "response": text,
        "synthesized_context": "",
        "retrieved_documents": [],
        "use_retrieved_context": False,
        "agent_log": [log_entry],
    }


def iter_session_recall_events(state: CogniFlowState) -> Iterator[dict[str, Any]]:
    """Streaming API parity: single ``complete`` event (no tokenization)."""
    patch = session_recall_node(state)
    yield {
        "type": "complete",
        "response": patch.get("response", ""),
        "synthesized_context": patch.get("synthesized_context", ""),
        "retrieved_documents": patch.get("retrieved_documents") or [],
        "agent_log": patch.get("agent_log") or [],
    }

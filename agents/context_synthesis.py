from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.graph_state import CogniFlowState
from core.llm_provider import get_chat_model

logger = logging.getLogger(__name__)

_SYSTEM = """You are CogniFlow, a precise assistant grounded in retrieved documentation and conversation context.
If sources are provided, cite them implicitly (filename/title) and do not invent APIs not in the sources.
If no sources apply, answer briefly from general knowledge and say when uncertain."""


def _format_docs(docs: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, d in enumerate(docs[:8], 1):
        meta = d.get("metadata") or {}
        src = meta.get("source") or meta.get("title") or "source"
        body = (d.get("content") or "")[:4000]
        lines.append(f"[{i}] {src}\n{body}")
    return "\n\n".join(lines) if lines else "(no documents retrieved)"


def context_synthesis_node(state: CogniFlowState) -> dict[str, Any]:
    model = get_chat_model()
    history = state.get("conversation_history") or []
    hist_snip = "\n".join(
        f'{m.get("role", "user")}: {m.get("content", "")}'
        for m in history[-12:]
    )
    docs = state.get("retrieved_documents") or []
    summary = state.get("conversation_summary") or ""
    mem_ctx = (state.get("user_memory_context") or "").strip()
    doc_block = _format_docs(docs)
    mem_block = f"Known user preferences/context (cross-session):\n{mem_ctx}\n\n" if mem_ctx else ""
    user_block = (
        f"Conversation summary (if any): {summary or '(none)'}\n\n"
        f"{mem_block}"
        f"Recent messages:\n{hist_snip or '(empty)'}\n\n"
        f"Retrieved sources:\n{doc_block}\n\n"
        f"User question:\n{state.get('user_query', '')}"
    )
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=user_block),
    ]
    resp = model.invoke(messages)
    text = (getattr(resp, "content", None) or str(resp)).strip()
    log_entry = {
        "node": "context_synthesis",
        "response_chars": len(text),
        "intent": state.get("query_intent"),
    }
    sources = [
        {"id": d.get("id"), "metadata": d.get("metadata")}
        for d in docs[:10]
        if isinstance(d, dict)
    ]
    return {
        "response": text,
        "synthesized_context": doc_block[:8000],
        "agent_log": [log_entry],
    }

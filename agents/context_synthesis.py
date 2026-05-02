from __future__ import annotations

import logging
import time
from typing import Any, Iterator

from langchain_core.messages import HumanMessage, SystemMessage

from agents.graph_state import CogniFlowState
from agents.node_utils import with_log_timing
from agents.retrieval_hints import (
    corpus_available_for_chat,
    query_answer_from_chat_skip_retrieval,
)
from config import settings
from core.llm_provider import get_chat_model

logger = logging.getLogger(__name__)

_SYSTEM = """You are CogniFlow, a precise assistant grounded in retrieved documentation and conversation context.
If sources are provided, cite them implicitly (filename/title) and do not invent APIs not in the sources.
If no sources apply, answer briefly from general knowledge and say when uncertain."""

_SYSTEM_SESSION_DOCS = """You are CogniFlow. This user has technical documents indexed for **this chat and/or their other chats** (retrieval is scoped to this session plus their account).
Ground factual answers in the "Retrieved sources" below when they are relevant. Do not invent product names, companies, or document topics that do not appear in those sources or the user's words.
For short greetings or introductions, reply warmly in one or two sentences and invite the user to ask about their materials.
If the question needs facts but the sources are empty or not relevant, say you could not find that in their indexed documents and suggest rephrasing or uploading more context."""

_SYSTEM_CHAT_AND_DOCS = """You are CogniFlow for this chat.
- If the user asks about **their name**, **document/file names**, what **they uploaded**, what **they said earlier**, or other **conversation-only** facts, answer from **Recent messages** (and user context below). Do not say that information is missing from "documents" when it appears in the conversation.
- If the user asks about **content inside** uploaded files, ground answers in **Retrieved sources** when relevant.
- If retrieved sources are weak or empty but the question is only about the conversation, ignore document retrieval and answer from the chat history."""


def _chunk_relevance(doc: dict[str, Any]) -> float:
    """Match api.routes._doc_distance_to_relevance — higher is better."""
    d = doc.get("distance")
    if d is None:
        return 0.0
    try:
        x = float(d)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, 1.0 - x))


def _format_docs(docs: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, d in enumerate(docs[:8], 1):
        meta = d.get("metadata") or {}
        src = meta.get("source") or meta.get("title") or "source"
        body = (d.get("content") or "")[:4000]
        lines.append(f"[{i}] {src}\n{body}")
    return "\n\n".join(lines) if lines else "(no documents retrieved)"


def _synthesis_prompts(state: CogniFlowState) -> tuple[str, str, list[dict[str, Any]], str | None]:
    """Return (system_prompt, user_block, good_docs, fast_refuse_message_or_none)."""
    sid = (state.get("session_id") or "").strip()
    uid = (state.get("user_id") or "").strip()
    has_user_corpus = bool(sid) and corpus_available_for_chat(sid, uid)
    history = state.get("conversation_history") or []
    hist_snip = "\n".join(
        f'{m.get("role", "user")}: {m.get("content", "")}'
        for m in history[-12:]
    )
    docs = state.get("retrieved_documents") or []
    min_rel = float(settings.retrieval_min_relevance)
    good_docs = [
        d for d in docs if isinstance(d, dict) and _chunk_relevance(d) >= min_rel
    ]

    user_q = str(state.get("user_query") or "")
    conv_first = query_answer_from_chat_skip_retrieval(user_q)

    if has_user_corpus and not good_docs and not conv_first:
        msg = (
            "I couldn't find anything in your indexed documents that matches this question "
            "(the closest matches weren't relevant enough). Try rephrasing, or ask about "
            "something covered in your files."
        )
        return "", "", [], msg

    if conv_first and has_user_corpus:
        system = _SYSTEM_CHAT_AND_DOCS
    elif has_user_corpus:
        system = _SYSTEM_SESSION_DOCS
    else:
        system = _SYSTEM
    summary = state.get("conversation_summary") or ""
    mem_ctx = (state.get("user_memory_context") or "").strip()
    peer_ctx = (state.get("cross_session_context") or "").strip()
    doc_block = _format_docs(good_docs)
    mem_block = f"Known user preferences/context (cross-session):\n{mem_ctx}\n\n" if mem_ctx else ""
    peer_block = f"Other recent chats (short summaries):\n{peer_ctx}\n\n" if peer_ctx else ""
    user_block = (
        f"Conversation summary (if any): {summary or '(none)'}\n\n"
        f"{mem_block}"
        f"{peer_block}"
        f"Recent messages:\n{hist_snip or '(empty)'}\n\n"
        f"Retrieved sources:\n{doc_block}\n\n"
        f"User question:\n{state.get('user_query', '')}"
    )
    return system, user_block, good_docs, None


def context_synthesis_node(state: CogniFlowState) -> dict[str, Any]:
    """Produce the assistant reply from history, memory, and retrieved chunks."""
    t0 = time.perf_counter()
    system, user_block, good_docs, fast_refuse = _synthesis_prompts(state)

    if fast_refuse is not None:
        min_rel = float(settings.retrieval_min_relevance)
        raw_n = len(state.get("retrieved_documents") or [])
        logger.info(
            "context_synthesis: fast_refuse (no chunks >= min_relevance=%s, raw_n=%s)",
            min_rel,
            raw_n,
        )
        return {
            "response": fast_refuse,
            "synthesized_context": "",
            "retrieved_documents": [],
            "agent_log": [
                with_log_timing(
                    {
                        "node": "context_synthesis",
                        "fast_refuse": True,
                        "retrieval_min_relevance": min_rel,
                        "raw_retrieved": raw_n,
                    },
                    t0,
                )
            ],
        }

    model = get_chat_model()
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user_block),
    ]
    resp = model.invoke(messages)
    text = (getattr(resp, "content", None) or str(resp)).strip()
    log_entry = with_log_timing(
        {
            "node": "context_synthesis",
            "response_chars": len(text),
            "intent": state.get("query_intent"),
        },
        t0,
    )
    return {
        "response": text,
        "synthesized_context": _format_docs(good_docs)[:8000],
        "retrieved_documents": good_docs,
        "agent_log": [log_entry],
    }


def iter_context_synthesis_events(state: CogniFlowState) -> Iterator[dict[str, Any]]:
    """Stream LLM tokens for synthesis, then emit a final ``complete`` event (mirrors ``context_synthesis_node``)."""
    t0 = time.perf_counter()
    system, user_block, good_docs, fast_refuse = _synthesis_prompts(state)

    if fast_refuse is not None:
        min_rel = float(settings.retrieval_min_relevance)
        raw_n = len(state.get("retrieved_documents") or [])
        log_entry = with_log_timing(
            {
                "node": "context_synthesis",
                "fast_refuse": True,
                "retrieval_min_relevance": min_rel,
                "raw_retrieved": raw_n,
            },
            t0,
        )
        yield {
            "type": "complete",
            "response": fast_refuse,
            "synthesized_context": "",
            "retrieved_documents": [],
            "agent_log": [log_entry],
        }
        return

    model = get_chat_model()
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user_block),
    ]
    full_response: list[str] = []
    for chunk in model.stream(messages):
        token = getattr(chunk, "content", "") or ""
        if isinstance(token, list):
            token = "".join(str(part) for part in token)
        if token:
            full_response.append(token)
            yield {"type": "token", "data": token}

    text = "".join(full_response).strip()
    log_entry = with_log_timing(
        {
            "node": "context_synthesis",
            "response_chars": len(text),
            "intent": state.get("query_intent"),
        },
        t0,
    )
    yield {
        "type": "complete",
        "response": text,
        "synthesized_context": _format_docs(good_docs)[:8000],
        "retrieved_documents": good_docs,
        "agent_log": [log_entry],
    }

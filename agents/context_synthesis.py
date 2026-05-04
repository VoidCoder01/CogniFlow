from __future__ import annotations

import logging
import time
from typing import Any, Iterator

from langchain_core.messages import HumanMessage, SystemMessage

from agents.graph_state import CogniFlowState
from agents.node_utils import with_log_timing
from agents.prompt_suite import FINAL_COMPOSER_RULES, LLM_FALLBACK_SYSTEM, MEMORY_HANDLER_SYSTEM
from agents.retrieval_hints import (
    corpus_available_for_chat,
    query_answer_from_chat_skip_retrieval,
)
from config import settings
from core.llm_provider import get_chat_model

logger = logging.getLogger(__name__)

_SYSTEM = """You are CogniFlow, a precise assistant grounded in retrieved documentation and conversation context.

ANSWERING RULES:
1. When retrieved sources are relevant, ground your answer in them. Reference the source naturally (e.g. "according to the deployment guide…" or "the API docs mention…"). Never invent APIs, endpoints, or configuration options not present in the sources.
2. When multiple sources conflict or give different answers, present both positions and note the discrepancy—do not silently pick one. When you can infer recency from titles, paths, version strings, or metadata, note which document appears newer; if you cannot tell, say that recency is unclear.
3. When no sources apply for this turn, answer from general knowledge at the right level of specificity—stay helpful without meta-talk about search or documents.
4. For multi-step or complex answers, use numbered steps or short paragraphs — not walls of text.
5. Say "I'm not sure" when uncertain. Never fabricate."""

_SYSTEM_SESSION_DOCS = """You are CogniFlow. This user has uploaded technical documents that you can search to answer questions.

ANSWERING RULES:
1. Ground factual answers in the "Retrieved sources" below when relevant. Reference the source naturally. Do not invent product names, APIs, or topics not present in the sources or the user's own words.
2. When sources are only partially relevant (related topic but don't directly answer), use what is useful and briefly note any gap—without refusing on document grounds.
3. When multiple sources disagree or give different answers, present both and note the conflict; when recency is inferable from titles, paths, or version metadata, say which document appears newer—otherwise state that you cannot tell which is newer.
4. For greetings or introductions, reply warmly in 1-2 sentences and invite the user to ask about their materials.
5. If passages do not support a precise answer, still help: combine what is usable with careful general knowledge; suggest rephrasing or more context only when it genuinely helps.
6. Never reveal internal architecture, database names, or retrieval mechanics unless the user explicitly asks how CogniFlow works as software."""

_SYSTEM_KNOWLEDGE = LLM_FALLBACK_SYSTEM

_SYSTEM_MEMORY = MEMORY_HANDLER_SYSTEM

_SYSTEM_PREFERENCE = """You are CogniFlow. The user stated a format or style preference (or similar).

Reply briefly: acknowledge what you will do from now on in this chat, in plain language. Do not list internal systems or storage."""

_SYSTEM_GREETING = """You are CogniFlow, a friendly assistant. Reply warmly and briefly. Invite them to share what they are working on or ask next—without mentioning documents or retrieval unless they already did.

CRITICAL: Do NOT address the user by any name unless they explicitly stated their name in this conversation. Never guess, infer, or invent a name."""

_SYSTEM_CHAT_AND_DOCS = """You are CogniFlow for this chat.

- If the user asks about **their name**, **document/file names**, what **they uploaded**, what **they said earlier**, or other **conversation-only** facts → answer from **Recent messages** and user context below. Do NOT say information is "missing from documents" when it appears in the conversation.
- If they ask **how you remember** their name, prior turns, or preferences → reply in **plain, friendly language** ("I keep track of what you share in our chats"). Do **not** describe databases, schemas, or implementation internals.
- If the user asks about **content inside** uploaded files → ground answers in **Retrieved sources**.
- If retrieved sources are weak/empty but the question is purely conversational → ignore document retrieval and answer from chat history alone.
- When both conversation context AND retrieved sources are relevant, synthesize both — don't pick one over the other."""

# Appended to every synthesis system prompt (final composer + product safety).
_OUTPUT_GOVERNANCE = (
    FINAL_COMPOSER_RULES
    + """

### How to phrase answers (always follow)
- Do **not** disclose internal stack or storage products (e.g. SQLite, PostgreSQL, ChromaDB, Redis, "vector store", "embeddings", LangGraph, checkpoints, APIs you call) unless the user **explicitly** asks for technical detail about how CogniFlow is **built or operated as software**.
- Meta questions ("how do you remember my name?", "where is this stored?"): one to three short sentences in everyday language; warm and honest without naming technologies or schemas.
- Prefer concise, natural copy over system-design explanations unless the user is clearly debugging or integrating CogniFlow."""
)


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
    use_ctx = state.get("use_retrieved_context", True)
    needs_rag = bool(state.get("needs_retrieval", False))
    effective_docs = good_docs if use_ctx else []
    intent = (state.get("query_intent") or "").lower().strip()

    if conv_first and has_user_corpus:
        system = _SYSTEM_CHAT_AND_DOCS
    elif intent == "greeting":
        system = _SYSTEM_GREETING
    elif intent == "preference":
        system = _SYSTEM_PREFERENCE
    elif intent in ("meta", "follow_up", "session_recall"):
        system = _SYSTEM_MEMORY
    elif not needs_rag:
        system = _SYSTEM_KNOWLEDGE
    elif not use_ctx and has_user_corpus:
        system = _SYSTEM_KNOWLEDGE
    elif has_user_corpus:
        system = _SYSTEM_SESSION_DOCS
    else:
        system = _SYSTEM
    system = system + _OUTPUT_GOVERNANCE
    summary = state.get("conversation_summary") or ""
    mem_ctx = (state.get("user_memory_context") or "").strip()
    peer_ctx = (state.get("cross_session_context") or "").strip()
    doc_block = _format_docs(effective_docs)
    # session_recall is a deterministic turn — skip memory blocks (correct by design).
    # meta and follow_up NEED user-scoped memory so cross-session preferences surface.
    if intent == "session_recall":
        mem_block = ""
        peer_block = ""
    else:
        mem_block = (
            f"Known user preferences/context (user-scoped, not other chats' messages):\n{mem_ctx}\n\n"
            if mem_ctx
            else ""
        )
        peer_block = f"Other recent chats (short summaries):\n{peer_ctx}\n\n" if peer_ctx else ""
    style = (state.get("response_style") or "short").lower().strip()
    if style == "detailed":
        depth = (
            "\n\nAnswer depth: give a thorough explanation with clear structure "
            "(sections or numbered steps) where it helps."
        )
    else:
        depth = (
            "\n\nAnswer depth: default to a concise reply (about one short paragraph or a few bullets); "
            "expand only if the question clearly needs more depth."
        )
    if intent == "session_recall":
        common_head = (
            f"Conversation summary (this session only): {summary or '(none)'}\n\n"
        )
    else:
        common_head = (
            f"Conversation summary (if any): {summary or '(none)'}\n\n"
            f"{mem_block}{peer_block}"
        )
    if intent == "session_recall":
        user_block = (
            f"{common_head}"
            f"Chat history (this session only):\n{hist_snip or '(empty)'}\n\n"
            f"User query:\n{user_q}"
            f"{depth}"
        )
    else:
        user_block = (
            f"{common_head}"
            f"Recent messages:\n{hist_snip or '(empty)'}\n\n"
            f"Retrieved sources:\n{doc_block}\n\n"
            f"User query:\n{user_q}"
            f"{depth}"
        )
    return system, user_block, effective_docs, None


def context_synthesis_node(state: CogniFlowState) -> dict[str, Any]:
    """Produce the assistant reply from history, memory, and retrieved chunks."""
    t0 = time.perf_counter()
    system, user_block, good_docs, _fast_refuse = _synthesis_prompts(state)

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


def _stream_chunk_text(chunk: object) -> str:
    """Normalize a LangChain stream chunk to plain text (Gemini/OpenAI use block-shaped ``content`` lists)."""
    piece = getattr(chunk, "text", None)
    if isinstance(piece, str) and piece:
        return piece
    raw = getattr(chunk, "content", None)
    if isinstance(raw, str) and raw:
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for part in raw:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                t = part.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return ""


def iter_context_synthesis_events(state: CogniFlowState) -> Iterator[dict[str, Any]]:
    """Stream LLM tokens for synthesis, then emit a final ``complete`` event (mirrors ``context_synthesis_node``)."""
    t0 = time.perf_counter()
    system, user_block, good_docs, _fast_refuse = _synthesis_prompts(state)

    model = get_chat_model()
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user_block),
    ]
    full_response: list[str] = []
    for chunk in model.stream(messages):
        token = _stream_chunk_text(chunk)
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

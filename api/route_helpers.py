"""Pure helpers for chat / SSE / retrieval formatting (importable without FastAPI multipart)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterator

from config import settings
from core.memory_store import MemoryStore
from core.models import ChatMessage


SSE_TOKEN_CHUNK_CHARS = 14


def sse_token_event_chunks(text: str) -> Iterator[bytes]:
    t = text or ""
    for i in range(0, len(t), SSE_TOKEN_CHUNK_CHARS):
        piece = t[i : i + SSE_TOKEN_CHUNK_CHARS]
        line = json.dumps({"event": "token", "data": piece}, ensure_ascii=False)
        yield f"data: {line}\n\n".encode()


def doc_distance_to_relevance(distance: float | None) -> float:
    if distance is None:
        return 0.0
    try:
        return max(0.0, min(1.0, 1.0 - float(distance)))
    except (TypeError, ValueError):
        return 0.0


def sources_from_retrieved(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in docs[:10]:
        if not isinstance(d, dict):
            continue
        meta = d.get("metadata") or {}
        out.append(
            {
                "id": d.get("id"),
                "title": meta.get("title"),
                "source": meta.get("source"),
                "original_filename": meta.get("original_filename"),
                "doc_instance_id": meta.get("doc_instance_id"),
                "relevance": doc_distance_to_relevance(d.get("distance")),
            }
        )
    return out


def format_user_memory_context(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines: list[str] = []
    for r in rows:
        mt = r.get("memory_type") or "context"
        content = (r.get("content") or "").strip()
        if content:
            lines.append(f"- ({mt}) {content}")
    return "\n".join(lines)


def format_peer_session_context(rows: list[dict[str, Any]]) -> str:
    """Compact lines for other chats' rolling summaries (cross-session awareness)."""
    if not rows:
        return ""
    lines: list[str] = []
    for r in rows:
        sid = str(r.get("session_id") or "")[:8]
        s = (r.get("summary") or "").strip().replace("\n", " ")
        if s:
            lines.append(f"- [{sid}…] {s[:650]}")
    return "\n".join(lines)


def cross_session_context_block(
    store: MemoryStore, user_id: str, session_id: str
) -> str:
    rows = store.get_peer_session_summaries(user_id, session_id, limit=6)
    return format_peer_session_context(rows)


def append_pipeline_timing(agent_log: list[dict[str, Any]]) -> None:
    """Append a rollup row: sum of per-node ``elapsed_seconds`` (excludes orchestrator wall clock)."""
    timed_sum = 0.0
    n = 0
    for x in agent_log:
        if not isinstance(x, dict):
            continue
        if x.get("node") in ("pipeline", "orchestrator"):
            continue
        es = x.get("elapsed_seconds")
        if es is not None:
            timed_sum += float(es)
            n += 1
    agent_log.append(
        {
            "node": "pipeline",
            "timed_node_steps": n,
            "elapsed_seconds_sum_nodes": round(timed_sum, 4),
        }
    )


def response_cache_context_fp(summary: str, history: list[ChatMessage]) -> str:
    """Stable fingerprint so cached replies respect conversation thread (optional)."""
    if not getattr(settings, "chat_response_cache_include_context", False):
        return ""
    n = max(0, int(getattr(settings, "chat_response_cache_context_messages", 16)))
    tail = history[-n:] if n else []
    lines = [f"{m.role.value}:{(m.content or '').strip()}" for m in tail]
    blob = ((summary or "").strip() + "\n" + "\n".join(lines)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

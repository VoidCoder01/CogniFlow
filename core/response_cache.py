"""
Response cache for /chat and /chat/stream.

- **exact**: same normalized text (+ optional conversation fingerprint) → fast lookup.
- **semantic**: cosine similarity between query embeddings (same session + user + fingerprint).
- **both** (default): exact first, then semantic.

**Backends**: ``memory`` (ephemeral) or ``sqlite`` (default — survives API restarts / Docker).

Cleared when new documents are indexed for that session.
"""

from __future__ import annotations

import hashlib
import math
import threading
import uuid
from collections import OrderedDict
from typing import Any, Literal

from config import settings

_lock = threading.Lock()
# entry_id -> payload (includes session_id, user_id, context_fp, embedding, cached fields)
_entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
# exact composite key -> entry_id
_exact_index: dict[str, str] = {}

_emb_manager_instance: Any | None = None


def _get_embedding_manager():
    global _emb_manager_instance
    if _emb_manager_instance is None:
        from core.embeddings import EmbeddingManager

        _emb_manager_instance = EmbeddingManager()
    return _emb_manager_instance


def _normalize_message(message: str) -> str:
    m = (message or "").strip()
    if settings.chat_message_cache_normalize_whitespace:
        return " ".join(m.split())
    return m


def _norm_hash(message: str) -> str:
    norm = _normalize_message(message)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _exact_key(session_id: str, user_id: str, context_fp: str, message: str) -> str:
    sid = (session_id or "").strip()
    uid = (user_id or "").strip()
    ctx = (context_fp or "").strip()
    nh = _norm_hash(message)
    return f"{sid}\x00{uid}\x00{ctx}\x00{nh}"


def _l2_normalize(v: list[float]) -> list[float]:
    s = math.sqrt(sum(x * x for x in v))
    if s < 1e-12:
        return v
    return [x / s for x in v]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return -1.0
    an = _l2_normalize(a)
    bn = _l2_normalize(b)
    return sum(x * y for x, y in zip(an, bn))


def _cache_enabled() -> bool:
    return bool(settings.chat_exact_message_cache_enabled)


def _mode() -> Literal["exact", "semantic", "both"]:
    m = (getattr(settings, "chat_response_cache_mode", None) or "both").lower().strip()
    if m in ("exact", "semantic", "both"):
        return m  # type: ignore[return-value]
    return "both"


def _sqlite_path() -> str:
    return (
        getattr(settings, "chat_response_cache_sqlite_path", None)
        or "./data/response_cache.db"
    )


def _use_sqlite_backend() -> bool:
    b = (getattr(settings, "chat_response_cache_backend", None) or "sqlite").lower().strip()
    return b == "sqlite"


def _payload_from_entry(ent: dict[str, Any]) -> dict[str, Any]:
    return {
        "response": ent.get("response", ""),
        "sources": list(ent.get("sources") or []),
        "conversation_summary": ent.get("conversation_summary") or "",
    }


def get_cached(
    session_id: str,
    user_id: str,
    message: str,
    *,
    context_fp: str = "",
) -> dict[str, Any] | None:
    if not _cache_enabled():
        return None
    sid = (session_id or "").strip()
    uid = (user_id or "").strip()
    ctx = (context_fp or "").strip()
    norm = _normalize_message(message)
    nh = _norm_hash(message)
    ek = _exact_key(session_id, user_id, ctx, message)
    mode = _mode()

    if _use_sqlite_backend():
        from core.response_cache_sqlite import get_exact_row, semantic_best

        if mode in ("exact", "both"):
            row = get_exact_row(_sqlite_path(), sid, uid, ctx, nh)
            if row is not None:
                return row

        if mode in ("semantic", "both"):
            try:
                qraw = _get_embedding_manager().embed_text(norm)
                qvec = _l2_normalize(list(qraw)) if qraw else []
            except Exception:
                return None
            if not qvec:
                return None
            thr = float(getattr(settings, "chat_response_cache_min_similarity", 0.82))
            hit = semantic_best(
                _sqlite_path(),
                sid,
                uid,
                ctx,
                qvec,
                thr,
                _cosine_similarity,
            )
            if hit is not None:
                out, sim = hit
                out["_cache_match"] = {"mode": "semantic", "similarity": round(sim, 5)}
                return out
        return None

    with _lock:
        if mode in ("exact", "both"):
            eid = _exact_index.get(ek)
            if eid and eid in _entries:
                _entries.move_to_end(eid)
                return _payload_from_entry(_entries[eid])

        if mode in ("semantic", "both"):
            try:
                qraw = _get_embedding_manager().embed_text(norm)
                qvec = _l2_normalize(list(qraw)) if qraw else []
            except Exception:
                return None
            if not qvec:
                return None
            thr = float(getattr(settings, "chat_response_cache_min_similarity", 0.82))
            best_eid: str | None = None
            best_sim = -1.0
            for eid, ent in _entries.items():
                if (
                    ent.get("session_id") != sid
                    or ent.get("user_id") != uid
                    or (ent.get("context_fp") or "") != ctx
                ):
                    continue
                emb = ent.get("embedding")
                if not isinstance(emb, list):
                    continue
                sim = _cosine_similarity(qvec, emb)
                if sim > best_sim:
                    best_sim = sim
                    best_eid = eid
            if best_eid is not None and best_sim >= thr:
                _entries.move_to_end(best_eid)
                out = _payload_from_entry(_entries[best_eid])
                out["_cache_match"] = {"mode": "semantic", "similarity": round(best_sim, 5)}
                return out

    return None


def put_cached(
    session_id: str,
    user_id: str,
    message: str,
    *,
    response: str,
    sources: list[dict[str, Any]],
    conversation_summary: str,
    context_fp: str = "",
) -> None:
    if not _cache_enabled():
        return
    sid = (session_id or "").strip()
    uid = (user_id or "").strip()
    ctx = (context_fp or "").strip()
    norm = _normalize_message(message)
    nh = _norm_hash(message)
    ek = _exact_key(session_id, user_id, ctx, message)
    max_n = max(16, settings.chat_exact_message_cache_max_entries)

    try:
        raw = _get_embedding_manager().embed_text(norm)
        emb = _l2_normalize(list(raw)) if raw else []
    except Exception:
        emb = []

    if _use_sqlite_backend():
        from core.response_cache_sqlite import upsert_row

        upsert_row(
            _sqlite_path(),
            session_id=sid,
            user_id=uid,
            context_fp=ctx,
            norm_hash=nh,
            embedding=emb,
            response=response or "",
            sources=list(sources or []),
            conversation_summary=conversation_summary or "",
            query_preview=norm[:240],
            max_rows=max_n,
        )
        return

    payload = {
        "session_id": sid,
        "user_id": uid,
        "context_fp": ctx,
        "embedding": emb,
        "query_preview": norm[:240],
        "response": response,
        "sources": list(sources or []),
        "conversation_summary": conversation_summary or "",
    }

    with _lock:
        if ek in _exact_index:
            eid = _exact_index[ek]
            if eid in _entries:
                _entries[eid].update(payload)
                _entries.move_to_end(eid)
                _trim(max_n)
                return

        eid = str(uuid.uuid4())
        _entries[eid] = payload
        _exact_index[ek] = eid
        _trim(max_n)


def _trim(max_n: int) -> None:
    while len(_entries) > max_n:
        old_id, _ = _entries.popitem(last=False)
        for k, v in list(_exact_index.items()):
            if v == old_id:
                del _exact_index[k]


def invalidate_session(session_id: str) -> None:
    """Drop all cache entries for a chat session (e.g. after new docs indexed)."""
    sid = (session_id or "").strip()
    if not sid:
        return
    if _use_sqlite_backend():
        from core.response_cache_sqlite import invalidate_session as sqlite_invalidate

        sqlite_invalidate(_sqlite_path(), sid)
        return
    with _lock:
        dead_eids = [
            eid for eid, ent in _entries.items() if ent.get("session_id") == sid
        ]
        for eid in dead_eids:
            del _entries[eid]
        for k, v in list(_exact_index.items()):
            if v in dead_eids:
                del _exact_index[k]


def clear_for_tests() -> None:
    with _lock:
        _entries.clear()
        _exact_index.clear()
    global _emb_manager_instance
    _emb_manager_instance = None
    if _use_sqlite_backend():
        try:
            from core.response_cache_sqlite import truncate_table

            truncate_table(_sqlite_path())
        except Exception:
            pass

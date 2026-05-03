from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import TYPE_CHECKING

from fastapi import HTTPException

from agents.orchestrator import CogniFlowOrchestrator
from config import settings
from core.memory_store import MemoryStore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.vector_store import VectorStore

_dep_vector_store: VectorStore | None = None
_dep_vector_store_error: str | None = None
_dep_vector_store_fail_monotonic: float = 0.0
# After a failed Chroma open, retry after this many seconds (e.g. user deleted ./data/chroma without restarting uvicorn).
_VECTOR_STORE_RETRY_COOLDOWN_SEC = 10.0


@lru_cache(maxsize=1)
def get_memory_store() -> MemoryStore:
    return MemoryStore()


def get_vector_store() -> "VectorStore":
    """Return the process-wide ``VectorStore``, or raise ``HTTPException(503)`` if Chroma cannot open."""
    global _dep_vector_store, _dep_vector_store_error, _dep_vector_store_fail_monotonic
    if _dep_vector_store is not None:
        return _dep_vector_store

    from core.vector_store import ChromaPersistenceError, VectorStore

    now = time.monotonic()
    if _dep_vector_store_error is not None:
        if now - _dep_vector_store_fail_monotonic < _VECTOR_STORE_RETRY_COOLDOWN_SEC:
            raise HTTPException(status_code=503, detail=_dep_vector_store_error)
        logger.info(
            "Retrying VectorStore open after cooldown (catalog may have been reset on disk)"
        )
        _dep_vector_store_error = None

    try:
        _dep_vector_store = VectorStore()
        try:
            from agents.retrieval_router import reset_retrieval_vector_store

            reset_retrieval_vector_store()
        except ImportError:
            pass
        return _dep_vector_store
    except ChromaPersistenceError as exc:
        _dep_vector_store_error = str(exc)
        _dep_vector_store_fail_monotonic = time.monotonic()
        logger.exception("Chroma catalog unreadable")
        raise HTTPException(status_code=503, detail=_dep_vector_store_error) from exc
    except Exception as exc:
        _dep_vector_store_error = (
            f"Vector store failed to initialize: {exc}. "
            f"If you upgraded ChromaDB, remove `{settings.chroma_persist_dir}` "
            "and re-ingest documents."
        )
        _dep_vector_store_fail_monotonic = time.monotonic()
        logger.exception("VectorStore initialization failed")
        raise HTTPException(status_code=503, detail=_dep_vector_store_error) from exc


def peek_vector_store() -> tuple["VectorStore | None", str | None]:
    """Return ``(store, None)`` or ``(None, detail)`` without raising (for degraded ``/stats``)."""
    if _dep_vector_store is not None:
        return _dep_vector_store, None
    try:
        return get_vector_store(), None
    except HTTPException as e:
        d = e.detail
        detail = d if isinstance(d, str) else str(d)
        return None, detail


@lru_cache(maxsize=1)
def get_orchestrator() -> CogniFlowOrchestrator:
    return CogniFlowOrchestrator()


def clear_app_caches() -> None:
    """Used by tests to reset singletons."""
    global _dep_vector_store, _dep_vector_store_error, _dep_vector_store_fail_monotonic
    _dep_vector_store = None
    _dep_vector_store_error = None
    _dep_vector_store_fail_monotonic = 0.0
    get_memory_store.cache_clear()
    get_orchestrator.cache_clear()
    try:
        from agents.retrieval_router import reset_retrieval_vector_store

        reset_retrieval_vector_store()
    except ImportError:
        pass

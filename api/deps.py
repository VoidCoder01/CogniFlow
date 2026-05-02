from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from agents.orchestrator import CogniFlowOrchestrator
from core.memory_store import MemoryStore

if TYPE_CHECKING:
    from core.vector_store import VectorStore


@lru_cache(maxsize=1)
def get_memory_store() -> MemoryStore:
    return MemoryStore()


@lru_cache(maxsize=1)
def get_vector_store() -> "VectorStore":
    from core.vector_store import VectorStore

    return VectorStore()


@lru_cache(maxsize=1)
def get_orchestrator() -> CogniFlowOrchestrator:
    return CogniFlowOrchestrator()


def clear_app_caches() -> None:
    """Used by tests to reset singletons."""
    get_memory_store.cache_clear()
    get_vector_store.cache_clear()
    get_orchestrator.cache_clear()

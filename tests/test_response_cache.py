from __future__ import annotations

import pytest

from config import settings
from core.response_cache import (
    clear_for_tests,
    get_cached,
    invalidate_session,
    put_cached,
)
import core.response_cache as response_cache


@pytest.fixture(autouse=True)
def _clear_response_cache(monkeypatch):
    """Isolate tests from SQLite persistence and avoid wiping dev ./data caches."""
    monkeypatch.setattr(settings, "chat_response_cache_backend", "memory")
    clear_for_tests()
    yield
    clear_for_tests()


def test_response_cache_roundtrip():
    put_cached(
        "sess-a",
        "user-1",
        "What is X?",
        response="Answer",
        sources=[{"title": "t"}],
        conversation_summary="",
    )
    g = get_cached("sess-a", "user-1", "What is X?")
    assert g is not None
    assert g["response"] == "Answer"
    assert g["sources"][0]["title"] == "t"


def test_response_cache_whitespace_normalized():
    put_cached(
        "s",
        "u",
        "hello   world",
        response="R",
        sources=[],
        conversation_summary="",
    )
    g = get_cached("s", "u", "hello world")
    assert g is not None
    assert g["response"] == "R"


def test_invalidate_session():
    put_cached("s1", "u", "q", response="a", sources=[], conversation_summary="")
    assert get_cached("s1", "u", "q") is not None
    invalidate_session("s1")
    assert get_cached("s1", "u", "q") is None


def test_semantic_cache_paraphrase(monkeypatch):
    """Same embedding for any text → high similarity → hit without identical string."""
    monkeypatch.setattr(settings, "chat_response_cache_mode", "semantic")
    monkeypatch.setattr(settings, "chat_response_cache_min_similarity", 0.5)
    dim = 8
    vec = [1.0 / (dim**0.5)] * dim

    class FakeEmb:
        def embed_text(self, text: str):
            return list(vec)

    monkeypatch.setattr(response_cache, "_get_embedding_manager", lambda: FakeEmb())

    put_cached(
        "s2",
        "u2",
        "List six agents and HTTP endpoints and Gemini defaults",
        response="Cached body",
        sources=[],
        conversation_summary="",
    )
    hit = get_cached(
        "s2",
        "u2",
        "What is the default LLM provider for Gemini?",
    )
    assert hit is not None
    assert hit["response"] == "Cached body"
    assert hit.get("_cache_match", {}).get("mode") == "semantic"


def test_sqlite_backend_roundtrip(monkeypatch, tmp_path):
    """SQLite backend writes ``response_cache.db`` and serves exact lookups."""
    from core.response_cache_sqlite import truncate_table

    db = tmp_path / "rc.db"
    monkeypatch.setattr(settings, "chat_response_cache_backend", "sqlite")
    monkeypatch.setattr(settings, "chat_response_cache_sqlite_path", str(db))

    class FakeEmb:
        def embed_text(self, text: str):
            return [0.0, 1.0, 0.0]

    monkeypatch.setattr(response_cache, "_get_embedding_manager", lambda: FakeEmb())
    truncate_table(str(db))

    put_cached("s-db", "u-db", "Stored via SQLite?", response="yes", sources=[], conversation_summary="")
    assert db.exists()
    hit = get_cached("s-db", "u-db", "Stored via SQLite?")
    assert hit is not None
    assert hit["response"] == "yes"
    truncate_table(str(db))

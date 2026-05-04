"""Unit tests for ``api.routes`` helpers (high-risk path coverage)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.memory_store import MemoryStore
from core.models import ChatMessage, MessageRole


def test_sse_token_chunks():
    from api.route_helpers import sse_token_event_chunks

    parts = list(sse_token_event_chunks("abcdefghijklm"))
    assert len(parts) >= 1
    joined = b"".join(parts).decode()
    assert "token" in joined and "abcd" in joined


def test_doc_distance_to_relevance():
    from api.route_helpers import doc_distance_to_relevance

    assert doc_distance_to_relevance(None) == 0.0
    assert doc_distance_to_relevance("bad") == 0.0
    assert doc_distance_to_relevance(0.2) == pytest.approx(0.8, rel=1e-3)


def test_sources_from_retrieved_skips_non_dicts():
    from api.route_helpers import sources_from_retrieved

    docs = [
        {"id": "1", "content": "x", "metadata": {"title": "T"}, "distance": 0.1},
        "not-a-dict",
        {"id": "2", "content": "y", "metadata": {}, "distance": None},
    ]
    out = sources_from_retrieved(docs)
    assert len(out) == 2
    assert out[0]["relevance"] == pytest.approx(0.9, rel=1e-3)
    assert out[1]["relevance"] == 0.0


def test_format_user_memory_context_and_peer():
    from api.route_helpers import format_peer_session_context, format_user_memory_context

    assert format_user_memory_context([]) == ""
    rows = [
        {"memory_type": "preference", "content": "  bullets  "},
        {"memory_type": "context", "content": ""},
    ]
    u = format_user_memory_context(rows)
    assert "preference" in u and "bullets" in u

    assert format_peer_session_context([]) == ""
    peer = format_peer_session_context(
        [{"session_id": "abc12345", "summary": "line1\nline2"}]
    )
    assert "abc12345" in peer and "line1 line2" in peer


def test_append_pipeline_timing():
    from api.route_helpers import append_pipeline_timing

    log: list = [
        {"node": "query_understanding", "elapsed_seconds": 0.1},
        {"node": "pipeline", "elapsed_seconds": 99.0},
        {"node": "orchestrator", "elapsed_seconds": 1.0},
        {"node": "retrieval_router", "elapsed_seconds": 0.2},
    ]
    append_pipeline_timing(log)
    tail = log[-1]
    assert tail["node"] == "pipeline"
    assert tail["timed_node_steps"] == 2
    assert tail["elapsed_seconds_sum_nodes"] == pytest.approx(0.3, rel=1e-3)


def test_response_cache_context_fp_respects_settings(monkeypatch):
    from api.route_helpers import response_cache_context_fp

    monkeypatch.setattr(
        "api.route_helpers.settings", MagicMock(chat_response_cache_include_context=False)
    )
    m = ChatMessage(role=MessageRole.user, content="x", id="1")
    assert response_cache_context_fp("sum", [m]) == ""

    monkeypatch.setattr(
        "api.route_helpers.settings",
        MagicMock(
            chat_response_cache_include_context=True,
            chat_response_cache_context_messages=2,
        ),
    )
    fp1 = response_cache_context_fp("s", [m])
    assert len(fp1) == 64
    m2 = ChatMessage(role=MessageRole.assistant, content="y", id="2")
    fp2 = response_cache_context_fp("s", [m, m2])
    assert fp1 != fp2


def test_cross_session_context_block(tmp_path):
    from api.route_helpers import cross_session_context_block

    store = MemoryStore(db_path=str(tmp_path / "m.db"))
    u = "same-user"
    s1 = store.create_session(u)
    s2 = store.create_session(u)
    store.update_session_summary(s2.session_id, "Other chat about deploy.")
    block = cross_session_context_block(store, u, s1.session_id)
    assert "deploy" in block
    assert s2.session_id[:8] in block

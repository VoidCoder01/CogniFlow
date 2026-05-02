"""Tests for VectorStore with a temporary Chroma directory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.models import DocumentChunk, DocumentMetadata
from core.vector_store import VectorStore


@pytest.fixture
def vstore(tmp_path):
    with patch("core.vector_store.EmbeddingManager") as MockEmb:
        mock_emb = MockEmb.return_value
        mock_emb.embed_text.return_value = [0.1] * 384
        mock_emb.embed_texts.return_value = [[0.1] * 384]
        vs = VectorStore(persist_dir=str(tmp_path / "chroma"), collection_name="test")
        yield vs


def test_add_and_search(vstore):
    chunk = DocumentChunk(
        content="FastAPI is a modern Python web framework.",
        metadata=DocumentMetadata(
            source="test.md",
            doc_type="markdown",
            title="Test",
            session_id="s1",
            user_id="u1",
        ),
    )
    vstore.add_documents([chunk])
    results = vstore.semantic_search("FastAPI framework", top_k=3)
    assert len(results) >= 1


def test_scope_filter():
    f = VectorStore.scope_filter("s1", "u1")
    assert "$or" in f
    f2 = VectorStore.scope_filter("s1", "")
    assert f2 == {"session_id": "s1"}
    f3 = VectorStore.scope_filter("", "")
    assert f3 is None


def test_hybrid_search(vstore):
    chunk = DocumentChunk(
        content="Docker containers use namespaces for isolation.",
        metadata=DocumentMetadata(
            source="docker.md",
            doc_type="markdown",
            title="Docker",
            session_id="s1",
            user_id="u1",
        ),
    )
    vstore.add_documents([chunk])
    results = vstore.hybrid_search("Docker isolation", top_k=3)
    assert len(results) >= 1


def test_has_document(vstore):
    assert vstore.has_document("s1", "abc123") is False


def test_collection_stats(vstore):
    stats = vstore.get_collection_stats()
    assert "count" in stats
    assert stats["count"] == 0

from __future__ import annotations

import pytest

from agents.retrieval_hints import query_answer_from_chat_skip_retrieval
from core.document_processor import DocumentProcessor
from core.memory_store import MemoryStore
from core.models import ChatMessage, MessageRole, QueryIntent, RetrievalStrategy, Session
from core.text_splitting import iter_markdown_sections, recursive_chunk


def test_memory_store_session_and_messages(tmp_path):
    db = tmp_path / "t.db"
    store = MemoryStore(db_path=str(db))
    s = store.create_session("u1")
    assert s.user_id == "u1"
    m = ChatMessage(role=MessageRole.user, content="hi")
    store.add_message(s.session_id, m)
    loaded = store.get_session(s.session_id)
    assert loaded is not None
    assert len(loaded.messages) == 1
    counts = store.table_counts()
    assert counts["sessions"] == 1
    assert counts["messages"] == 1


def test_document_processor_markdown_chunks(tmp_path):
    md = tmp_path / "x.md"
    md.write_text(
        "# Title\n\n## Section A\n\nHello world.\n\n```py\nprint(1)\n```\n",
        encoding="utf-8",
    )
    proc = DocumentProcessor(chunk_size=200, chunk_overlap=20)
    chunks = proc.process_file(md)
    assert len(chunks) >= 1
    assert chunks[0].metadata.doc_type == "markdown"
    assert chunks[0].metadata.title == "Title"


def test_document_processor_upload_disambiguates_same_filename(tmp_path):
    """API uploads use temp paths; stamp original name + id so two README.md differ."""
    md = tmp_path / "tmpabcd.md"
    md.write_text("# Same heading\n\nDoc A body.\n", encoding="utf-8")
    proc = DocumentProcessor(chunk_size=200, chunk_overlap=20)
    uid = "11111111-2222-3333-4444-555555555555"
    chunks = proc.process_file(
        md,
        original_filename="README.md",
        doc_instance_id=uid,
        session_id="sess-chat-1",
        content_hash="abc123",
    )
    assert chunks[0].metadata.source == "README.md · 11111111"
    assert chunks[0].metadata.original_filename == "README.md"
    assert chunks[0].metadata.doc_instance_id == uid
    assert chunks[0].metadata.session_id == "sess-chat-1"
    assert chunks[0].metadata.content_hash == "abc123"
    assert "11111111" in chunks[0].metadata.title


def test_session_not_found(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "m.db"))
    assert store.get_session("nonexistent-session-id") is None


def test_recent_messages_ordering(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "m.db"))
    s = store.create_session("u1")
    for i in range(10):
        store.add_message(
            s.session_id,
            ChatMessage(role=MessageRole.user, content=f"msg-{i}"),
        )
    recent = store.get_recent_messages(s.session_id, n=3)
    assert [m.content for m in recent] == ["msg-7", "msg-8", "msg-9"]


def test_message_count(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "m.db"))
    s = store.create_session("u1")
    assert store.get_message_count(s.session_id) == 0
    store.add_message(s.session_id, ChatMessage(role=MessageRole.user, content="a"))
    store.add_message(s.session_id, ChatMessage(role=MessageRole.user, content="b"))
    assert store.get_message_count(s.session_id) == 2


def test_update_session_summary(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "m.db"))
    s = store.create_session("u1")
    store.update_session_summary(s.session_id, "rolling summary text")
    loaded = store.get_session(s.session_id)
    assert loaded is not None
    assert loaded.summary == "rolling summary text"


def test_user_sessions_list(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "m.db"))
    store.create_session("user_1")
    store.create_session("user_1")
    store.create_session("user_2")
    rows = store.get_user_sessions("user_1")
    assert len(rows) == 2
    assert all(r["user_id"] == "user_1" for r in rows)


def test_store_and_get_user_memory(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "m.db"))
    store.store_user_memory("alice", "preference", "Prefers dark mode UI")
    rows = store.get_user_memories("alice", limit=10)
    assert len(rows) == 1
    assert rows[0]["content"] == "Prefers dark mode UI"
    assert rows[0]["memory_type"] == "preference"


def test_get_memories_by_type(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "m.db"))
    store.store_user_memory("bob", "preference", "p1")
    store.store_user_memory("bob", "preference", "p2")
    store.store_user_memory("bob", "context", "c1")
    pref = store.get_user_memories("bob", memory_type="preference", limit=10)
    assert len(pref) == 2
    assert {r["content"] for r in pref} == {"p1", "p2"}


def test_prune_old_memories(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "m.db"))
    for i in range(10):
        store.store_user_memory("carol", "context", f"m{i}")
    store.prune_old_memories("carol", keep_count=3)
    remaining = store.get_user_memories("carol", limit=100)
    assert len(remaining) == 3


def test_session_isolation(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "m.db"))
    s1 = store.create_session("user_a")
    s2 = store.create_session("user_b")
    store.add_message(s1.session_id, ChatMessage(role=MessageRole.user, content="only-a"))
    store.add_message(s2.session_id, ChatMessage(role=MessageRole.user, content="only-b"))
    m1 = store.get_messages(s1.session_id)
    m2 = store.get_messages(s2.session_id)
    assert len(m1) == 1 and m1[0].content == "only-a"
    assert len(m2) == 1 and m2[0].content == "only-b"


def test_document_processor_html(tmp_path):
    html = tmp_path / "p.html"
    html.write_text(
        "<!DOCTYPE html><html><head><title>Doc</title></head><body>"
        "<h1>Heading</h1><p>Intro paragraph.</p><pre><code>print(42)</code></pre></body></html>",
        encoding="utf-8",
    )
    proc = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    chunks = proc.process_file(html)
    assert len(chunks) >= 1
    assert chunks[0].metadata.doc_type == "html"
    assert chunks[0].metadata.has_code_blocks is True


def test_document_processor_unsupported(tmp_path):
    bad = tmp_path / "f.xyz"
    bad.write_text("x", encoding="utf-8")
    proc = DocumentProcessor()
    with pytest.raises(ValueError, match="Unsupported"):
        proc.process_file(bad)


def test_recursive_chunk_basic():
    parts = recursive_chunk("One paragraph.\n\nSecond here.", chunk_size=200, chunk_overlap=20)
    assert len(parts) >= 1
    assert "One paragraph" in parts[0]


def test_recursive_chunk_overlap():
    # Single oversized paragraph triggers character-level split with overlap
    word = "abcdefghij "
    text = (word * 80).strip()
    chunks = recursive_chunk(text, chunk_size=80, chunk_overlap=15)
    assert len(chunks) > 1
    assert all(len(c) <= 80 for c in chunks)
    # Overlap duplicates boundary spans across chunk strings, so sum(lengths) > len(text)
    assert sum(len(c) for c in chunks) > len(text)


def test_iter_markdown_sections():
    md = "# Title\n\n## Section One\n\nBody one.\n\n## Section Two\n\nBody two.\n"
    sections = iter_markdown_sections(md)
    bodies = [b for _, b in sections if b.strip()]
    assert any("Body one" in b for b in bodies)
    assert any("Body two" in b for b in bodies)


def test_chat_message_creation():
    m = ChatMessage(role=MessageRole.user, content="hello")
    assert m.id
    assert m.timestamp
    assert m.content == "hello"


def test_session_model_defaults():
    s = Session(user_id="x")
    assert s.user_id == "x"
    assert s.summary == ""
    assert s.messages == []
    assert len(s.session_id) > 0


def test_query_intent_enum_values():
    names = set(QueryIntent.__members__.keys())
    assert names == {
        "factual",
        "follow_up",
        "clarification",
        "comparison",
        "multi_part",
        "greeting",
        "off_topic",
    }


def test_retrieval_strategy_enum():
    names = set(RetrievalStrategy.__members__.keys())
    assert names == {"semantic", "keyword", "hybrid", "none"}


def test_chat_meta_routing_skips_vector_path():
    assert query_answer_from_chat_skip_retrieval("What is the document name?") is True
    assert query_answer_from_chat_skip_retrieval("What did I upload?") is True
    assert query_answer_from_chat_skip_retrieval("What is my name?") is True
    assert (
        query_answer_from_chat_skip_retrieval(
            "Explain the authentication section in the uploaded PDF in detail"
        )
        is False
    )


def test_synthesis_prompts_helper():
    """`_synthesis_prompts` builds prompts from graph-shaped state."""
    from agents.context_synthesis import _synthesis_prompts

    state = {
        "session_id": "",
        "user_id": "",
        "user_query": "test",
        "conversation_history": [],
        "conversation_summary": "",
        "user_memory_context": "",
        "cross_session_context": "",
        "retrieved_documents": [],
        "query_intent": "factual",
    }
    system, user_block, docs, refuse = _synthesis_prompts(state)
    assert isinstance(system, str)
    assert "test" in user_block
    assert refuse is None



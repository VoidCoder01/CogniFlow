from __future__ import annotations

from core.document_processor import DocumentProcessor
from core.memory_store import MemoryStore
from core.models import ChatMessage, MessageRole


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



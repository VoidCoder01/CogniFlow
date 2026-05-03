"""Deterministic session_recall handler and store isolation."""

from __future__ import annotations

from agents.session_recall import (
    EMPTY_SESSION_NO_DISCUSSION,
    LONG_HISTORY_MAX_MESSAGES,
    SCOPE_HEADER,
    handle_session_recall,
    session_recall_node,
)
from core.memory_store import MemoryStore
from core.models import ChatMessage, MessageRole


class TestHandleSessionRecall:
    def test_same_session_lists_user_queries(self):
        """Prior user turns → scoped header + bullets (no LLM in handler)."""
        state = {
            "session_id": "s-a",
            "conversation_history": [
                {"role": "user", "content": "Explain RAG"},
                {"role": "assistant", "content": "RAG is retrieval-augmented generation."},
            ],
            "conversation_summary": "",
            "cross_session_context": "SECRET_OTHER_CHAT",
            "user_memory_context": "SECRET_USER_MEM",
        }
        text = handle_session_recall(state)
        assert SCOPE_HEADER in text
        assert "Explain RAG" in text
        assert "- " in text
        assert "SECRET" not in text

    def test_empty_history_and_summary(self):
        state = {
            "session_id": "s-b",
            "conversation_history": [],
            "conversation_summary": "",
        }
        assert handle_session_recall(state) == EMPTY_SESSION_NO_DISCUSSION

    def test_empty_messages_but_summary_only(self):
        """Summary without message rows still counts as session signal."""
        state = {
            "session_id": "s-c",
            "conversation_history": [],
            "conversation_summary": "User asked about deployment.",
        }
        text = handle_session_recall(state)
        assert SCOPE_HEADER in text
        assert "deployment" in text

    def test_long_thread_uses_summary_and_recent_window(self):
        msgs = []
        for i in range(LONG_HISTORY_MAX_MESSAGES + 2):
            msgs.append({"role": "user", "content": f"Q{i}"})
            msgs.append({"role": "assistant", "content": f"A{i}"})
        state = {
            "session_id": "s-long",
            "conversation_history": msgs,
            "conversation_summary": "Long thread overview.",
        }
        text = handle_session_recall(state)
        assert "Session summary (this chat thread)" in text
        assert "Recent questions" in text
        assert "Q" in text

    def test_session_recall_node_no_llm(self):
        state: dict = {
            "session_id": "x",
            "user_query": "What did I ask?",
            "query_intent": "session_recall",
            "conversation_history": [{"role": "user", "content": "Hi"}],
            "conversation_summary": "",
        }
        out = session_recall_node(state)
        assert out["response"]
        assert out["retrieved_documents"] == []
        assert any(e.get("node") == "session_recall" for e in out["agent_log"])


def test_memory_store_messages_scoped_by_session(tmp_path):
    db = tmp_path / "m.sqlite"
    store = MemoryStore(str(db))
    sa = store.create_session("user-1")
    sb = store.create_session("user-1")
    store.add_message(
        sa.session_id,
        ChatMessage(role=MessageRole.user, content="Explain vector retrieval"),
    )
    assert store.get_recent_messages(sb.session_id, n=10) == []
    ra = store.get_recent_messages(sa.session_id, n=10)
    assert len(ra) == 1
    assert ra[0].content == "Explain vector retrieval"

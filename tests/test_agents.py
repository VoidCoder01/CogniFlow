"""Unit tests for LangGraph agent nodes with mocked LLM."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agents.conversation_summarizer import conversation_summarizer_node
from agents.memory_manager import memory_manager_node
from agents.query_understanding import (
    _heuristic_classify,
    _normalize_intent,
    query_understanding_node,
)
from agents.retrieval_router import _heuristic_strategy, retrieval_router_node
from agents.schemas import MemoryExtractionResult, QueryUnderstandingResult


class TestQueryUnderstandingHeuristics:
    def test_greeting(self):
        intent, hist, rewrite = _heuristic_classify("hello", "")
        assert intent == "greeting"
        assert hist is False
        assert rewrite is False

    def test_follow_up_with_pronoun(self):
        intent, hist, rewrite = _heuristic_classify(
            "what about that one?", "user: Tell me about FastAPI"
        )
        assert intent == "follow_up"
        assert hist is True
        assert rewrite is True

    def test_comparison(self):
        intent, _, _ = _heuristic_classify("compare Python vs Java", "")
        assert intent == "comparison"

    def test_multi_part(self):
        intent, _, _ = _heuristic_classify(
            "what is X? and how does Y work?", ""
        )
        assert intent == "multi_part"

    def test_factual_default(self):
        intent, _, _ = _heuristic_classify(
            "how does dependency injection work in FastAPI", ""
        )
        assert intent == "factual"


class TestNormalizeIntent:
    def test_valid(self):
        assert _normalize_intent("follow_up") == "follow_up"

    def test_alias(self):
        assert _normalize_intent("followup") == "follow_up"

    def test_unknown(self):
        assert _normalize_intent("garbage") == "factual"


class TestHeuristicStrategy:
    def test_greeting(self):
        assert _heuristic_strategy("greeting", "hi") == "none"

    def test_error_code(self):
        assert _heuristic_strategy("factual", "ECONN_REFUSED error") == "hybrid"

    def test_identifier(self):
        assert _heuristic_strategy("factual", "OAuth2") == "keyword"

    def test_general(self):
        assert (
            _heuristic_strategy("factual", "how does authentication work") == "semantic"
        )


class TestQueryUnderstandingNodeWithMock:
    @patch("agents.query_understanding.get_chat_model")
    def test_successful_structured_output(self, mock_get_model):
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = QueryUnderstandingResult(
            intent="factual", needs_history=False, needs_rewrite=False
        )
        mock_model.with_structured_output.return_value = mock_structured
        mock_get_model.return_value = mock_model

        state = {"user_query": "What is FastAPI?", "conversation_history": []}
        result = query_understanding_node(state)
        assert result["query_intent"] == "factual"
        assert result["needs_rewrite"] is False
        assert len(result["agent_log"]) == 1

    @patch("agents.query_understanding.get_chat_model")
    def test_fallback_on_llm_failure(self, mock_get_model):
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("LLM failed")
        mock_model.with_structured_output.return_value = mock_structured
        mock_get_model.return_value = mock_model

        state = {"user_query": "hello", "conversation_history": []}
        result = query_understanding_node(state)
        assert result["query_intent"] == "greeting"


class TestConversationSummarizerNode:
    @patch("agents.conversation_summarizer.get_chat_model")
    def test_skips_when_too_few_messages(self, mock_get_model):
        state = {
            "conversation_history": [{"role": "user", "content": "hi"}],
            "conversation_summary": "",
            "should_summarize": False,
        }
        result = conversation_summarizer_node(state)
        assert any(e.get("skipped") for e in result.get("agent_log", []))
        mock_get_model.assert_not_called()

    @patch("agents.conversation_summarizer.get_chat_model")
    def test_runs_when_enough_messages(self, mock_get_model):
        mock_model = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "Summary: user discussed FastAPI auth."
        mock_model.invoke.return_value = mock_resp
        mock_get_model.return_value = mock_model

        msgs = [{"role": "user", "content": f"msg-{i}"} for i in range(10)]
        state = {
            "conversation_history": msgs,
            "conversation_summary": "",
            "should_summarize": True,
        }
        result = conversation_summarizer_node(state)
        assert "conversation_summary" in result
        assert len(result["conversation_summary"]) > 0


class TestMemoryManagerNode:
    @patch("agents.memory_manager.get_chat_model")
    def test_extracts_memories(self, mock_get_model):
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MemoryExtractionResult(items=[])
        mock_model.with_structured_output.return_value = mock_structured
        mock_get_model.return_value = mock_model

        state = {
            "user_id": "alice",
            "response": "Use PostgreSQL with B-tree indexes.",
            "conversation_history": [
                {"role": "user", "content": "I'm using PostgreSQL"},
            ],
        }
        result = memory_manager_node(state)
        assert "memory_updates" in result
        assert isinstance(result["memory_updates"], list)

    @patch("agents.memory_manager.get_chat_model")
    def test_handles_llm_failure(self, mock_get_model):
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("structured output failed")
        mock_model.with_structured_output.return_value = mock_structured
        mock_get_model.return_value = mock_model

        state = {"user_id": "bob", "response": "test", "conversation_history": []}
        result = memory_manager_node(state)
        assert result["memory_updates"] == []


def test_merge_graph_patch_appends_lists():
    from agents.orchestrator import merge_graph_patch

    state: dict = {"agent_log": [{"node": "a"}], "memory_updates": [{"x": 1}], "x": 1}
    merge_graph_patch(
        state,
        {
            "agent_log": [{"node": "b"}],
            "memory_updates": [{"y": 2}],
            "response": "ok",
        },
    )
    assert len(state["agent_log"]) == 2
    assert len(state["memory_updates"]) == 2
    assert state["response"] == "ok"


class TestRetrievalRouterWithMockVS:
    @patch("agents.retrieval_router._get_vector_store")
    @patch("agents.retrieval_router.get_chat_model")
    def test_routes_with_mock_vector(self, mock_llm, mock_vs):
        mock_struct = MagicMock()
        from agents.schemas import RetrievalRoutingResult

        mock_struct.invoke.return_value = RetrievalRoutingResult(
            strategy="none", rationale="test"
        )
        mock_llm.return_value.with_structured_output.return_value = mock_struct
        mock_vs.return_value.semantic_search.return_value = []

        state = {
            "query_intent": "greeting",
            "user_query": "hi",
            "rewritten_query": "",
            "session_id": "",
            "user_id": "",
        }
        out = retrieval_router_node(state)
        assert "retrieval_strategy" in out
        assert "agent_log" in out

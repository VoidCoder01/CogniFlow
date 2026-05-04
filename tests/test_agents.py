"""Unit tests for LangGraph agent nodes with mocked LLM."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agents.context_synthesis import _synthesis_prompts
from agents.context_validation import context_validation_node
from agents.conversation_summarizer import conversation_summarizer_node
from agents.memory_manager import memory_manager_node
from agents.query_understanding import (
    _coerce_router_output,
    _heuristic_classify,
    _looks_like_lightweight_greeting,
    _normalize_intent,
    query_understanding_node,
)
from agents.retrieval_router import _heuristic_strategy, retrieval_router_node
from agents.schemas import ContextValidationResult, MemoryExtractionResult, QueryRouterResult


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

    def test_general_knowledge_default(self):
        intent, _, _ = _heuristic_classify(
            "how does dependency injection work in FastAPI", ""
        )
        assert intent == "general_knowledge"

    def test_factual_when_doc_signal(self):
        intent, _, _ = _heuristic_classify(
            "what does my uploaded file say about timeouts?", ""
        )
        assert intent == "factual"

    def test_hello_dude_heuristic_greeting(self):
        intent, _, _ = _heuristic_classify("hello dude", "")
        assert intent == "greeting"


class TestLightweightGreetingFastPath:
    def test_hello_dude(self):
        assert _looks_like_lightweight_greeting("hello dude")

    def test_not_technical_question(self):
        assert not _looks_like_lightweight_greeting("hello what is the REST API")


class TestNormalizeIntent:
    def test_valid(self):
        assert _normalize_intent("follow_up") == "follow_up"

    def test_alias(self):
        assert _normalize_intent("followup") == "follow_up"

    def test_unknown(self):
        assert _normalize_intent("garbage") == "general_knowledge"


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

    def test_no_retrieval_intents(self):
        assert _heuristic_strategy("general_knowledge", "what is gravity") == "none"
        assert _heuristic_strategy("meta", "what did I ask") == "none"
        assert _heuristic_strategy("session_recall", "recap") == "none"


class TestQueryUnderstandingNodeWithMock:
    @patch("agents.query_understanding.get_chat_model")
    def test_successful_structured_output(self, mock_get_model):
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = QueryRouterResult(
            intent="factual_doc",
            needs_retrieval=True,
            needs_memory=False,
            response_style="short",
        )
        mock_model.with_structured_output.return_value = mock_structured
        mock_get_model.return_value = mock_model

        state = {"user_query": "What is FastAPI?", "conversation_history": []}
        result = query_understanding_node(state)
        assert result["query_intent"] == "factual_doc"
        assert result["needs_retrieval"] is True
        assert result["needs_rewrite"] is False
        assert len(result["agent_log"]) == 1

    @patch("agents.query_understanding.get_chat_model")
    def test_fallback_on_llm_failure(self, mock_get_model):
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("LLM failed")
        mock_model.with_structured_output.return_value = mock_structured
        mock_get_model.return_value = mock_model

        state = {
            "user_query": "compare Redis versus Memcached for session storage",
            "conversation_history": [],
        }
        result = query_understanding_node(state)
        assert result["query_intent"] == "factual_doc"
        assert result["needs_retrieval"] is True

    @patch("agents.query_understanding.get_chat_model")
    def test_fast_path_skips_llm_for_casual_greeting(self, mock_get_model):
        state = {"user_query": "hello dude", "conversation_history": []}
        result = query_understanding_node(state)
        assert result["query_intent"] == "greeting"
        assert result["needs_retrieval"] is False
        assert result["agent_log"][0].get("fast_path") is True
        mock_get_model.assert_not_called()

    @patch("agents.query_understanding.get_chat_model")
    def test_structured_session_recall_from_router(self, mock_get_model):
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = QueryRouterResult(
            intent="session_recall",
            needs_retrieval=False,
            needs_memory=False,
            response_style="short",
        )
        mock_model.with_structured_output.return_value = mock_structured
        mock_get_model.return_value = mock_model

        state = {"user_query": "What did we discuss?", "conversation_history": []}
        result = query_understanding_node(state)
        assert result["query_intent"] == "session_recall"
        assert result["needs_retrieval"] is False
        assert result["needs_history"] is True


class TestCoerceRouterOutput:
    def test_session_recall_never_retrieves(self):
        out = QueryRouterResult(
            intent="session_recall",
            needs_retrieval=True,
            needs_memory=True,
            response_style="short",
        )
        patch = _coerce_router_output(out, "Recap this chat", "(empty)")
        assert patch["query_intent"] == "session_recall"
        assert patch["needs_retrieval"] is False
        assert patch["needs_memory"] is False
        assert patch["needs_history"] is True

    def test_multi_part_routes_to_decomposer(self):
        """LLM-returned multi_part must stay multi_part (not general_knowledge) with retrieval on."""
        out = QueryRouterResult(
            intent="multi_part",
            needs_retrieval=False,
            needs_memory=False,
            response_style="short",
        )
        patch = _coerce_router_output(
            out, "What is X? Also how does Y work?", "(empty)"
        )
        assert patch["query_intent"] == "multi_part"
        assert patch["needs_retrieval"] is True
        assert patch["needs_memory"] is False

    def test_multi_part_spaced_label_normalized(self):
        out = QueryRouterResult(
            intent="multi part",
            needs_retrieval=True,
            needs_memory=False,
            response_style="short",
        )
        patch = _coerce_router_output(out, "Q1? Q2?", "(empty)")
        assert patch["query_intent"] == "multi_part"


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


class TestSynthesisPromptIsolation:
    def test_meta_includes_user_memory_for_cross_session_prefs(self):
        state = {
            "session_id": "s1",
            "user_id": "u1",
            "user_query": "How do you remember my name?",
            "query_intent": "meta",
            "conversation_history": [{"role": "user", "content": "I'm Alex."}],
            "conversation_summary": "",
            "retrieved_documents": [],
            "needs_retrieval": False,
            "use_retrieved_context": False,
            "user_memory_context": "User likes bullet lists.",
            "cross_session_context": "OTHER_SESSION_SUMMARY",
            "response_style": "short",
        }
        _system, user_block, _docs, _fast = _synthesis_prompts(state)
        assert "Known user preferences" in user_block
        assert "User likes bullet lists." in user_block
        assert "Other recent chats" in user_block
        assert "OTHER_SESSION_SUMMARY" in user_block

    def test_session_recall_strips_memory_and_peer_blocks(self):
        state = {
            "session_id": "s1",
            "user_id": "u1",
            "user_query": "What did I ask earlier?",
            "query_intent": "session_recall",
            "conversation_history": [{"role": "user", "content": "Hi"}],
            "conversation_summary": "",
            "retrieved_documents": [],
            "needs_retrieval": False,
            "use_retrieved_context": False,
            "user_memory_context": "User prefers Python.",
            "cross_session_context": "OTHER_CHAT",
            "response_style": "short",
        }
        _system, user_block, _docs, _fast = _synthesis_prompts(state)
        assert "Known user preferences" not in user_block
        assert "Other recent chats" not in user_block
        assert "OTHER_CHAT" not in user_block


class TestContextValidationNode:
    @patch("agents.context_validation.get_chat_model")
    def test_empty_retrieval_skips_llm(self, mock_get_model):
        state: dict = {"user_query": "What is X?", "retrieved_documents": []}
        out = context_validation_node(state)
        assert out["use_retrieved_context"] is False
        assert out["context_validation_reason"] == "empty_context"
        mock_get_model.assert_not_called()

    @patch("agents.context_validation.get_chat_model")
    def test_nonempty_invokes_structured_model(self, mock_get_model):
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = ContextValidationResult(
            use_context=False,
            reason="off_topic",
        )
        mock_model.with_structured_output.return_value = mock_structured
        mock_get_model.return_value = mock_model

        state = {
            "user_query": "What is X?",
            "retrieved_documents": [
                {"content": "Some chunk about Y.", "metadata": {"source": "a.md"}}
            ],
        }
        out = context_validation_node(state)
        assert out["use_retrieved_context"] is False
        assert "off_topic" in (out.get("context_validation_reason") or "")
        mock_get_model.assert_called_once()


class TestMemoryManagerNode:
    @patch("agents.memory_manager.get_chat_model")
    def test_skips_greeting_without_llm(self, mock_get_model):
        state = {
            "user_id": "u1",
            "query_intent": "greeting",
            "response": "Hey! How can I help?",
            "conversation_history": [],
        }
        out = memory_manager_node(state)
        assert out["memory_updates"] == []
        mock_get_model.assert_not_called()

    @patch("agents.memory_manager.get_chat_model")
    def test_skips_session_recall_without_llm(self, mock_get_model):
        state = {
            "user_id": "u1",
            "query_intent": "session_recall",
            "response": "We haven't discussed anything yet in this session.",
            "conversation_history": [],
        }
        out = memory_manager_node(state)
        assert out["memory_updates"] == []
        assert any(
            e.get("reason") == "session_recall_turn"
            for e in (out.get("agent_log") or [])
        )
        mock_get_model.assert_not_called()

    @patch("agents.memory_manager.get_chat_model")
    def test_off_topic_short_still_invokes_extractor(self, mock_get_model):
        """Misclassified stack intros must not be skipped by a length heuristic."""
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MemoryExtractionResult(items=[])
        mock_model.with_structured_output.return_value = mock_structured
        mock_get_model.return_value = mock_model

        short_stack = (
            "I'm building a REST API with FastAPI and PostgreSQL for my startup ShopFlow"
        )
        state = {
            "user_id": "u1",
            "user_query": short_stack,
            "query_intent": "off_topic",
            "response": "Sounds great — how can I help?",
            "conversation_history": [{"role": "user", "content": short_stack}],
        }
        memory_manager_node(state)
        mock_get_model.assert_called_once()

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


class TestRetrievalRouterSubQueries:
    @patch("agents.retrieval_router._get_vector_store")
    @patch("agents.retrieval_router.get_chat_model")
    def test_sub_queries_expand_retrieval(self, mock_llm, mock_vs):
        from agents.schemas import RetrievalRoutingResult

        mock_struct = MagicMock()
        mock_struct.invoke.return_value = RetrievalRoutingResult(
            strategy="semantic", rationale="test"
        )
        mock_llm.return_value.with_structured_output.return_value = mock_struct

        mock_vs.return_value.semantic_search.side_effect = [
            [{"id": "doc1", "content": "FastAPI auth", "metadata": {}, "distance": 0.2}],
            [{"id": "doc2", "content": "Django auth", "metadata": {}, "distance": 0.3}],
        ]

        state = {
            "query_intent": "multi_part",
            "user_query": "compare FastAPI and Django auth",
            "rewritten_query": "compare FastAPI and Django authentication",
            "sub_queries": [
                "FastAPI authentication setup",
                "Django authentication setup",
            ],
            "session_id": "",
            "user_id": "",
        }
        out = retrieval_router_node(state)
        assert len(out["retrieved_documents"]) == 2
        ids = {d["id"] for d in out["retrieved_documents"]}
        assert ids == {"doc1", "doc2"}


class TestQueryDecomposer:
    @patch("agents.query_decomposer.get_chat_model")
    def test_decomposes_multi_part(self, mock_llm):
        from agents.query_decomposer import query_decomposer_node
        from agents.schemas import QueryDecompositionResult

        mock_struct = MagicMock()
        mock_struct.invoke.return_value = QueryDecompositionResult(
            sub_queries=["What is FastAPI?", "What is Django?"]
        )
        mock_llm.return_value.with_structured_output.return_value = mock_struct

        state = {
            "user_query": "What is FastAPI and what is Django?",
            "rewritten_query": "",
        }
        result = query_decomposer_node(state)
        assert len(result["sub_queries"]) == 2
        assert "FastAPI" in result["sub_queries"][0]

    @patch("agents.query_decomposer.get_chat_model")
    def test_fallback_on_failure(self, mock_llm):
        from agents.query_decomposer import query_decomposer_node

        mock_struct = MagicMock()
        mock_struct.invoke.side_effect = Exception("fail")
        mock_llm.return_value.with_structured_output.return_value = mock_struct

        state = {"user_query": "original query", "rewritten_query": ""}
        result = query_decomposer_node(state)
        assert result["sub_queries"] == ["original query"]


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

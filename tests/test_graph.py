"""Integration test: build the LangGraph and invoke with fully mocked LLM."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from agents.graph_state import CogniFlowState
from agents.orchestrator import build_graph
from agents.schemas import (
    ContextValidationResult,
    MemoryExtractionResult,
    QueryDecompositionResult,
    QueryRouterResult,
    RetrievalRoutingResult,
)


def _mock_chat_model():
    """Return a mock that handles both .invoke() and .with_structured_output()."""
    model = MagicMock()

    plain_resp = AIMessage(content="Mocked LLM response about FastAPI.")
    model.invoke.return_value = plain_resp

    def structured_factory(schema):
        structured = MagicMock()
        if schema == QueryRouterResult:
            structured.invoke.return_value = QueryRouterResult(
                intent="factual_doc",
                needs_retrieval=True,
                needs_memory=False,
                response_style="short",
            )
        elif schema == ContextValidationResult:
            structured.invoke.return_value = ContextValidationResult(
                use_context=True,
                reason="test_ok",
            )
        elif schema == RetrievalRoutingResult:
            structured.invoke.return_value = RetrievalRoutingResult(
                strategy="semantic",
                rationale="test",
            )
        elif schema == MemoryExtractionResult:
            structured.invoke.return_value = MemoryExtractionResult(items=[])
        elif schema == QueryDecompositionResult:
            structured.invoke.return_value = QueryDecompositionResult(
                sub_queries=["What is FastAPI?"]
            )
        else:
            structured.invoke.return_value = MagicMock()
        return structured

    model.with_structured_output.side_effect = structured_factory
    return model


def _mock_chat_model_session_recall_router():
    model = MagicMock()
    plain_resp = AIMessage(content="Summary stub.")
    model.invoke.return_value = plain_resp

    def structured_factory(schema):
        structured = MagicMock()
        if schema == QueryRouterResult:
            structured.invoke.return_value = QueryRouterResult(
                intent="session_recall",
                needs_retrieval=False,
                needs_memory=False,
                response_style="short",
            )
        elif schema == ContextValidationResult:
            structured.invoke.return_value = ContextValidationResult(
                use_context=False, reason="unused"
            )
        elif schema == RetrievalRoutingResult:
            structured.invoke.return_value = RetrievalRoutingResult(
                strategy="none", rationale="unused"
            )
        elif schema == MemoryExtractionResult:
            structured.invoke.return_value = MemoryExtractionResult(items=[])
        elif schema == QueryDecompositionResult:
            structured.invoke.return_value = QueryDecompositionResult(sub_queries=[])
        else:
            structured.invoke.return_value = MagicMock()
        return structured

    model.with_structured_output.side_effect = structured_factory
    return model


@patch("agents.memory_manager.get_chat_model")
@patch("agents.conversation_summarizer.get_chat_model")
@patch("agents.context_synthesis.get_chat_model")
@patch("agents.context_validation.get_chat_model")
@patch("agents.retrieval_router.get_chat_model")
@patch("agents.retrieval_router._get_vector_store")
@patch("agents.query_decomposer.get_chat_model")
@patch("agents.query_rewriting.get_chat_model")
@patch("agents.query_understanding.get_chat_model")
def test_full_graph_invocation(mm, cs_sum, cs, cv, rr, mock_vs, qd, qr, qu):
    mock_vs.return_value.semantic_search.return_value = []
    mock_vs.return_value.keyword_search.return_value = []
    mock_vs.return_value.hybrid_search.return_value = []

    mock = _mock_chat_model()
    for m in (mm, cs_sum, cs, cv, rr, qd, qr, qu):
        m.return_value = mock

    graph = build_graph(enable_checkpointing=False)

    state: CogniFlowState = {
        "session_id": "test-session",
        "user_id": "test-user",
        "user_query": "What is FastAPI?",
        "conversation_history": [],
        "user_memory_context": "",
        "cross_session_context": "",
        "query_intent": "",
        "needs_retrieval": True,
        "needs_memory": False,
        "response_style": "short",
        "needs_history": False,
        "needs_rewrite": False,
        "rewritten_query": "",
        "sub_queries": [],
        "retrieval_strategy": "semantic",
        "retrieved_documents": [],
        "synthesized_context": "",
        "response": "",
        "should_summarize": False,
        "conversation_summary": "",
        "memory_updates": [],
        "agent_log": [],
    }

    result = graph.invoke(state)

    assert result["response"]
    assert len(result["agent_log"]) >= 4
    assert result["query_intent"] == "factual_doc"
    assert result.get("use_retrieved_context") is False
    assert any(e.get("node") == "context_validation" for e in result["agent_log"])


@patch("agents.memory_manager.get_chat_model")
@patch("agents.conversation_summarizer.get_chat_model")
@patch("agents.context_synthesis.get_chat_model")
@patch("agents.context_validation.get_chat_model")
@patch("agents.retrieval_router.get_chat_model")
@patch("agents.retrieval_router._get_vector_store")
@patch("agents.query_decomposer.get_chat_model")
@patch("agents.query_rewriting.get_chat_model")
@patch("agents.query_understanding.get_chat_model")
def test_session_recall_branch_skips_retrieval(
    mm, cs_sum, cs, cv, rr, mock_vs, qd, qr, qu
):
    mock_vs.return_value.semantic_search.return_value = []

    mock = _mock_chat_model_session_recall_router()
    for m in (mm, cs_sum, cs, cv, rr, qd, qr, qu):
        m.return_value = mock

    graph = build_graph(enable_checkpointing=False)
    state: CogniFlowState = {
        "session_id": "sr-session",
        "user_id": "u1",
        "user_query": "What did I ask earlier?",
        "conversation_history": [
            {"role": "user", "content": "Explain RAG briefly."},
            {"role": "assistant", "content": "RAG combines retrieval with generation."},
        ],
        "user_memory_context": "",
        "cross_session_context": "",
        "query_intent": "",
        "needs_retrieval": True,
        "needs_memory": False,
        "response_style": "short",
        "needs_history": False,
        "needs_rewrite": False,
        "rewritten_query": "",
        "sub_queries": [],
        "retrieval_strategy": "semantic",
        "retrieved_documents": [],
        "synthesized_context": "",
        "response": "",
        "should_summarize": False,
        "conversation_summary": "",
        "memory_updates": [],
        "agent_log": [],
    }

    result = graph.invoke(state)
    nodes = [e.get("node") for e in result["agent_log"]]
    assert "session_recall" in nodes
    assert "retrieval_router" not in nodes
    assert "context_validation" not in nodes
    assert "context_synthesis" not in nodes
    assert result["query_intent"] == "session_recall"
    assert "In this session, you asked:" in (result.get("response") or "")
    mock_vs.return_value.semantic_search.assert_not_called()

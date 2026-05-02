"""Integration test: build the LangGraph and invoke with fully mocked LLM."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from agents.graph_state import CogniFlowState
from agents.orchestrator import build_graph
from agents.schemas import (
    MemoryExtractionResult,
    QueryDecompositionResult,
    QueryUnderstandingResult,
    RetrievalRoutingResult,
)


def _mock_chat_model():
    """Return a mock that handles both .invoke() and .with_structured_output()."""
    model = MagicMock()

    plain_resp = AIMessage(content="Mocked LLM response about FastAPI.")
    model.invoke.return_value = plain_resp

    def structured_factory(schema):
        structured = MagicMock()
        if schema == QueryUnderstandingResult:
            structured.invoke.return_value = QueryUnderstandingResult(
                intent="factual",
                needs_history=False,
                needs_rewrite=False,
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


@patch("agents.memory_manager.get_chat_model")
@patch("agents.conversation_summarizer.get_chat_model")
@patch("agents.context_synthesis.get_chat_model")
@patch("agents.retrieval_router.get_chat_model")
@patch("agents.retrieval_router._get_vector_store")
@patch("agents.query_decomposer.get_chat_model")
@patch("agents.query_rewriting.get_chat_model")
@patch("agents.query_understanding.get_chat_model")
def test_full_graph_invocation(mm, cs_sum, cs, rr, mock_vs, qd, qr, qu):
    mock_vs.return_value.semantic_search.return_value = []
    mock_vs.return_value.keyword_search.return_value = []
    mock_vs.return_value.hybrid_search.return_value = []

    mock = _mock_chat_model()
    for m in (mm, cs_sum, cs, rr, qd, qr, qu):
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
    assert len(result["agent_log"]) >= 3
    assert result["query_intent"] == "factual"

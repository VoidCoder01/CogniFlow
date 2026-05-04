"""Checkpoint parity between streaming finalize and LangGraph state."""

from __future__ import annotations

from unittest.mock import patch

from agents.orchestrator import CogniFlowOrchestrator


def _minimal_graph_state(session_id: str) -> dict:
    return {
        "session_id": session_id,
        "user_id": "u1",
        "user_query": "hello",
        "conversation_history": [],
        "user_memory_context": "",
        "cross_session_context": "",
        "query_intent": "general_knowledge",
        "needs_retrieval": False,
        "needs_memory": False,
        "response_style": "short",
        "needs_history": False,
        "needs_rewrite": False,
        "rewritten_query": "hello",
        "retrieval_strategy": "none",
        "sub_queries": [],
        "retrieved_documents": [],
        "use_retrieved_context": True,
        "context_validation_reason": "",
        "synthesized_context": "",
        "response": "streamed assistant text",
        "should_summarize": False,
        "conversation_summary": "",
        "memory_updates": [],
        "agent_log": [{"node": "stub"}],
    }


@patch("agents.orchestrator.memory_manager_node", return_value={})
@patch("agents.orchestrator.conversation_summarizer_node", return_value={})
def test_finalize_persists_streaming_checkpoint(_mock_sum, _mock_mem):
    orch = CogniFlowOrchestrator()
    assert getattr(orch.graph, "checkpointer", None) is not None
    sid = "session-checkpoint-stream-test"
    gs = _minimal_graph_state(sid)
    orch.finalize_after_synthesis(gs)
    snap = orch.graph.get_state({"configurable": {"thread_id": sid}})
    assert snap.values.get("response") == "streamed assistant text"
    assert any(
        (e.get("node") == "stub") for e in (snap.values.get("agent_log") or [])
    )

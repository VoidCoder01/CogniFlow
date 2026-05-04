"""Branch coverage for ``agents.orchestrator`` routing and streaming prep."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.orchestrator import (
    CogniFlowOrchestrator,
    merge_graph_patch,
    route_after_rewriting,
    route_after_understanding,
)


def test_merge_graph_patch_appends_logs_and_memory():
    st: dict = {"agent_log": [{"a": 1}], "memory_updates": [{"m": 1}], "x": 1}
    merge_graph_patch(
        st,
        {
            "agent_log": [{"b": 2}],
            "memory_updates": [{"m": 2}],
            "x": 2,
        },
    )
    assert st["x"] == 2
    assert len(st["agent_log"]) == 2
    assert len(st["memory_updates"]) == 2


@pytest.mark.parametrize(
    "state,expected",
    [
        ({"query_intent": "greeting"}, "direct_synthesize"),
        ({"query_intent": "session_recall"}, "session_recall"),
        ({"query_intent": "meta", "needs_retrieval": True}, "direct_synthesize"),
        ({"query_intent": "general_knowledge"}, "direct_synthesize"),
        ({"query_intent": "factual_doc", "needs_retrieval": False}, "direct_synthesize"),
        ({"query_intent": "multi_part", "needs_retrieval": True}, "decompose"),
        ({"query_intent": "factual_doc", "needs_retrieval": True, "needs_rewrite": True}, "rewrite"),
        ({"query_intent": "factual_doc", "needs_retrieval": True, "needs_rewrite": False}, "retrieve"),
    ],
)
def test_route_after_understanding_branches(state, expected):
    assert route_after_understanding(state) == expected


def test_route_after_rewriting():
    assert route_after_rewriting({"query_intent": "multi_part"}) == "decompose"
    assert route_after_rewriting({"query_intent": "factual_doc"}) == "retrieve"


def _base_graph_state() -> dict:
    return {
        "session_id": "sid",
        "user_id": "u",
        "user_query": "q",
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
        "use_retrieved_context": True,
        "context_validation_reason": "",
        "synthesized_context": "",
        "response": "",
        "should_summarize": False,
        "conversation_summary": "",
        "memory_updates": [],
        "agent_log": [],
    }


@patch("agents.orchestrator.session_recall_node")
@patch("agents.orchestrator.query_understanding_node")
def test_run_until_before_synthesis_session_recall_branch(sr, qu):
    qu.return_value = {
        "query_intent": "session_recall",
        "needs_retrieval": False,
        "agent_log": [],
    }
    sr.return_value = {"response": "You asked about RAG.", "agent_log": []}
    orch = CogniFlowOrchestrator()
    gs = _base_graph_state()
    orch.run_until_before_synthesis(gs)
    sr.assert_called_once_with(gs)
    assert "RAG" in (gs.get("response") or "")


@patch("agents.orchestrator.query_understanding_node")
def test_run_until_before_synthesis_greeting_early_exit(qu):
    qu.return_value = {
        "query_intent": "greeting",
        "needs_retrieval": False,
        "agent_log": [{"node": "query_understanding"}],
    }
    orch = CogniFlowOrchestrator()
    gs = _base_graph_state()
    orch.run_until_before_synthesis(gs)
    assert gs["query_intent"] == "greeting"
    assert len(gs["agent_log"]) == 1


@patch("agents.orchestrator.context_validation_node")
@patch("agents.orchestrator.retrieval_router_node")
@patch("agents.orchestrator.query_understanding_node")
def test_run_until_before_synthesis_retrieve_path(qu, rr, cv):
    qu.return_value = {
        "query_intent": "factual_doc",
        "needs_retrieval": True,
        "needs_rewrite": False,
        "agent_log": [{"node": "qu"}],
    }
    rr.return_value = {"retrieval_strategy": "semantic", "retrieved_documents": [], "agent_log": []}
    cv.return_value = {"use_retrieved_context": True, "agent_log": []}
    orch = CogniFlowOrchestrator()
    gs = _base_graph_state()
    orch.run_until_before_synthesis(gs)
    rr.assert_called_once()
    cv.assert_called_once()


@patch("agents.orchestrator.context_validation_node")
@patch("agents.orchestrator.retrieval_router_node")
@patch("agents.orchestrator.query_decomposer_node")
@patch("agents.orchestrator.query_rewriting_node")
@patch("agents.orchestrator.query_understanding_node")
def test_run_until_rewrite_then_decompose(qu, qr_node, qd, rr, cv):
    qu.return_value = {
        "query_intent": "factual_doc",
        "needs_retrieval": True,
        "needs_rewrite": True,
        "agent_log": [],
    }
    qr_node.return_value = {
        "rewritten_query": "q1 and q2",
        "query_intent": "multi_part",
        "agent_log": [],
    }
    qd.return_value = {"sub_queries": ["s1", "s2"], "agent_log": []}
    rr.return_value = {"retrieved_documents": [], "agent_log": []}
    cv.return_value = {"use_retrieved_context": True, "agent_log": []}
    orch = CogniFlowOrchestrator()
    gs = _base_graph_state()
    orch.run_until_before_synthesis(gs)
    qr_node.assert_called_once()
    qd.assert_called_once()
    assert gs.get("query_intent") == "multi_part"


@patch("agents.orchestrator.context_validation_node")
@patch("agents.orchestrator.retrieval_router_node")
@patch("agents.orchestrator.query_decomposer_node")
@patch("agents.orchestrator.query_understanding_node")
def test_run_until_before_synthesis_decompose_branch(qu, qd, rr, cv):
    qu.return_value = {
        "query_intent": "multi_part",
        "needs_retrieval": True,
        "needs_rewrite": False,
        "agent_log": [],
    }
    qd.return_value = {"sub_queries": ["a", "b"], "agent_log": []}
    rr.return_value = {"retrieved_documents": [], "agent_log": []}
    cv.return_value = {"use_retrieved_context": True, "agent_log": []}
    orch = CogniFlowOrchestrator()
    gs = _base_graph_state()
    orch.run_until_before_synthesis(gs)
    qd.assert_called_once()
    assert gs.get("sub_queries") == ["a", "b"]


def test_build_graph_with_checkpointing_uses_injected_checkpointer():
    from agents.orchestrator import build_graph

    cp = MagicMock()
    g = build_graph(checkpointer=cp, enable_checkpointing=True)
    assert getattr(g, "checkpointer", None) is cp


def test_apply_context_synthesis_delegates_for_session_recall():
    orch = CogniFlowOrchestrator()
    gs = _base_graph_state()
    gs["query_intent"] = "session_recall"
    with patch("agents.orchestrator.session_recall_node", return_value={"response": "r", "agent_log": []}) as sn:
        orch.apply_context_synthesis(gs)
        sn.assert_called_once_with(gs)


def test_apply_context_synthesis_delegates_default():
    orch = CogniFlowOrchestrator()
    gs = _base_graph_state()
    gs["query_intent"] = "general_knowledge"
    with patch(
        "agents.orchestrator.context_synthesis_node",
        return_value={"response": "ok", "agent_log": []},
    ) as cn:
        orch.apply_context_synthesis(gs)
        cn.assert_called_once_with(gs)

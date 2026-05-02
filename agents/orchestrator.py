from __future__ import annotations

import logging
import os
from typing import Any, Iterator, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.conversation_summarizer import conversation_summarizer_node
from agents.context_synthesis import context_synthesis_node
from agents.graph_state import (
    CogniFlowState,
    RouteAfterUnderstanding,
    agent_state_to_graph,
    graph_to_agent_state,
)
from agents.memory_manager import memory_manager_node
from agents.query_rewriting import query_rewriting_node
from agents.query_understanding import query_understanding_node
from agents.retrieval_hints import query_suggests_document_lookup
from agents.retrieval_router import retrieval_router_node
from config import settings
from core.models import AgentState

logger = logging.getLogger(__name__)


def build_checkpointer() -> BaseCheckpointSaver:
    """
    Advanced LangGraph feature: persistent checkpoints (thread_id = session_id).
    Falls back to in-memory saver if sqlite backend is unavailable.
    """
    backend = settings.checkpoint_backend.lower().strip()
    if backend != "sqlite":
        return MemorySaver()

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        logger.warning(
            "langgraph.checkpoint.sqlite unavailable; install langgraph-checkpoint-sqlite. Using MemorySaver."
        )
        return MemorySaver()

    path = settings.checkpoint_sqlite_path
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        return SqliteSaver.from_conn_string(path)
    except Exception as exc:
        logger.warning("SqliteSaver init failed (%s); using MemorySaver", exc)
        return MemorySaver()


def route_after_understanding(state: CogniFlowState) -> RouteAfterUnderstanding:
    intent = (state.get("query_intent") or "").lower().strip()
    user_q = str(state.get("user_query") or "")
    if intent in ("greeting", "off_topic") and not query_suggests_document_lookup(
        user_q
    ):
        return "direct_synthesize"
    if state.get("needs_rewrite"):
        return "rewrite"
    return "retrieve"


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    g = StateGraph(CogniFlowState)
    g.add_node("query_understanding", query_understanding_node)
    g.add_node("query_rewriting", query_rewriting_node)
    g.add_node("retrieval_router", retrieval_router_node)
    g.add_node("context_synthesis", context_synthesis_node)
    g.add_node("conversation_summarizer", conversation_summarizer_node)
    g.add_node("memory_manager", memory_manager_node)

    g.add_edge(START, "query_understanding")
    g.add_conditional_edges(
        "query_understanding",
        route_after_understanding,
        {
            "direct_synthesize": "context_synthesis",
            "rewrite": "query_rewriting",
            "retrieve": "retrieval_router",
        },
    )
    g.add_edge("query_rewriting", "retrieval_router")
    g.add_edge("retrieval_router", "context_synthesis")
    g.add_edge("context_synthesis", "conversation_summarizer")
    g.add_edge("conversation_summarizer", "memory_manager")
    g.add_edge("memory_manager", END)

    return g.compile(checkpointer=checkpointer or build_checkpointer())


class CogniFlowOrchestrator:
    """LangGraph RAG pipeline with checkpointed thread state."""

    def __init__(self, checkpointer: BaseCheckpointSaver | None = None):
        self._checkpointer = checkpointer
        self.graph = build_graph(checkpointer=checkpointer)

    def invoke(
        self,
        state: AgentState,
        *,
        stream: bool = False,
        stream_mode: Literal["updates", "values", "messages"] = "updates",
    ) -> AgentState | Iterator[Any]:
        payload = agent_state_to_graph(state)
        payload.setdefault("memory_updates", [])
        payload.setdefault("agent_log", [])
        config: dict[str, Any] = {
            "configurable": {"thread_id": state.session_id},
        }

        if stream:
            return self.graph.stream(
                payload,
                config=config,
                stream_mode=stream_mode,
            )

        out = self.graph.invoke(payload, config=config)
        return graph_to_agent_state(state, out)

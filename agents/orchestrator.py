from __future__ import annotations

import logging
import os
from typing import Any, Iterator, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.conversation_summarizer import conversation_summarizer_node
from agents.context_synthesis import (
    context_synthesis_node,
    iter_context_synthesis_events,
)
from agents.graph_state import (
    CogniFlowState,
    RouteAfterUnderstanding,
    RouteAfterRewriting,
    agent_state_to_graph,
    graph_to_agent_state,
)
from agents.memory_manager import memory_manager_node
from agents.query_decomposer import query_decomposer_node
from agents.query_rewriting import query_rewriting_node
from agents.query_understanding import query_understanding_node
from agents.context_validation import context_validation_node
from agents.retrieval_router import retrieval_router_node
from agents.session_recall import iter_session_recall_events, session_recall_node
from config import settings
from core.models import AgentState

logger = logging.getLogger(__name__)


def merge_graph_patch(state: dict[str, Any], patch: dict[str, Any]) -> None:
    """Apply a node output dict onto runnable LangGraph state (append logs / memory deltas)."""
    for k, v in patch.items():
        if k == "agent_log":
            state.setdefault("agent_log", []).extend(v or [])
        elif k == "memory_updates":
            state.setdefault("memory_updates", []).extend(v or [])
        else:
            state[k] = v


def build_checkpointer() -> BaseCheckpointSaver:
    """Return persistent SQLite checkpointer when configured; otherwise in-memory saver."""
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
    """Branch after the master router (query_understanding).

    Intent-first routing (see ``agents.prompt_suite`` for prompt bodies):

    - ``factual_doc`` → ``retrieve`` (then ``context_validation`` → grounded vs knowledge
      in synthesis per ``use_retrieved_context``).
    - ``multi_part`` → ``decompose`` then retrieval as in the graph.
    - ``general_knowledge`` → ``direct_synthesize`` (no retrieval).
    - ``follow_up`` / ``meta`` → ``direct_synthesize`` (memory / history prompts).
    - ``session_recall`` → ``session_recall`` node (deterministic recap; no retrieval, no LLM).
    - ``preference`` → ``direct_synthesize`` (acknowledgement); durable prefs via
      ``memory_manager`` after the reply.

    Optional ``rewrite`` is inserted for contextualized factual search when chat
    disambiguation is needed (see ``query_understanding``).
    """
    intent = (state.get("query_intent") or "").lower().strip()
    if intent == "greeting":
        return "direct_synthesize"
    if intent == "session_recall":
        return "session_recall"
    # Never vector-search chat-about-self, even if needs_retrieval was mis-set
    if intent in ("meta", "follow_up", "preference", "general_knowledge"):
        return "direct_synthesize"
    if not state.get("needs_retrieval", True):
        return "direct_synthesize"
    if intent == "multi_part":
        return "decompose"
    if state.get("needs_rewrite"):
        return "rewrite"
    return "retrieve"


def route_after_rewriting(state: CogniFlowState) -> RouteAfterRewriting:
    """After contextual rewrite, optionally decompose multi-part queries before retrieval."""
    if (state.get("query_intent") or "").lower().strip() == "multi_part":
        return "decompose"
    return "retrieve"


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    *,
    enable_checkpointing: bool = True,
):
    """Compile the CogniFlow LangGraph with retrieval, synthesis, summarization, and memory."""
    g = StateGraph(CogniFlowState)
    g.add_node("query_understanding", query_understanding_node)
    g.add_node("query_rewriting", query_rewriting_node)
    g.add_node("query_decomposemr", query_decomposer_node)
    g.add_node("retrieval_router", retrieval_router_node)
    g.add_node("context_validation", context_validation_node)
    g.add_node("context_synthesis", context_synthesis_node)
    g.add_node("session_recall", session_recall_node)
    g.add_node("conversation_summarizer", conversation_summarizer_node)
    g.add_node("memory_manager", memory_manager_node)

    g.add_edge(START, "query_understanding")
    g.add_conditional_edges(
        "query_understanding",
        route_after_understanding,
        {
            "direct_synthesize": "context_synthesis",
            "session_recall": "session_recall",
            "rewrite": "query_rewriting",
            "retrieve": "retrieval_router",
            "decompose": "query_decomposer",
        },
    )
    g.add_edge("session_recall", "conversation_summarizer")
    g.add_conditional_edges(
        "query_rewriting",
        route_after_rewriting,
        {
            "decompose": "query_decomposer",
            "retrieve": "retrieval_router",
        },
    )
    g.add_edge("query_decomposer", "retrieval_router")
    g.add_edge("retrieval_router", "context_validation")
    g.add_edge("context_validation", "context_synthesis")
    g.add_edge("context_synthesis", "conversation_summarizer")
    g.add_edge("conversation_summarizer", "memory_manager")
    g.add_edge("memory_manager", END)

    if not enable_checkpointing:
        return g.compile()
    return g.compile(checkpointer=checkpointer or build_checkpointer())


class CogniFlowOrchestrator:
    """LangGraph RAG pipeline with checkpointed thread state."""

    def __init__(self, checkpointer: BaseCheckpointSaver | None = None):
        self._checkpointer = checkpointer
        self.graph = build_graph(checkpointer=checkpointer)

    def prepare_graph_payload(self, state: AgentState) -> dict[str, Any]:
        """Build initial mutable graph dict from API ``AgentState``."""
        payload = agent_state_to_graph(state)
        payload.setdefault("memory_updates", [])
        payload.setdefault("agent_log", [])
        payload.setdefault("sub_queries", [])
        payload.setdefault("use_retrieved_context", True)
        payload.setdefault("context_validation_reason", "")
        return payload

    # Streaming helpers (run_until_before_synthesis, iter_streaming_synthesis,
    # finalize_after_synthesis) run nodes outside the compiled graph so tokens
    # can stream from the LLM. After finalize_after_synthesis, we persist the
    # resulting state with graph.update_state(...) so SQLite checkpoints align
    # with the non-streaming invoke() path for the same thread_id (session_id).

    def run_until_before_synthesis(self, graph_state: dict[str, Any]) -> None:
        """Execute QU → optional QR/QD → retrieval; skip synthesis (used for SSE streaming)."""
        merge_graph_patch(graph_state, query_understanding_node(graph_state))
        branch = route_after_understanding(graph_state)
        if branch == "direct_synthesize":
            return
        if branch == "session_recall":
            merge_graph_patch(graph_state, session_recall_node(graph_state))
            return
        if branch == "rewrite":
            merge_graph_patch(graph_state, query_rewriting_node(graph_state))
            br2 = route_after_rewriting(graph_state)
            if br2 == "decompose":
                merge_graph_patch(graph_state, query_decomposer_node(graph_state))
            merge_graph_patch(graph_state, retrieval_router_node(graph_state))
            merge_graph_patch(graph_state, context_validation_node(graph_state))
            return
        if branch == "decompose":
            merge_graph_patch(graph_state, query_decomposer_node(graph_state))
            merge_graph_patch(graph_state, retrieval_router_node(graph_state))
            merge_graph_patch(graph_state, context_validation_node(graph_state))
            return
        merge_graph_patch(graph_state, retrieval_router_node(graph_state))
        merge_graph_patch(graph_state, context_validation_node(graph_state))

    def apply_context_synthesis(self, graph_state: dict[str, Any]) -> None:
        """Non-streaming synthesis node (same semantics as compiled graph)."""
        if (graph_state.get("query_intent") or "").lower().strip() == "session_recall":
            merge_graph_patch(graph_state, session_recall_node(graph_state))
        else:
            merge_graph_patch(graph_state, context_synthesis_node(graph_state))

    def finalize_after_synthesis(self, graph_state: dict[str, Any]) -> None:
        """Run summarizer + memory manager after the assistant reply exists."""
        merge_graph_patch(graph_state, conversation_summarizer_node(graph_state))
        merge_graph_patch(graph_state, memory_manager_node(graph_state))
        self._persist_streaming_checkpoint(graph_state)

    def _persist_streaming_checkpoint(self, graph_state: dict[str, Any]) -> None:
        """Mirror LangGraph checkpoint state after a manually streamed turn."""
        if getattr(self.graph, "checkpointer", None) is None:
            return
        sid = (graph_state.get("session_id") or "").strip()
        if not sid:
            return
        try:
            self.graph.update_state(
                {"configurable": {"thread_id": sid}},
                graph_state,
                as_node="memory_manager",
            )
        except Exception as exc:
            logger.warning(
                "streaming checkpoint persist failed (non-fatal): %s",
                exc,
                exc_info=True,
            )

    def iter_streaming_synthesis(self, graph_state: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yield token and complete events from streaming LLM synthesis."""
        if (graph_state.get("query_intent") or "").lower().strip() == "session_recall":
            yield from iter_session_recall_events(graph_state)
            return
        yield from iter_context_synthesis_events(graph_state)

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
        payload.setdefault("sub_queries", [])
        payload.setdefault("use_retrieved_context", True)
        payload.setdefault("context_validation_reason", "")
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

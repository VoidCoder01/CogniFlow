from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from typing_extensions import TypedDict

from core.models import AgentState, ChatMessage, MessageRole, QueryIntent, RetrievalStrategy


class CogniFlowState(TypedDict, total=False):
    """LangGraph state: reducers on *_log fields append across nodes (advanced LangGraph pattern)."""

    session_id: str
    user_id: str
    user_query: str
    conversation_history: list[dict[str, Any]]
    query_intent: str
    needs_history: bool
    needs_rewrite: bool
    rewritten_query: str
    retrieval_strategy: str
    retrieved_documents: list[dict[str, Any]]
    synthesized_context: str
    response: str
    should_summarize: bool
    conversation_summary: str
    memory_updates: Annotated[list[dict[str, Any]], operator.add]
    agent_log: Annotated[list[dict[str, Any]], operator.add]


def chat_message_to_dict(m: ChatMessage) -> dict[str, Any]:
    return {
        "id": m.id,
        "role": m.role.value,
        "content": m.content,
        "timestamp": m.timestamp.isoformat(),
        "metadata": m.metadata,
    }


def dict_to_chat_message(d: dict[str, Any]) -> ChatMessage:
    from datetime import datetime
    from uuid import uuid4

    ts = d.get("timestamp")
    if isinstance(ts, str):
        t = datetime.fromisoformat(ts)
    else:
        t = datetime.utcnow()
    return ChatMessage(
        id=str(d.get("id") or uuid4()),
        role=MessageRole(d["role"]),
        content=str(d["content"]),
        timestamp=t,
        metadata=dict(d.get("metadata") or {}),
    )


def agent_state_to_graph(s: AgentState) -> CogniFlowState:
    return {
        "session_id": s.session_id,
        "user_id": s.user_id,
        "user_query": s.user_query,
        "conversation_history": [chat_message_to_dict(m) for m in s.conversation_history],
        "query_intent": s.query_intent.value if s.query_intent else "",
        "needs_history": s.needs_history,
        "needs_rewrite": s.needs_rewrite,
        "rewritten_query": s.rewritten_query,
        "retrieval_strategy": s.retrieval_strategy.value,
        "retrieved_documents": list(s.retrieved_documents),
        "synthesized_context": s.synthesized_context,
        "response": s.response,
        "should_summarize": s.should_summarize,
        "conversation_summary": s.conversation_summary,
        "memory_updates": list(s.memory_updates),
        "agent_log": list(s.agent_log),
    }


def graph_to_agent_state(base: AgentState, g: dict[str, Any]) -> AgentState:
    qi = g.get("query_intent") or ""
    try:
        intent = QueryIntent(qi) if qi else None
    except ValueError:
        intent = QueryIntent.factual

    rs = g.get("retrieval_strategy") or "semantic"
    try:
        strat = RetrievalStrategy(rs)
    except ValueError:
        strat = RetrievalStrategy.semantic

    hist_raw = g.get("conversation_history") or []
    history: list[ChatMessage] = []
    for x in hist_raw:
        if isinstance(x, ChatMessage):
            history.append(x)
        elif isinstance(x, dict):
            history.append(dict_to_chat_message(x))

    return base.model_copy(
        update={
            "conversation_history": history,
            "query_intent": intent,
            "needs_history": bool(g.get("needs_history", False)),
            "needs_rewrite": bool(g.get("needs_rewrite", False)),
            "rewritten_query": str(g.get("rewritten_query") or ""),
            "retrieval_strategy": strat,
            "retrieved_documents": list(g.get("retrieved_documents") or []),
            "synthesized_context": str(g.get("synthesized_context") or ""),
            "response": str(g.get("response") or ""),
            "should_summarize": bool(g.get("should_summarize", False)),
            "conversation_summary": str(g.get("conversation_summary") or ""),
            "memory_updates": list(g.get("memory_updates") or []),
            "agent_log": list(g.get("agent_log") or []),
        }
    )


RouteAfterUnderstanding = Literal["rewrite", "retrieve", "direct_synthesize"]

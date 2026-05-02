from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.graph_state import CogniFlowState
from agents.schemas import RetrievalRoutingResult
from core.llm_provider import get_chat_model

logger = logging.getLogger(__name__)

_SYSTEM = """Pick retrieval strategy for the user's information need against a vector store of technical docs.
Return JSON: strategy (semantic|keyword|hybrid|none), rationale (short).
Use none for greetings or when no doc lookup is needed.
Use keyword for exact identifiers, error codes, function names. Use hybrid when both matter."""


def _heuristic_strategy(intent: str, query: str) -> str:
    q = (query or "").lower()
    if intent in ("greeting", "off_topic"):
        return "none"
    if any(c.isupper() for c in query) and len(query.split()) <= 6:
        return "keyword"
    if "_" in query or "-" in query or any(x in q for x in ("err_", "error", "0x", "econn")):
        return "hybrid"
    return "semantic"


def retrieval_router_node(state: CogniFlowState) -> dict[str, Any]:
    intent = (state.get("query_intent") or "factual").lower()
    base_q = state.get("rewritten_query") or state.get("user_query") or ""

    model = get_chat_model().with_structured_output(RetrievalRoutingResult)
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(
            content=f"intent={intent}\nquery={base_q}",
        ),
    ]
    try:
        out: RetrievalRoutingResult = model.invoke(messages)
        strategy = (out.strategy or "semantic").lower().strip()
        rationale = out.rationale or ""
    except Exception as exc:
        logger.warning("retrieval_router LLM routing failed; using heuristic: %s", exc)
        strategy = _heuristic_strategy(intent, base_q)
        rationale = "heuristic_fallback"

    allowed = {"semantic", "keyword", "hybrid", "none"}
    if strategy not in allowed:
        strategy = _heuristic_strategy(intent, base_q)

    retrieved: list[dict[str, Any]] = []
    if strategy != "none":
        try:
            from core.vector_store import VectorStore

            vs = VectorStore()
            q = base_q
            if strategy == "semantic":
                retrieved = vs.semantic_search(q, top_k=5)
            elif strategy == "keyword":
                retrieved = vs.keyword_search(q, top_k=5)
            else:
                retrieved = vs.hybrid_search(q, top_k=5)
        except Exception as exc:
            logger.warning("Vector retrieval failed: %s", exc)

    log_entry = {
        "node": "retrieval_router",
        "retrieval_strategy": strategy,
        "rationale": rationale,
        "num_docs": len(retrieved),
    }
    return {
        "retrieval_strategy": strategy,
        "retrieved_documents": retrieved,
        "agent_log": [log_entry],
    }

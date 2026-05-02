from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.graph_state import CogniFlowState
from agents.node_utils import with_log_timing
from agents.retrieval_hints import (
    corpus_available_for_chat,
    query_suggests_document_lookup,
)
from agents.schemas import RetrievalRoutingResult
from core.llm_provider import get_chat_model
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)

_SYSTEM = """Pick retrieval strategy for the user's information need against a vector store of technical docs.
Return JSON: strategy (semantic|keyword|hybrid|none), rationale (short).
Use none for greetings or when no doc lookup is needed.
Use keyword for exact identifiers, error codes, function names. Use hybrid when both matter."""


@lru_cache(maxsize=1)
def _get_vector_store() -> VectorStore:
    """Reuse one ``VectorStore`` per process (Chroma client + collection)."""
    return VectorStore()


def _heuristic_strategy(intent: str, query: str) -> str:
    """Pick retrieval strategy without LLM routing."""
    q = (query or "").lower()
    if intent in ("greeting", "off_topic"):
        return "none"
    if "_" in query or "-" in query or any(x in q for x in ("err_", "error", "0x", "econn")):
        return "hybrid"
    if any(c.isupper() for c in query) and len(query.split()) <= 6:
        return "keyword"
    return "semantic"


def _distance_key(doc: dict[str, Any]) -> float:
    d = doc.get("distance")
    try:
        return float(d) if d is not None else 1e9
    except (TypeError, ValueError):
        return 1e9


def _merge_retrieved_docs(
    doc_lists: list[list[dict[str, Any]]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Merge multi-query results, keeping best distance per chunk id."""
    merged: dict[str, dict[str, Any]] = {}
    for docs in doc_lists:
        for d in docs:
            if not isinstance(d, dict):
                continue
            did = d.get("id")
            if did is None:
                continue
            prev = merged.get(did)
            if prev is None or _distance_key(d) < _distance_key(prev):
                merged[did] = d
    ranked = sorted(merged.values(), key=_distance_key)
    return ranked[:top_k]


def retrieval_router_node(state: CogniFlowState) -> dict[str, Any]:
    """Choose retrieval strategy, run vector search (possibly over sub-queries), attach chunks."""
    t0 = time.perf_counter()
    intent = (state.get("query_intent") or "factual").lower()
    base_q = state.get("rewritten_query") or state.get("user_query") or ""
    sub_queries = state.get("sub_queries") or []
    queries: list[str] = []
    if sub_queries:
        queries = [str(s).strip() for s in sub_queries if str(s).strip()]
    if not queries:
        queries = [base_q.strip() or base_q]

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

    sid = (state.get("session_id") or "").strip()
    uid = (state.get("user_id") or "").strip()
    if strategy == "none" and sid and corpus_available_for_chat(sid, uid):
        strategy = "semantic"
        rationale = f"{rationale} | session_corpus".strip()
    elif strategy == "none" and sid:
        if query_suggests_document_lookup(base_q):
            strategy = "semantic"
            rationale = f"{rationale} | session_doc_lookup".strip()
        elif intent in (
            "factual",
            "follow_up",
            "clarification",
            "comparison",
            "multi_part",
        ) and len((base_q or "").strip()) > 8:
            strategy = "semantic"
            rationale = f"{rationale} | intent_needs_retrieval".strip()

    retrieved_lists: list[list[dict[str, Any]]] = []
    if strategy != "none":
        try:
            vs = _get_vector_store()
            filt = VectorStore.scope_filter(sid, uid)
            for q in queries:
                q = q.strip()
                if not q:
                    continue
                if strategy == "semantic":
                    retrieved_lists.append(
                        vs.semantic_search(q, top_k=5, filter_metadata=filt)
                    )
                elif strategy == "keyword":
                    retrieved_lists.append(
                        vs.keyword_search(q, top_k=5, filter_metadata=filt)
                    )
                else:
                    retrieved_lists.append(
                        vs.hybrid_search(q, top_k=5, filter_metadata=filt)
                    )
            retrieved = (
                _merge_retrieved_docs(retrieved_lists, top_k=5)
                if len(retrieved_lists) > 1
                else (retrieved_lists[0] if retrieved_lists else [])
            )
        except Exception as exc:
            logger.warning("Vector retrieval failed: %s", exc)
            retrieved = []
    else:
        retrieved = []

    log_entry = with_log_timing(
        {
            "node": "retrieval_router",
            "retrieval_strategy": strategy,
            "rationale": rationale,
            "num_docs": len(retrieved),
            "num_sub_queries": len(queries),
        },
        t0,
    )
    return {
        "retrieval_strategy": strategy,
        "retrieved_documents": retrieved,
        "agent_log": [log_entry],
    }

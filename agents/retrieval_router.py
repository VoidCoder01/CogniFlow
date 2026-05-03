from __future__ import annotations

import logging
import time
from typing import Any

_RETRIEVAL_VS_FAIL_MONO: float = 0.0
_RETRIEVAL_VS_RETRY_SEC = 10.0

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

_MAX_MERGED_DOCS = 8

_retrieval_vs: VectorStore | None = None
_retrieval_vs_failed = False


def reset_retrieval_vector_store() -> None:
    """Clear cached VectorStore for tests or after fixing Chroma on disk."""
    global _retrieval_vs, _retrieval_vs_failed, _RETRIEVAL_VS_FAIL_MONO
    _retrieval_vs = None
    _retrieval_vs_failed = False
    _RETRIEVAL_VS_FAIL_MONO = 0.0


def _get_vector_store() -> VectorStore | None:
    """Reuse one ``VectorStore`` per process; return ``None`` if Chroma catalog cannot load."""
    global _retrieval_vs, _retrieval_vs_failed, _RETRIEVAL_VS_FAIL_MONO
    if _retrieval_vs is not None:
        return _retrieval_vs
    now = time.monotonic()
    if _retrieval_vs_failed and now - _RETRIEVAL_VS_FAIL_MONO < _RETRIEVAL_VS_RETRY_SEC:
        return None
    if _retrieval_vs_failed:
        logger.info("Retrying retrieval VectorStore after cooldown (Chroma may have been reset)")
        _retrieval_vs_failed = False
    try:
        _retrieval_vs = VectorStore()
        return _retrieval_vs
    except Exception as exc:
        logger.warning(
            "VectorStore unavailable for retrieval (answers may lack doc grounding): %s",
            exc,
        )
        _retrieval_vs_failed = True
        _RETRIEVAL_VS_FAIL_MONO = time.monotonic()
        return None


def _heuristic_strategy(intent: str, query: str) -> str:
    """Pick retrieval strategy without LLM routing."""
    q = (query or "").lower()
    if intent in (
        "greeting",
        "off_topic",
        "general_knowledge",
        "meta",
        "preference",
        "session_recall",
    ):
        return "none"
    if "_" in query or "-" in query or any(x in q for x in ("err_", "error", "0x", "econn")):
        return "hybrid"
    if any(c.isupper() for c in query) and len(query.split()) <= 6:
        return "keyword"
    return "semantic"


def retrieval_router_node(state: CogniFlowState) -> dict[str, Any]:
    """Choose retrieval strategy, run vector search (possibly over sub-queries), attach chunks."""
    t0 = time.perf_counter()
    intent = (state.get("query_intent") or "factual").lower()
    base_q = state.get("rewritten_query") or state.get("user_query") or ""
    sid = (state.get("session_id") or "").strip()
    uid = (state.get("user_id") or "").strip()

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

    if strategy == "none" and sid and corpus_available_for_chat(sid, uid):
        strategy = "semantic"
        rationale = f"{rationale} | session_corpus".strip()
    elif strategy == "none" and sid:
        if query_suggests_document_lookup(base_q):
            strategy = "semantic"
            rationale = f"{rationale} | session_doc_lookup".strip()
        elif intent in (
            "factual",
            "factual_doc",
            "follow_up",
            "clarification",
            "comparison",
            "multi_part",
        ) and len((base_q or "").strip()) > 8:
            strategy = "semantic"
            rationale = f"{rationale} | intent_needs_retrieval".strip()

    retrieved: list[dict[str, Any]] = []
    queries_to_run: list[str] = []

    if strategy != "none":
        raw_sub = state.get("sub_queries") or []
        stripped_sub = [str(s).strip() for s in raw_sub if str(s).strip()]
        queries_to_run = stripped_sub if len(stripped_sub) > 1 else [base_q.strip() or base_q]
        queries_to_run = [q for q in queries_to_run if q and str(q).strip()]

        vs = _get_vector_store()
        if vs is None:
            retrieved = []
        else:
            try:
                filt = VectorStore.scope_filter(sid, uid)
                seen_ids: set[str] = set()
                for q in queries_to_run:
                    qt = str(q).strip()
                    if not qt:
                        continue
                    if strategy == "semantic":
                        hits = vs.semantic_search(qt, top_k=5, filter_metadata=filt)
                    elif strategy == "keyword":
                        hits = vs.keyword_search(qt, top_k=5, filter_metadata=filt)
                    else:
                        hits = vs.hybrid_search(qt, top_k=5, filter_metadata=filt)
                    for doc in hits:
                        if not isinstance(doc, dict):
                            continue
                        doc_id = str(doc.get("id", "") or "")
                        if doc_id and doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            retrieved.append(doc)

                def _dist_key(d: dict[str, Any]) -> float:
                    try:
                        return float(d.get("distance") or 999)
                    except (TypeError, ValueError):
                        return 999.0

                retrieved.sort(key=_dist_key)
                retrieved = retrieved[:_MAX_MERGED_DOCS]
            except Exception as exc:
                logger.warning("Vector retrieval failed: %s", exc)
                retrieved = []

    log_entry = with_log_timing(
        {
            "node": "retrieval_router",
            "retrieval_strategy": strategy,
            "rationale": rationale,
            "num_docs": len(retrieved),
            "num_sub_queries": len(queries_to_run) if strategy != "none" else 0,
        },
        t0,
    )
    return {
        "retrieval_strategy": strategy,
        "retrieved_documents": retrieved,
        "agent_log": [log_entry],
    }

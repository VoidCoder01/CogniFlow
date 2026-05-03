"""Decomposes multi-part questions into sub-queries for independent retrieval."""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.graph_state import CogniFlowState
from agents.node_utils import with_log_timing
from agents.schemas import QueryDecompositionResult
from core.llm_provider import get_chat_model

logger = logging.getLogger(__name__)

_SYSTEM = """The user asked a multi-part question. Decompose it into 2-4 independent sub-queries, each answerable on its own from a technical documentation knowledge base.

RULES:
1. Each sub-query should be a complete, standalone question.
2. Comparison questions ("X vs Y") should become separate queries for each item, NOT a single comparison query (retrieval works best with focused queries).
3. If the question is actually single-part, return it unchanged.
4. Return JSON only: {"sub_queries": ["query1", "query2", ...]}

EXAMPLES:
"What is FastAPI and how does Django handle authentication?"
→ {"sub_queries": ["What is FastAPI?", "How does Django handle authentication?"]}

"Compare Docker and Kubernetes for microservices deployment"
→ {"sub_queries": ["Docker for microservices deployment", "Kubernetes for microservices deployment"]}

"How does connection pooling work?"
→ {"sub_queries": ["How does connection pooling work?"]}"""


def query_decomposer_node(state: CogniFlowState) -> dict[str, Any]:
    """Split multi-part questions into sub-queries for retrieval; fallback is the original query."""
    t0 = time.perf_counter()
    query = state.get("rewritten_query") or state.get("user_query") or ""

    try:
        model = get_chat_model().with_structured_output(QueryDecompositionResult)
        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=f"Question: {query}"),
        ]
        out = model.invoke(messages)
        sub_queries = out.sub_queries or [query]
    except Exception as exc:
        logger.warning("query_decomposer failed: %s", exc)
        sub_queries = [query]

    log_entry = with_log_timing(
        {
            "node": "query_decomposer",
            "num_sub_queries": len(sub_queries),
            "sub_queries": sub_queries,
        },
        t0,
    )
    return {
        "sub_queries": sub_queries,
        "agent_log": [log_entry],
    }

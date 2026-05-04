from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.graph_state import CogniFlowState
from agents.node_utils import with_log_timing
from agents.prompt_suite import RETRIEVAL_GUARD_SYSTEM
from agents.schemas import ContextValidationResult
from core.llm_provider import get_chat_model

logger = logging.getLogger(__name__)


def _format_retrieved_context(docs: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, d in enumerate(docs[:8], 1):
        if not isinstance(d, dict):
            continue
        body = (d.get("content") or "").strip()
        if not body:
            continue
        meta = d.get("metadata") or {}
        src = meta.get("source") or meta.get("title") or "source"
        lines.append(f"[{i}] {src}\n{body[:4000]}")
    return "\n\n".join(lines) if lines else ""


def _nonempty_chunks(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        d
        for d in (docs or [])
        if isinstance(d, dict) and (d.get("content") or "").strip()
    ]


def context_validation_node(state: CogniFlowState) -> dict[str, Any]:
    """Decide if retrieved chunks are sufficient; if not, synthesis falls back to general knowledge."""
    t0 = time.perf_counter()
    docs = _nonempty_chunks(state.get("retrieved_documents") or [])
    user_q = str(state.get("user_query") or "")

    if not docs:
        log_entry = with_log_timing(
            {
                "node": "context_validation",
                "use_retrieved_context": False,
                "reason": "empty_context",
                "skipped_llm": True,
            },
            t0,
        )
        return {
            "use_retrieved_context": False,
            "context_validation_reason": "empty_context",
            "agent_log": [log_entry],
        }

    context_block = _format_retrieved_context(docs)
    if not context_block.strip():
        log_entry = with_log_timing(
            {
                "node": "context_validation",
                "use_retrieved_context": False,
                "reason": "empty_context_after_format",
                "skipped_llm": True,
            },
            t0,
        )
        return {
            "use_retrieved_context": False,
            "context_validation_reason": "empty_context_after_format",
            "agent_log": [log_entry],
        }

    try:
        model = get_chat_model().with_structured_output(ContextValidationResult)
        messages = [
            SystemMessage(content=RETRIEVAL_GUARD_SYSTEM),
            HumanMessage(
                content=(
                    "User query:\n"
                    f"{user_q}\n\n"
                    "Retrieved context:\n"
                    f"{context_block}"
                )
            ),
        ]
        out: ContextValidationResult = model.invoke(messages)
        use_ctx = bool(out.use_context)
        reason = (out.reason or "").strip() or ("validated" if use_ctx else "model_rejected_context")
    except Exception as exc:
        logger.warning(
            "context_validation structured output failed: %s", exc, exc_info=True
        )
        use_ctx = False
        reason = "llm_validation_failed_prefer_knowledge_fallback"

    log_entry = with_log_timing(
        {
            "node": "context_validation",
            "use_retrieved_context": use_ctx,
            "reason": reason,
        },
        t0,
    )
    return {
        "use_retrieved_context": use_ctx,
        "context_validation_reason": reason,
        "agent_log": [log_entry],
    }

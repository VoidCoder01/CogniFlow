from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.graph_state import CogniFlowState
from agents.node_utils import with_log_timing
from agents.schemas import MemoryExtractionResult
from core.llm_provider import get_chat_model

logger = logging.getLogger(__name__)

_SYSTEM = """Extract durable user-specific facts worth remembering across future chat sessions.
Return JSON: {"items": [{"memory_type": "...", "content": "...", "metadata": {}}]}

MEMORY TYPES:
- preference: tools, languages, frameworks the user prefers ("I use TypeScript", "I prefer PostgreSQL over MySQL")
- context: project details, tech stack, role ("I'm building a SaaS app", "I'm a junior developer")
- decision: choices the user made during this conversation ("decided to use Redis for caching")
- issue: problems or errors the user is dealing with ("getting CORS errors in production")

WHAT TO EXTRACT: facts that would help a future assistant personalize responses.
WHAT TO SKIP: transient questions, greetings, facts already in the conversation (don't re-extract what was just said — only new durable facts).

Return {"items": []} if the latest exchange contains no new durable facts. Most exchanges won't — that's fine."""


def memory_manager_node(state: CogniFlowState) -> dict[str, Any]:
    """Extract structured long-term memories from the latest assistant turn."""
    t0 = time.perf_counter()
    intent = (state.get("query_intent") or "").lower().strip()
    # Only skip obvious pleasantries. Do not skip off_topic: misclassified turns often
    # carry stack/product/role facts; the extractor returns [] when there is nothing to store.
    if intent == "greeting":
        log_entry = with_log_timing(
            {
                "node": "memory_manager",
                "skipped": True,
                "reason": "greeting_turn",
            },
            t0,
        )
        return {"memory_updates": [], "agent_log": [log_entry]}

    # Deterministic recap / empty-thread replies carry no new durable user facts to extract.
    if intent == "session_recall":
        log_entry = with_log_timing(
            {
                "node": "memory_manager",
                "skipped": True,
                "reason": "session_recall_turn",
            },
            t0,
        )
        return {"memory_updates": [], "agent_log": [log_entry]}

    model = get_chat_model().with_structured_output(MemoryExtractionResult)
    history = state.get("conversation_history") or []
    hist_snip = "\n".join(
        f'{m.get("role", "user")}: {m.get("content", "")}'
        for m in history[-16:]
    )
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(
            content=(
                f"user_id={state.get('user_id')}\n"
                f"Latest assistant reply:\n{state.get('response', '')}\n\n"
                f"Recent conversation:\n{hist_snip or '(empty)'}"
            )
        ),
    ]
    try:
        out: MemoryExtractionResult = model.invoke(messages)
        items = [i.model_dump() for i in (out.items or [])]
    except Exception as exc:
        logger.warning(
            "memory_manager structured output failed: %s", exc, exc_info=True
        )
        items = []

    log_entry = with_log_timing(
        {"node": "memory_manager", "num_items": len(items)}, t0
    )
    return {
        "memory_updates": items,
        "agent_log": [log_entry],
    }

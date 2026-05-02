from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.graph_state import CogniFlowState
from agents.schemas import MemoryExtractionResult
from core.llm_provider import get_chat_model

logger = logging.getLogger(__name__)

_SYSTEM = """Extract durable user-specific facts worth storing for later sessions.
Return JSON with items: list of {memory_type, content, metadata}.
memory_type must be one of: preference, context, decision, issue.
Only include high-signal facts; otherwise return items: []."""


def memory_manager_node(state: CogniFlowState) -> dict[str, Any]:
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
        logger.warning("memory_manager structured output failed: %s", exc)
        items = []

    log_entry = {"node": "memory_manager", "num_items": len(items)}
    return {
        "memory_updates": items,
        "agent_log": [log_entry],
    }

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.graph_state import CogniFlowState
from core.llm_provider import get_chat_model

logger = logging.getLogger(__name__)

_SYSTEM = """Rewrite the user query into a standalone search query for a technical knowledge base.
Resolve pronouns using the conversation. Output only the rewritten query text, no quotes or preamble."""


def query_rewriting_node(state: CogniFlowState) -> dict[str, Any]:
    model = get_chat_model()
    history = state.get("conversation_history") or []
    hist_snip = "\n".join(
        f'{m.get("role", "user")}: {m.get("content", "")}'
        for m in history[-10:]
    )
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(
            content=(
                f"Recent conversation:\n{hist_snip or '(empty)'}\n\n"
                f"Original query:\n{state.get('user_query', '')}"
            )
        ),
    ]
    resp = model.invoke(messages)
    text = (getattr(resp, "content", None) or str(resp)).strip()
    log_entry = {"node": "query_rewriting", "rewritten_query": text}
    logger.debug("query_rewriting: %s", log_entry)
    return {
        "rewritten_query": text,
        "agent_log": [log_entry],
    }

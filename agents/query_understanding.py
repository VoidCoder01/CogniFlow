from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.graph_state import CogniFlowState
from agents.schemas import QueryUnderstandingResult
from core.llm_provider import get_chat_model
from core.models import QueryIntent

logger = logging.getLogger(__name__)

_SYSTEM = """You classify the user's latest message for a technical documentation assistant.
Return JSON matching the schema: intent, needs_history, needs_rewrite.
intent must be exactly one of: factual, follow_up, clarification, comparison, multi_part, greeting, off_topic.
- greeting: hi, hello, thanks with no technical question
- off_topic: unrelated chit-chat
- follow_up: refers to prior turns (pronouns, "that", "it") and needs history
- needs_rewrite: query is ambiguous without conversation context
"""


def _normalize_intent(raw: str) -> str:
    v = (raw or "").strip().lower().replace("-", "_")
    allowed = {e.value for e in QueryIntent}
    if v in allowed:
        return v
    # minimal aliases
    aliases = {
        "followup": "follow_up",
        "offtopic": "off_topic",
        "multi part": "multi_part",
    }
    return aliases.get(v, "factual")


def query_understanding_node(state: CogniFlowState) -> dict[str, Any]:
    model = get_chat_model().with_structured_output(QueryUnderstandingResult)
    history = state.get("conversation_history") or []
    hist_snip = "\n".join(
        f'{m.get("role", "user")}: {m.get("content", "")}'
        for m in history[-8:]
    )
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(
            content=(
                f"Conversation (recent):\n{hist_snip or '(empty)'}\n\n"
                f"User message:\n{state.get('user_query', '')}"
            )
        ),
    ]
    out: QueryUnderstandingResult = model.invoke(messages)
    intent = _normalize_intent(out.intent)

    log_entry = {
        "node": "query_understanding",
        "intent": intent,
        "needs_history": out.needs_history,
        "needs_rewrite": out.needs_rewrite,
    }
    logger.debug("query_understanding: %s", log_entry)

    return {
        "query_intent": intent,
        "needs_history": out.needs_history,
        "needs_rewrite": out.needs_rewrite,
        "agent_log": [log_entry],
    }

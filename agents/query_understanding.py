from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.graph_state import CogniFlowState
from agents.node_utils import with_log_timing
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
    """Map LLM intent string onto allowed QueryIntent values."""
    v = (raw or "").strip().lower().replace("-", "_")
    allowed = {e.value for e in QueryIntent}
    if v in allowed:
        return v
    aliases = {
        "followup": "follow_up",
        "offtopic": "off_topic",
        "multi part": "multi_part",
    }
    return aliases.get(v, "factual")


def _heuristic_classify(query: str, history_snippet: str) -> tuple[str, bool, bool]:
    """Rule-based fallback when LLM structured output fails."""
    q = (query or "").strip().lower()
    if not q or q in ("hi", "hello", "hey", "thanks", "thank you", "bye"):
        return "greeting", False, False
    pronouns = ("it", "that", "this", "they", "them", "those", "the same")
    has_pronoun_ref = any(f" {p} " in f" {q} " for p in pronouns)
    has_history = bool(history_snippet and history_snippet.strip() != "(empty)")
    if has_pronoun_ref and has_history:
        return "follow_up", True, True
    if "compare" in q or "vs" in q or "versus" in q or "difference between" in q:
        return "comparison", True, False
    if q.count("?") > 1 or (" and " in q and "?" in q):
        return "multi_part", False, False
    return "factual", False, False


def query_understanding_node(state: CogniFlowState) -> dict[str, Any]:
    """Classify user intent and determine if history or query rewriting is needed.

    Uses structured LLM output into seven intents; falls back to rule-based heuristics
    if the LLM call fails (malformed JSON, provider limits).
    """
    t0 = time.perf_counter()
    history = state.get("conversation_history") or []
    hist_snip = "\n".join(
        f'{m.get("role", "user")}: {m.get("content", "")}'
        for m in history[-8:]
    )
    user_q = state.get("user_query", "")

    try:
        model = get_chat_model().with_structured_output(QueryUnderstandingResult)
        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(
                content=(
                    f"Conversation (recent):\n{hist_snip or '(empty)'}\n\n"
                    f"User message:\n{user_q}"
                )
            ),
        ]
        out: QueryUnderstandingResult = model.invoke(messages)
        intent = _normalize_intent(out.intent)
        needs_history = out.needs_history
        needs_rewrite = out.needs_rewrite
    except Exception as exc:
        logger.warning(
            "query_understanding structured output failed, using heuristic: %s", exc
        )
        intent, needs_history, needs_rewrite = _heuristic_classify(user_q, hist_snip)

    log_entry = with_log_timing(
        {
            "node": "query_understanding",
            "intent": intent,
            "needs_history": needs_history,
            "needs_rewrite": needs_rewrite,
        },
        t0,
    )
    logger.debug("query_understanding: %s", log_entry)
    return {
        "query_intent": intent,
        "needs_history": needs_history,
        "needs_rewrite": needs_rewrite,
        "agent_log": [log_entry],
    }

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.graph_state import CogniFlowState
from config import settings
from core.llm_provider import get_chat_model

logger = logging.getLogger(__name__)

def _summary_system_prompt() -> str:
    n = max(3, min(12, settings.summary_max_bullets))
    return (
        f"Summarize the conversation for future context in <= {n} bullet sentences. "
        "Focus on user goals, decisions, and unresolved issues. Output plain text only."
    )


def conversation_summarizer_node(state: CogniFlowState) -> dict[str, Any]:
    history = state.get("conversation_history") or []
    prev = state.get("conversation_summary") or ""
    threshold = settings.summary_threshold
    if len(history) < threshold and not state.get("should_summarize"):
        return {
            "agent_log": [
                {
                    "node": "conversation_summarizer",
                    "skipped": True,
                    "message_count": len(history),
                }
            ],
        }

    model = get_chat_model()
    hist_snip = "\n".join(
        f'{m.get("role", "user")}: {m.get("content", "")}'
        for m in history[-30:]
    )
    messages = [
        SystemMessage(content=_summary_system_prompt()),
        HumanMessage(
            content=(
                f"Previous summary (if any):\n{prev or '(none)'}\n\n"
                f"Messages:\n{hist_snip}"
            )
        ),
    ]
    resp = model.invoke(messages)
    text = (getattr(resp, "content", None) or str(resp)).strip()
    log_entry = {"node": "conversation_summarizer", "summary_chars": len(text)}
    return {
        "conversation_summary": text,
        "should_summarize": False,
        "agent_log": [log_entry],
    }

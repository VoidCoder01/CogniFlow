from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.graph_state import CogniFlowState
from agents.node_utils import with_log_timing
from config import settings
from core.llm_provider import get_chat_model

logger = logging.getLogger(__name__)


def _summary_system_prompt(message_count: int, avg_msg_len: int) -> str:
    """System instructions sized from message volume and ``summary_compression_ratio``."""
    ratio = float(settings.summary_compression_ratio)
    target_chars = int(max(120, message_count * max(avg_msg_len, 20) * ratio))
    target_sentences = max(2, min(12, target_chars // 80))
    return (
        f"Summarize this conversation in approximately {target_sentences} sentences "
        f"(~{target_chars} characters).\n\n"
        "PRESERVE: user goals, decisions made, unresolved issues, specific technical details "
        "(version numbers, error codes, config values, file names).\n"
        "DROP: greetings, filler, repeated questions, assistant explanations that were superseded by later corrections.\n"
        "Output plain text only — no bullet points, no headers, no markdown."
    )


def conversation_summarizer_node(state: CogniFlowState) -> dict[str, Any]:
    """Refresh the rolling conversation summary when enough messages warrant it."""
    t0 = time.perf_counter()
    history = state.get("conversation_history") or []
    prev = state.get("conversation_summary") or ""
    min_msgs = max(2, settings.summary_threshold)
    intent = (state.get("query_intent") or "").lower().strip()
    if intent == "greeting" and len(history) <= 12:
        return {
            "agent_log": [
                with_log_timing(
                    {
                        "node": "conversation_summarizer",
                        "skipped": True,
                        "reason": "greeting_turn",
                        "message_count": len(history),
                    },
                    t0,
                )
            ],
        }
    if len(history) < min_msgs and not state.get("should_summarize"):
        return {
            "agent_log": [
                with_log_timing(
                    {
                        "node": "conversation_summarizer",
                        "skipped": True,
                        "message_count": len(history),
                    },
                    t0,
                )
            ],
        }

    lens = [len(str(m.get("content", ""))) for m in history[-30:]]
    avg_msg_len = int(sum(lens) / max(len(lens), 1))

    model = get_chat_model()
    hist_snip = "\n".join(
        f'{m.get("role", "user")}: {m.get("content", "")}'
        for m in history[-30:]
    )
    messages = [
        SystemMessage(content=_summary_system_prompt(len(history[-30:]), avg_msg_len)),
        HumanMessage(
            content=(
                f"Previous summary (if any):\n{prev or '(none)'}\n\n"
                f"Messages:\n{hist_snip}"
            )
        ),
    ]
    resp = model.invoke(messages)
    text = (getattr(resp, "content", None) or str(resp)).strip()
    log_entry = with_log_timing(
        {"node": "conversation_summarizer", "summary_chars": len(text)}, t0
    )
    return {
        "conversation_summary": text,
        "should_summarize": False,
        "agent_log": [log_entry],
    }

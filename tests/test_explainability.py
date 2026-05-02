"""Tests for explainability helpers."""

from __future__ import annotations

from datetime import datetime

from agents.explainability import collect_agent_logs_from_message_rows
from core.models import ChatMessage, MessageRole


def test_collect_agent_logs_from_message_rows():
    ts = datetime.utcnow()
    msgs = [
        ChatMessage(role=MessageRole.user, content="hi"),
        ChatMessage(
            role=MessageRole.assistant,
            content="hello",
            timestamp=ts,
            metadata={"agent_log": [{"node": "query_understanding"}]},
        ),
    ]
    logs = collect_agent_logs_from_message_rows(msgs)
    assert len(logs) == 1
    assert logs[0]["agent_log"][0]["node"] == "query_understanding"

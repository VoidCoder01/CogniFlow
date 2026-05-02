"""Helpers for exposing LangGraph ``agent_log`` timelines per session message."""

from __future__ import annotations

from typing import Any


def collect_agent_logs_from_message_rows(messages: list[Any]) -> list[dict[str, Any]]:
    """Build ``agent_logs`` payloads compatible with ``GET /sessions/{id}/agent-logs``."""
    logs: list[dict[str, Any]] = []
    for m in messages:
        meta = getattr(m, "metadata", None) or {}
        if not isinstance(meta, dict):
            continue
        if "agent_log" not in meta:
            continue
        logs.append(
            {
                "message_id": getattr(m, "id", ""),
                "timestamp": getattr(m, "timestamp", None),
                "agent_log": meta["agent_log"],
            }
        )
    return logs

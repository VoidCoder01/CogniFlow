"""Small helpers for LangGraph nodes (timing, shared heuristics)."""

from __future__ import annotations

import time
from typing import Any


def elapsed_since(t0: float) -> float:
    return round(time.perf_counter() - t0, 4)


def with_log_timing(log_entry: dict[str, Any], t0: float) -> dict[str, Any]:
    """Mutate log_entry with elapsed_seconds and return it."""
    log_entry["elapsed_seconds"] = elapsed_since(t0)
    return log_entry

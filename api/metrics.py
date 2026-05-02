from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any


class RequestMetrics:
    """Thread-safe rolling latency and request counters for `/stats` and `/metrics`."""

    def __init__(self, max_samples: int = 200):
        self._lock = threading.Lock()
        self._latencies: deque[float] = deque(maxlen=max_samples)
        self.chat_requests = 0
        self.upload_requests = 0
        self.errors = 0

    def record_chat_latency(self, seconds: float) -> None:
        with self._lock:
            self._latencies.append(seconds)
            self.chat_requests += 1

    def record_upload(self) -> None:
        with self._lock:
            self.upload_requests += 1

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self._latencies)
            n = len(samples)
            avg = sum(samples) / n if n else 0.0
            sorted_s = sorted(samples)
            p95 = sorted_s[int(0.95 * (n - 1))] if n else 0.0
            return {
                "chat_requests_total": self.chat_requests,
                "upload_requests_total": self.upload_requests,
                "errors_total": self.errors,
                "chat_latency_samples": n,
                "chat_latency_avg_seconds": round(avg, 4),
                "chat_latency_p95_seconds": round(p95, 4),
                "last_updated_unix": time.time(),
            }


request_metrics = RequestMetrics()

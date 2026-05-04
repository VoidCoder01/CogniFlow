"""Client-safe error messages and SSE payloads (no stack traces in production)."""

from __future__ import annotations

from typing import Any

from config import settings


def safe_client_error_detail(exc: BaseException) -> str:
    """
    Map an exception to text safe to return in HTTP ``detail`` or SSE ``error`` events.

    Full tracebacks are logged server-side; clients get actionable hints only when
    ``settings.expose_internal_errors`` is true or the error is already user-facing
    (e.g. provider auth / rate limits).
    """
    if settings.expose_internal_errors:
        return (str(exc) or "Request failed")[:2000]
    if isinstance(exc, ModuleNotFoundError):
        return (
            f"{exc} Install dependencies: `pip install -r requirements.txt` "
            "(provider packages must match LLM_PROVIDER)."
        )[:1500]
    if isinstance(exc, ImportError) and "ContextOverflowError" in str(exc):
        return (
            "LangChain package versions are incompatible (e.g. langchain-core too old for "
            "langchain-anthropic). Fix: `pip install -r requirements.txt` or upgrade "
            "`langchain-core` to >=1.3.0 to match langgraph."
        )[:1500]
    t = (str(exc) or "").strip()
    low = t.lower()
    if isinstance(exc, ValueError) and (
        "api_key" in low or "required when llm" in low or "unsupported llm_provider" in low
    ):
        return t[:1500]
    if any(
        x in low
        for x in (
            "401",
            "403",
            "429",
            "400",
            "incorrect api key",
            "invalid api key",
            "authentication",
            "rate limit",
            "dimension",
            "embedding",
            "credit balance",
            "too low to access",
            "purchase credits",
            "plans & billing",
            "invalid_request_error",
        )
    ):
        return t[:1500]
    return "The service could not complete this request."


def sse_error_payload(
    exc: BaseException, *, request_id: str | None, status_code: int | None = None
) -> dict[str, Any]:
    """Structured SSE ``error`` event body (JSON-serializable)."""
    out: dict[str, Any] = {
        "event": "error",
        "detail": safe_client_error_detail(exc),
    }
    if request_id:
        out["request_id"] = request_id
    if status_code is not None:
        out["status_code"] = status_code
    return out

"""Production-oriented HTTP middleware."""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach ``X-Request-ID`` (client-supplied or generated) for tracing and logs."""

    async def dispatch(self, request: Request, call_next) -> Response:
        header_rid = request.headers.get("x-request-id") or request.headers.get(
            "X-Request-ID"
        )
        rid = (header_rid or "").strip() or str(uuid.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        response.headers.setdefault("X-Request-ID", rid)
        return response

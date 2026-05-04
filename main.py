"""
CogniFlow FastAPI entrypoint: conversational RAG API with LangGraph orchestration.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.errors import safe_client_error_detail
from api.limiter import limiter
from api.middleware import RequestIDMiddleware
from api.routes import router
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    for path in (
        settings.chroma_persist_dir,
        os.path.dirname(os.path.abspath(settings.sqlite_db_path)) or ".",
        os.path.dirname(os.path.abspath(settings.checkpoint_sqlite_path)) or ".",
    ):
        if path:
            os.makedirs(path, exist_ok=True)
    logger.info(
        "CogniFlow API starting (llm=%s, embeddings=%s/%s, checkpoint=%s)",
        settings.llm_provider,
        settings.embedding_backend,
        settings.embedding_device if settings.embedding_backend == "local" else "api",
        settings.checkpoint_backend,
    )
    yield
    logger.info("CogniFlow API shutdown")


app = FastAPI(
    title="CogniFlow",
    description=(
        "Conversational RAG with LangGraph agents, ChromaDB, and SQLite session memory. "
        "See `/api/v1/health` and `/docs`."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ALLOWED_ORIGINS: comma-separated explicit origins (required when allow_credentials=True).
# Wildcard "*" with credentials is invalid per CORS spec and breaks browsers.
_raw_origins = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8501"
)
_allowed_origins: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Outermost on request: stable id for logs and error responses
app.add_middleware(RequestIDMiddleware)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    rid = getattr(request.state, "request_id", None)
    body: dict = {"detail": exc.detail}
    if rid:
        body["request_id"] = rid
    headers = {k: v for k, v in (exc.headers or {}).items()}
    if rid:
        headers.setdefault("X-Request-ID", rid)
    return JSONResponse(status_code=exc.status_code, content=body, headers=headers)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    rid = getattr(request.state, "request_id", None)
    content: dict = {"detail": exc.errors()}
    if rid:
        content["request_id"] = rid
    headers = {"X-Request-ID": rid} if rid else {}
    return JSONResponse(status_code=422, content=content, headers=headers)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    rid = getattr(request.state, "request_id", None)
    logger.exception(
        "Unhandled exception path=%s request_id=%s",
        request.url.path,
        rid,
    )
    content = {
        "detail": safe_client_error_detail(exc),
    }
    if rid:
        content["request_id"] = rid
    headers = {"X-Request-ID": rid} if rid else {}
    return JSONResponse(status_code=500, content=content, headers=headers)


app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "cogniflow",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )

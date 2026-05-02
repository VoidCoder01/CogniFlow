"""
CogniFlow FastAPI entrypoint: conversational RAG API with LangGraph orchestration.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
        "CogniFlow API starting (llm_provider=%s, checkpoint=%s)",
        settings.llm_provider,
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

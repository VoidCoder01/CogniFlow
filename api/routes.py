from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Annotated, Any, Iterator

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from agents.graph_state import agent_state_to_graph, graph_to_agent_state
from agents.orchestrator import CogniFlowOrchestrator
from api.deps import get_memory_store, get_orchestrator, get_vector_store
from api.metrics import request_metrics
from config import settings
from core.document_processor import DocumentProcessor
from core.memory_store import MemoryStore
from core.models import (
    AgentState,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    DocumentUploadResponse,
    MessageRole,
    SessionCreateRequest,
    SessionCreateResponse,
)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["cogniflow"])


def _doc_distance_to_relevance(distance: float | None) -> float:
    if distance is None:
        return 0.0
    try:
        return max(0.0, min(1.0, 1.0 - float(distance)))
    except (TypeError, ValueError):
        return 0.0


def _sources_from_retrieved(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in docs[:10]:
        if not isinstance(d, dict):
            continue
        meta = d.get("metadata") or {}
        out.append(
            {
                "id": d.get("id"),
                "title": meta.get("title"),
                "source": meta.get("source"),
                "relevance": _doc_distance_to_relevance(d.get("distance")),
            }
        )
    return out


def _format_user_memory_context(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines: list[str] = []
    for r in rows:
        mt = r.get("memory_type") or "context"
        content = (r.get("content") or "").strip()
        if content:
            lines.append(f"- ({mt}) {content}")
    return "\n".join(lines)


def _load_history_window(
    store: MemoryStore,
    session_id: str,
) -> tuple[list[ChatMessage], str]:
    """Return recent messages (sliding window) and session summary from DB."""
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    n = max(4, settings.memory_window_size * 2)
    messages = store.get_recent_messages(session_id, n=n)
    return messages, session.summary or ""


@router.post("/sessions", response_model=SessionCreateResponse)
def create_session(
    body: SessionCreateRequest,
    store: Annotated[MemoryStore, Depends(get_memory_store)],
):
    session = store.create_session(body.user_id)
    return SessionCreateResponse(
        session_id=session.session_id,
        user_id=session.user_id,
        created_at=session.created_at,
    )


@router.get("/users/{user_id}/sessions")
def list_user_sessions(
    user_id: str,
    store: Annotated[MemoryStore, Depends(get_memory_store)],
):
    """List sessions for a user (newest first)."""
    return {"user_id": user_id, "sessions": store.get_user_sessions(user_id)}


@router.get("/sessions/{session_id}/messages")
def get_session_messages(
    session_id: str,
    store: Annotated[MemoryStore, Depends(get_memory_store)],
):
    if store.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    msgs = store.get_messages(session_id)
    return {
        "session_id": session_id,
        "messages": [m.model_dump(mode="json") for m in msgs],
    }


@router.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    store: Annotated[MemoryStore, Depends(get_memory_store)],
    orchestrator: Annotated[CogniFlowOrchestrator, Depends(get_orchestrator)],
):
    if store.get_session(body.session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")

    t0 = time.perf_counter()
    try:
        history, summary_from_db = _load_history_window(store, body.session_id)
        mem_rows = store.get_user_memories(body.user_id, limit=16)
        user_memory_context = _format_user_memory_context(mem_rows)

        agent = AgentState(
            session_id=body.session_id,
            user_id=body.user_id,
            user_query=body.message.strip(),
            conversation_history=history,
            user_memory_context=user_memory_context,
            conversation_summary=summary_from_db,
        )
        out = orchestrator.invoke(agent)
        latency = time.perf_counter() - t0

        user_msg = ChatMessage(role=MessageRole.user, content=body.message.strip())
        asst_msg = ChatMessage(role=MessageRole.assistant, content=out.response)
        store.add_message(body.session_id, user_msg)
        store.add_message(body.session_id, asst_msg)

        if out.conversation_summary and out.conversation_summary.strip():
            store.update_session_summary(body.session_id, out.conversation_summary)

        for item in out.memory_updates or []:
            if not isinstance(item, dict):
                continue
            content = (item.get("content") or "").strip()
            if not content:
                continue
            mt = (item.get("memory_type") or "context").strip()
            store.store_user_memory(
                body.user_id,
                mt,
                content,
                item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
            )
        store.prune_old_memories(body.user_id, keep_count=50)

        sources = _sources_from_retrieved(out.retrieved_documents or [])
        log = list(out.agent_log or [])
        log.append(
            {
                "node": "orchestrator",
                "total_latency_seconds": round(latency, 4),
            }
        )
        request_metrics.record_chat_latency(latency)
        return ChatResponse(
            session_id=body.session_id,
            response=out.response,
            sources=sources,
            agent_log=log,
            latency_seconds=round(latency, 4),
            conversation_summary=out.conversation_summary or "",
        )
    except HTTPException:
        raise
    except Exception as exc:
        request_metrics.record_error()
        logger.exception("chat failed: %s", exc)
        raise HTTPException(status_code=500, detail="Chat processing failed") from exc


@router.post("/chat/stream")
def chat_stream(
    body: ChatRequest,
    store: Annotated[MemoryStore, Depends(get_memory_store)],
    orchestrator: Annotated[CogniFlowOrchestrator, Depends(get_orchestrator)],
):
    """SSE stream of LangGraph state snapshots (`values` mode) and a final `done` event."""

    if store.get_session(body.session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")

    def iter_sse() -> Iterator[bytes]:
        import json

        t0 = time.perf_counter()
        history, summary_from_db = _load_history_window(store, body.session_id)
        mem_rows = store.get_user_memories(body.user_id, limit=16)
        user_memory_context = _format_user_memory_context(mem_rows)

        agent = AgentState(
            session_id=body.session_id,
            user_id=body.user_id,
            user_query=body.message.strip(),
            conversation_history=history,
            user_memory_context=user_memory_context,
            conversation_summary=summary_from_db,
        )
        try:
            payload = agent_state_to_graph(agent)
            payload.setdefault("memory_updates", [])
            payload.setdefault("agent_log", [])
            config: dict[str, Any] = {"configurable": {"thread_id": body.session_id}}
            last_full: dict[str, Any] | None = None
            for chunk in orchestrator.graph.stream(
                payload,
                config=config,
                stream_mode="values",
            ):
                last_full = chunk
                line = json.dumps({"event": "update", "data": chunk}, default=str)
                yield f"data: {line}\n\n".encode()

            if last_full is None:
                raise RuntimeError("empty graph output")

            out = graph_to_agent_state(agent, last_full)
            latency = time.perf_counter() - t0

            user_msg = ChatMessage(role=MessageRole.user, content=body.message.strip())
            asst_msg = ChatMessage(role=MessageRole.assistant, content=out.response)
            store.add_message(body.session_id, user_msg)
            store.add_message(body.session_id, asst_msg)
            if out.conversation_summary and out.conversation_summary.strip():
                store.update_session_summary(body.session_id, out.conversation_summary)
            for item in out.memory_updates or []:
                if not isinstance(item, dict):
                    continue
                content = (item.get("content") or "").strip()
                if not content:
                    continue
                mt = (item.get("memory_type") or "context").strip()
                store.store_user_memory(
                    body.user_id,
                    mt,
                    content,
                    item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                )
            store.prune_old_memories(body.user_id, keep_count=50)

            sources = _sources_from_retrieved(out.retrieved_documents or [])
            log = list(out.agent_log or [])
            log.append({"node": "orchestrator", "total_latency_seconds": round(latency, 4)})
            request_metrics.record_chat_latency(latency)
            payload_out = ChatResponse(
                session_id=body.session_id,
                response=out.response,
                sources=sources,
                agent_log=log,
                latency_seconds=round(latency, 4),
                conversation_summary=out.conversation_summary or "",
            )
            done = json.dumps(
                {"event": "done", "data": payload_out.model_dump(mode="json")},
                default=str,
            )
            yield f"data: {done}\n\n".encode()
        except Exception as exc:
            request_metrics.record_error()
            err = json.dumps({"event": "error", "detail": str(exc)})
            yield f"data: {err}\n\n".encode()

    return StreamingResponse(iter_sse(), media_type="text/event-stream")


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    store=Depends(get_vector_store),
):
    suffix = Path(file.filename or "upload").suffix.lower()
    allowed = {".pdf", ".md", ".markdown", ".html", ".htm"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix}. Use PDF, Markdown, or HTML.",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        Path(tmp_path).write_bytes(data)
        processor = DocumentProcessor()
        chunks = processor.process_file(tmp_path)
        if not chunks:
            raise HTTPException(status_code=422, detail="No extractable text from document")
        store.add_documents(chunks)
        request_metrics.record_upload()
        return DocumentUploadResponse(
            filename=file.filename or "upload",
            num_chunks=len(chunks),
            status="indexed",
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.get("/stats")
def stats(
    vstore: Annotated[object, Depends(get_vector_store)],
    store: Annotated[MemoryStore, Depends(get_memory_store)],
):
    vs = vstore.get_collection_stats()
    counts = store.table_counts()
    return {
        "vector_store": vs,
        "sessions_total": counts["sessions"],
        "messages_total": counts["messages"],
        "user_memory_rows": counts["user_memory_rows"],
        "embedding_model": settings.embedding_model,
        "llm_provider": settings.llm_provider,
    }


@router.get("/metrics")
def metrics():
    """Rolling latency and counters (complements `/stats` with operational metrics)."""
    return request_metrics.snapshot()


@router.get("/health")
def health():
    return {"status": "ok", "service": "cogniflow-api"}

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import hashlib
from uuid import uuid4
from pathlib import Path
from typing import Annotated, Any, Iterator

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from agents.graph_state import graph_to_agent_state
from agents.orchestrator import CogniFlowOrchestrator, merge_graph_patch
from api.deps import get_memory_store, get_orchestrator, get_vector_store, peek_vector_store
from api.metrics import request_metrics
from config import settings
from core.document_processor import DocumentProcessor
from core.memory_store import MemoryStore
from core.response_cache import get_cached, invalidate_session as invalidate_response_cache, put_cached
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

# SSE: stream assistant text in small chunks so UIs can render progressively (not only on `done`).
_SSE_TOKEN_CHUNK_CHARS = 14


def _sse_token_event_chunks(text: str) -> Iterator[bytes]:
    t = text or ""
    for i in range(0, len(t), _SSE_TOKEN_CHUNK_CHARS):
        piece = t[i : i + _SSE_TOKEN_CHUNK_CHARS]
        line = json.dumps({"event": "token", "data": piece}, ensure_ascii=False)
        yield f"data: {line}\n\n".encode()


def _chat_error_detail(exc: BaseException) -> str:
    """Surface safe, actionable hints; full traceback stays in logs."""
    if settings.expose_internal_errors:
        return (str(exc) or "Chat processing failed")[:2000]
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
    return "Chat processing failed"


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
                "original_filename": meta.get("original_filename"),
                "doc_instance_id": meta.get("doc_instance_id"),
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


def _format_peer_session_context(rows: list[dict[str, Any]]) -> str:
    """Compact lines for other chats' rolling summaries (cross-session awareness)."""
    if not rows:
        return ""
    lines: list[str] = []
    for r in rows:
        sid = str(r.get("session_id") or "")[:8]
        s = (r.get("summary") or "").strip().replace("\n", " ")
        if s:
            lines.append(f"- [{sid}…] {s[:650]}")
    return "\n".join(lines)


def _cross_session_context_block(
    store: MemoryStore, user_id: str, session_id: str
) -> str:
    rows = store.get_peer_session_summaries(user_id, session_id, limit=6)
    return _format_peer_session_context(rows)


def _append_pipeline_timing(agent_log: list[dict[str, Any]]) -> None:
    """Append a rollup row: sum of per-node ``elapsed_seconds`` (excludes orchestrator wall clock)."""
    timed_sum = 0.0
    n = 0
    for x in agent_log:
        if not isinstance(x, dict):
            continue
        if x.get("node") in ("pipeline", "orchestrator"):
            continue
        es = x.get("elapsed_seconds")
        if es is not None:
            timed_sum += float(es)
            n += 1
    agent_log.append(
        {
            "node": "pipeline",
            "timed_node_steps": n,
            "elapsed_seconds_sum_nodes": round(timed_sum, 4),
        }
    )


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


def _response_cache_context_fp(summary: str, history: list[ChatMessage]) -> str:
    """Stable fingerprint so cached replies respect conversation thread (optional)."""
    if not getattr(settings, "chat_response_cache_include_context", False):
        return ""
    n = max(0, int(getattr(settings, "chat_response_cache_context_messages", 16)))
    tail = history[-n:] if n else []
    lines = [f"{m.role.value}:{(m.content or '').strip()}" for m in tail]
    blob = ((summary or "").strip() + "\n" + "\n".join(lines)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@router.post("/sessions", response_model=SessionCreateResponse)
def create_session(
    body: SessionCreateRequest,
    store: Annotated[MemoryStore, Depends(get_memory_store)],
):
    """Create a new chat thread for the given user."""
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


@router.get("/sessions/{session_id}/agent-logs")
def get_session_agent_logs(
    session_id: str,
    store: Annotated[MemoryStore, Depends(get_memory_store)],
):
    """Return ``agent_log`` snapshots stored on assistant messages for this session."""
    if store.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = store.get_messages(session_id)
    logs: list[dict[str, Any]] = []
    for m in messages:
        if m.metadata and "agent_log" in m.metadata:
            logs.append(
                {
                    "message_id": m.id,
                    "timestamp": m.timestamp.isoformat(),
                    "agent_log": m.metadata["agent_log"],
                }
            )
    return {"session_id": session_id, "agent_logs": logs}


@router.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    store: Annotated[MemoryStore, Depends(get_memory_store)],
    orchestrator: Annotated[CogniFlowOrchestrator, Depends(get_orchestrator)],
):
    """Run the full LangGraph pipeline and persist messages, summary, and memories."""
    if store.get_session(body.session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")

    t0 = time.perf_counter()
    msg = body.message.strip()
    history, summary_from_db = _load_history_window(store, body.session_id)
    ctx_fp = _response_cache_context_fp(summary_from_db, history)
    try:
        if settings.chat_exact_message_cache_enabled:
            cached = get_cached(
                body.session_id, body.user_id, msg, context_fp=ctx_fp
            )
            if cached is not None:
                latency = time.perf_counter() - t0
                user_msg = ChatMessage(role=MessageRole.user, content=msg)
                log = [
                    {
                        "node": "response_cache",
                        "hit": True,
                        "elapsed_seconds": round(latency, 6),
                    },
                    {
                        "node": "orchestrator",
                        "total_latency_seconds": round(latency, 4),
                    },
                ]
                _append_pipeline_timing(log)
                asst_msg = ChatMessage(
                    role=MessageRole.assistant,
                    content=cached["response"],
                    metadata={"agent_log": log},
                )
                store.add_message(body.session_id, user_msg)
                store.add_message(body.session_id, asst_msg)
                request_metrics.record_chat_latency(latency)
                return ChatResponse(
                    session_id=body.session_id,
                    response=cached["response"],
                    sources=list(cached.get("sources") or []),
                    agent_log=log,
                    latency_seconds=round(latency, 4),
                    conversation_summary=cached.get("conversation_summary") or "",
                )

        mem_rows = store.get_user_memories(body.user_id, limit=16)
        user_memory_context = _format_user_memory_context(mem_rows)
        cross_ctx = _cross_session_context_block(
            store, body.user_id, body.session_id
        )

        agent = AgentState(
            session_id=body.session_id,
            user_id=body.user_id,
            user_query=msg,
            conversation_history=history,
            user_memory_context=user_memory_context,
            conversation_summary=summary_from_db,
            cross_session_context=cross_ctx,
        )
        out = orchestrator.invoke(agent)
        latency = time.perf_counter() - t0

        user_msg = ChatMessage(role=MessageRole.user, content=msg)
        log = list(out.agent_log or [])
        log.append(
            {
                "node": "orchestrator",
                "total_latency_seconds": round(latency, 4),
            }
        )
        _append_pipeline_timing(log)
        asst_msg = ChatMessage(
            role=MessageRole.assistant,
            content=out.response,
            metadata={"agent_log": log},
        )
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
        store.prune_old_memories(
            body.user_id,
            keep_count=50,
            strategy=settings.memory_pruning_strategy,
        )

        sources = _sources_from_retrieved(out.retrieved_documents or [])
        put_cached(
            body.session_id,
            body.user_id,
            msg,
            response=out.response,
            sources=sources,
            conversation_summary=out.conversation_summary or "",
            context_fp=ctx_fp,
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
        raise HTTPException(status_code=500, detail=_chat_error_detail(exc)) from exc


@router.post("/chat/stream")
def chat_stream(
    body: ChatRequest,
    store: Annotated[MemoryStore, Depends(get_memory_store)],
    orchestrator: Annotated[CogniFlowOrchestrator, Depends(get_orchestrator)],
):
    """SSE stream: real token events during synthesis, then a final ``done`` payload."""

    if store.get_session(body.session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")

    def iter_sse() -> Iterator[bytes]:
        t0 = time.perf_counter()
        msg = body.message.strip()
        history, summary_from_db = _load_history_window(store, body.session_id)
        ctx_fp = _response_cache_context_fp(summary_from_db, history)
        try:
            if settings.chat_exact_message_cache_enabled:
                cached = get_cached(
                    body.session_id, body.user_id, msg, context_fp=ctx_fp
                )
                if cached is not None:
                    latency = time.perf_counter() - t0
                    user_msg = ChatMessage(role=MessageRole.user, content=msg)
                    log = [
                        {
                            "node": "response_cache",
                            "hit": True,
                            "elapsed_seconds": round(latency, 6),
                        },
                        {
                            "node": "orchestrator",
                            "total_latency_seconds": round(latency, 4),
                        },
                    ]
                    _append_pipeline_timing(log)
                    asst_msg = ChatMessage(
                        role=MessageRole.assistant,
                        content=cached["response"],
                        metadata={"agent_log": log},
                    )
                    store.add_message(body.session_id, user_msg)
                    store.add_message(body.session_id, asst_msg)
                    request_metrics.record_chat_latency(latency)
                    payload_out = ChatResponse(
                        session_id=body.session_id,
                        response=cached["response"],
                        sources=list(cached.get("sources") or []),
                        agent_log=log,
                        latency_seconds=round(latency, 4),
                        postprocess_latency_seconds=None,
                        conversation_summary=cached.get("conversation_summary") or "",
                    )
                    reply_text = payload_out.response or ""
                    if reply_text:
                        yield from _sse_token_event_chunks(reply_text)
                    done = json.dumps(
                        {"event": "done", "data": payload_out.model_dump(mode="json")},
                        default=str,
                    )
                    yield f"data: {done}\n\n".encode()
                    return

            mem_rows = store.get_user_memories(body.user_id, limit=16)
            user_memory_context = _format_user_memory_context(mem_rows)
            cross_ctx = _cross_session_context_block(
                store, body.user_id, body.session_id
            )

            agent = AgentState(
                session_id=body.session_id,
                user_id=body.user_id,
                user_query=msg,
                conversation_history=history,
                user_memory_context=user_memory_context,
                conversation_summary=summary_from_db,
                cross_session_context=cross_ctx,
            )
            graph_state = orchestrator.prepare_graph_payload(agent)
            orchestrator.run_until_before_synthesis(graph_state)
            n_tokens_streamed = 0
            for evt in orchestrator.iter_streaming_synthesis(graph_state):
                et = evt.get("type")
                if et == "token":
                    n_tokens_streamed += 1
                    piece = evt.get("data") or ""
                    line = json.dumps(
                        {"event": "token", "data": piece}, ensure_ascii=False
                    )
                    yield f"data: {line}\n\n".encode()
                elif et == "complete":
                    merge_graph_patch(
                        graph_state,
                        {
                            "response": evt.get("response", ""),
                            "synthesized_context": evt.get("synthesized_context", ""),
                            "retrieved_documents": evt.get("retrieved_documents")
                            or [],
                            "agent_log": evt.get("agent_log") or [],
                        },
                    )
            t_after_synthesis = time.perf_counter()
            response_latency = t_after_synthesis - t0
            orchestrator.finalize_after_synthesis(graph_state)
            out = graph_to_agent_state(agent, graph_state)
            postprocess_latency = time.perf_counter() - t_after_synthesis
            total_wall = response_latency + postprocess_latency

            user_msg = ChatMessage(role=MessageRole.user, content=msg)
            log = list(out.agent_log or [])
            log.append(
                {
                    "node": "orchestrator",
                    "response_latency_seconds": round(response_latency, 4),
                    "postprocess_latency_seconds": round(postprocess_latency, 4),
                    "total_latency_seconds": round(total_wall, 4),
                }
            )
            _append_pipeline_timing(log)
            asst_msg = ChatMessage(
                role=MessageRole.assistant,
                content=out.response,
                metadata={"agent_log": log},
            )
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
            store.prune_old_memories(
                body.user_id,
                keep_count=50,
                strategy=settings.memory_pruning_strategy,
            )

            sources = _sources_from_retrieved(out.retrieved_documents or [])
            put_cached(
                body.session_id,
                body.user_id,
                msg,
                response=out.response,
                sources=sources,
                conversation_summary=out.conversation_summary or "",
                context_fp=ctx_fp,
            )
            request_metrics.record_chat_latency(response_latency)
            payload_out = ChatResponse(
                session_id=body.session_id,
                response=out.response,
                sources=sources,
                agent_log=log,
                latency_seconds=round(response_latency, 4),
                postprocess_latency_seconds=round(postprocess_latency, 4),
                conversation_summary=out.conversation_summary or "",
            )
            reply_text = payload_out.response or ""
            if reply_text and n_tokens_streamed == 0:
                yield from _sse_token_event_chunks(reply_text)
            done = json.dumps(
                {"event": "done", "data": payload_out.model_dump(mode="json")},
                default=str,
            )
            yield f"data: {done}\n\n".encode()
        except Exception as exc:
            request_metrics.record_error()
            err = json.dumps({"event": "error", "detail": str(exc)})
            yield f"data: {err}\n\n".encode()

    return StreamingResponse(
        iter_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/documents/upload", response_model=DocumentUploadResponse)
def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(..., description="Chat session to attach this index to"),
    store=Depends(get_vector_store),
    mem: MemoryStore = Depends(get_memory_store),
):
    """Chunk and index an uploaded PDF/Markdown/HTML into the scoped Chroma collection."""
    sid = (session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=422, detail="session_id is required")
    sess = mem.get_session(sid)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    owner_uid = (sess.user_id or "").strip()

    suffix = Path(file.filename or "upload").suffix.lower()
    allowed = {".pdf", ".md", ".markdown", ".html", ".htm"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix}. Use PDF, Markdown, or HTML.",
        )
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    content_hash = hashlib.sha256(data).hexdigest()
    if store.has_document(sid, content_hash) or (
        owner_uid and store.has_user_document(owner_uid, content_hash)
    ):
        return DocumentUploadResponse(
            filename=file.filename or "upload",
            num_chunks=0,
            status="already_indexed",
        )

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    doc_instance_id = str(uuid4())
    orig_name = (file.filename or "upload").strip() or "upload"
    try:
        Path(tmp_path).write_bytes(data)
        processor = DocumentProcessor()
        chunks = processor.process_file(
            tmp_path,
            original_filename=orig_name,
            doc_instance_id=doc_instance_id,
            session_id=sid,
            user_id=owner_uid,
            content_hash=content_hash,
        )
        if not chunks:
            raise HTTPException(status_code=422, detail="No extractable text from document")
        store.add_documents(chunks)
        invalidate_response_cache(sid)
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
    request: Request,
    store: Annotated[MemoryStore, Depends(get_memory_store)],
    session_id: str | None = Query(
        default=None,
        description="If set with user_id, count chunks visible in RAG (this session ∪ this user).",
    ),
    user_id: str | None = Query(
        default=None,
        description="Same user as the chat; pair with session_id for scoped chunk count.",
    ),
):
    """Return vector-store and SQLite table counts plus configured model names."""
    ov = request.app.dependency_overrides.get(get_vector_store)
    if ov is not None:
        vstore = ov()
        vs_err = None
    else:
        vstore, vs_err = peek_vector_store()
    if vstore is not None:
        vs = vstore.get_collection_stats(session_id=session_id, user_id=user_id)
        vs.setdefault("status", "ok")
    else:
        vs = {
            "name": settings.chroma_collection_name,
            "count": 0,
            "status": "unavailable",
            "detail": vs_err or "Vector store unavailable",
        }
    counts = store.table_counts()
    return {
        "vector_store": vs,
        "sessions_total": counts["sessions"],
        "messages_total": counts["messages"],
        "user_memory_rows": counts["user_memory_rows"],
        "embedding_backend": settings.embedding_backend,
        "embedding_device": settings.embedding_device,
        "embedding_model": settings.embedding_model,
        "openai_embedding_model": settings.openai_embedding_model,
        "llm_provider": settings.llm_provider,
    }


@router.get("/metrics")
def metrics():
    """Expose rolling chat/upload latency stats and error counters for operators."""
    return request_metrics.snapshot()


@router.get("/health")
def health():
    """Liveness probe for load balancers and Docker healthchecks."""
    return {"status": "ok", "service": "cogniflow-api"}

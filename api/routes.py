from __future__ import annotations

import hmac
import json
import logging
import os
import tempfile
import time
import hashlib
from uuid import uuid4
from pathlib import Path
from typing import Annotated, Any, Iterator

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import StreamingResponse

from agents.graph_state import graph_to_agent_state
from agents.orchestrator import CogniFlowOrchestrator, merge_graph_patch
from api.deps import (
    ensure_api_user_matches,
    ensure_api_user_owns_session,
    get_memory_store,
    get_orchestrator,
    get_vector_store,
    peek_vector_store,
    verify_api_key,
)
from api.errors import safe_client_error_detail, sse_error_payload
from api.limiter import limiter
from api.route_helpers import (
    append_pipeline_timing,
    cross_session_context_block,
    format_user_memory_context,
    response_cache_context_fp,
    sources_from_retrieved,
    sse_token_event_chunks,
)
from api.metrics import request_metrics
from config import settings
from core.document_processor import DocumentProcessor
from core.memory_store import MemoryStore
from core.response_cache import get_cached, invalidate_session as invalidate_response_cache, put_cached
from core.models import (
    AgentState,
    ApiKeyCreatedResponse,
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
    auth_uid: Annotated[str | None, Depends(verify_api_key)],
):
    """Create a new chat thread for the given user."""
    ensure_api_user_matches(auth_uid, body.user_id)
    session = store.create_session(body.user_id)
    return SessionCreateResponse(
        session_id=session.session_id,
        user_id=session.user_id,
        created_at=session.created_at,
    )


@router.post("/users/{user_id}/api-keys", response_model=ApiKeyCreatedResponse)
def create_user_api_key(
    user_id: str,
    store: Annotated[MemoryStore, Depends(get_memory_store)],
    x_admin_secret: Annotated[str | None, Header(alias="X-Admin-Secret")] = None,
):
    """Mint a new API key for ``user_id`` (requires ``API_ADMIN_SECRET`` and matching ``X-Admin-Secret``)."""
    import config as _cfg

    admin = (_cfg.settings.api_admin_secret or "").strip()
    if not admin:
        raise HTTPException(
            status_code=403, detail="API key minting is not configured (set API_ADMIN_SECRET)"
        )
    if not hmac.compare_digest((x_admin_secret or "").strip(), admin):
        raise HTTPException(status_code=401, detail="Invalid admin secret")
    uid = (user_id or "").strip()
    if not uid:
        raise HTTPException(status_code=422, detail="user_id is required")
    raw = store.create_api_key(uid)
    return ApiKeyCreatedResponse(user_id=uid, api_key=raw)


@router.get("/users/{user_id}/sessions")
def list_user_sessions(
    user_id: str,
    store: Annotated[MemoryStore, Depends(get_memory_store)],
    auth_uid: Annotated[str | None, Depends(verify_api_key)],
):
    """List sessions for a user (newest first)."""
    ensure_api_user_matches(auth_uid, user_id)
    return {"user_id": user_id, "sessions": store.get_user_sessions(user_id)}


@router.get("/sessions/{session_id}/messages")
def get_session_messages(
    session_id: str,
    store: Annotated[MemoryStore, Depends(get_memory_store)],
    auth_uid: Annotated[str | None, Depends(verify_api_key)],
):
    ensure_api_user_owns_session(auth_uid, store, session_id)
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
    auth_uid: Annotated[str | None, Depends(verify_api_key)],
):
    """Return ``agent_log`` snapshots stored on assistant messages for this session."""
    ensure_api_user_owns_session(auth_uid, store, session_id)
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
@limiter.limit("30/minute")
def chat(
    request: Request,
    auth_uid: Annotated[str | None, Depends(verify_api_key)],
    body: ChatRequest,
    store: Annotated[MemoryStore, Depends(get_memory_store)],
    orchestrator: Annotated[CogniFlowOrchestrator, Depends(get_orchestrator)],
):
    """Run the full LangGraph pipeline and persist messages, summary, and memories."""
    ensure_api_user_matches(auth_uid, body.user_id)
    ensure_api_user_owns_session(auth_uid, store, body.session_id)
    if store.get_session(body.session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")

    t0 = time.perf_counter()
    msg = body.message.strip()
    history, summary_from_db = _load_history_window(store, body.session_id)
    ctx_fp = response_cache_context_fp(summary_from_db, history)
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
                append_pipeline_timing(log)
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
        user_memory_context = format_user_memory_context(mem_rows)
        cross_ctx = cross_session_context_block(
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
        append_pipeline_timing(log)
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

        sources = sources_from_retrieved(out.retrieved_documents or [])
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
        rid = getattr(request.state, "request_id", None)
        logger.exception("chat failed request_id=%s", rid)
        raise HTTPException(status_code=500, detail=safe_client_error_detail(exc)) from exc


@router.post("/chat/stream")
@limiter.limit("30/minute")
def chat_stream(
    request: Request,
    auth_uid: Annotated[str | None, Depends(verify_api_key)],
    body: ChatRequest,
    store: Annotated[MemoryStore, Depends(get_memory_store)],
    orchestrator: Annotated[CogniFlowOrchestrator, Depends(get_orchestrator)],
):
    """SSE stream: real token events during synthesis, then a final ``done`` payload."""
    ensure_api_user_matches(auth_uid, body.user_id)
    ensure_api_user_owns_session(auth_uid, store, body.session_id)

    if store.get_session(body.session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")

    def iter_sse() -> Iterator[bytes]:
        t0 = time.perf_counter()
        rid = getattr(request.state, "request_id", None)
        msg = body.message.strip()
        history, summary_from_db = _load_history_window(store, body.session_id)
        ctx_fp = response_cache_context_fp(summary_from_db, history)
        try:
            # Immediate SSE comment so proxies/clients see a byte on the wire (avoids "nothing until LLM" confusion).
            yield b": stream-open\n\n"
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
                    append_pipeline_timing(log)
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
                        yield from sse_token_event_chunks(reply_text)
                    done = json.dumps(
                        {"event": "done", "data": payload_out.model_dump(mode="json")},
                        default=str,
                    )
                    yield f"data: {done}\n\n".encode()
                    return

            mem_rows = store.get_user_memories(body.user_id, limit=16)
            user_memory_context = format_user_memory_context(mem_rows)
            cross_ctx = cross_session_context_block(
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
            yield (
                f"data: {json.dumps({'event': 'phase', 'data': {'step': 'routing'}}, ensure_ascii=False)}\n\n"
            ).encode()
            orchestrator.run_until_before_synthesis(graph_state)
            yield (
                f"data: {json.dumps({'event': 'phase', 'data': {'step': 'llm'}}, ensure_ascii=False)}\n\n"
            ).encode()
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
            append_pipeline_timing(log)
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

            sources = sources_from_retrieved(out.retrieved_documents or [])
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
                yield from sse_token_event_chunks(reply_text)
            done = json.dumps(
                {"event": "done", "data": payload_out.model_dump(mode="json")},
                default=str,
            )
            yield f"data: {done}\n\n".encode()
        except HTTPException as exc:
            request_metrics.record_error()
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            logger.warning(
                "chat_stream HTTP error status=%s detail=%s request_id=%s",
                exc.status_code,
                detail,
                rid,
            )
            payload: dict[str, Any] = {
                "event": "error",
                "detail": detail,
                "status_code": exc.status_code,
            }
            if rid:
                payload["request_id"] = rid
            err = json.dumps(payload, default=str, ensure_ascii=False)
            yield f"data: {err}\n\n".encode()
        except Exception as exc:
            request_metrics.record_error()
            logger.exception("chat_stream failed request_id=%s", rid)
            err = json.dumps(
                sse_error_payload(exc, request_id=rid), default=str, ensure_ascii=False
            )
            yield f"data: {err}\n\n".encode()

    hdrs = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        # Hint caches/CDNs not to treat this as a cacheable document (some strip SSE without this).
        "Surrogate-Control": "no-store",
    }
    if rid0 := getattr(request.state, "request_id", None):
        hdrs["X-Request-ID"] = rid0
    return StreamingResponse(
        iter_sse(),
        media_type="text/event-stream",
        headers=hdrs,
    )


@router.post("/documents/upload", response_model=DocumentUploadResponse)
@limiter.limit("10/minute")
def upload_document(
    request: Request,
    auth_uid: Annotated[str | None, Depends(verify_api_key)],
    file: UploadFile = File(...),
    session_id: str = Form(..., description="Chat session to attach this index to"),
    store=Depends(get_vector_store),
    mem: MemoryStore = Depends(get_memory_store),
):
    """Chunk and index an uploaded PDF/Markdown/HTML into the scoped Chroma collection."""
    sid = (session_id or "").strip()
    ensure_api_user_owns_session(auth_uid, mem, sid)
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
    _MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB hard limit
    data = file.file.read(_MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum allowed size is 50 MB.",
        )
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
    except HTTPException:
        raise
    except Exception as exc:
        request_metrics.record_error()
        logger.exception("documents/upload failed session_id=%s", sid)
        raise HTTPException(
            status_code=500, detail=safe_client_error_detail(exc)
        ) from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.get("/stats")
def stats(
    request: Request,
    auth_uid: Annotated[str | None, Depends(verify_api_key)],
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
    if auth_uid and user_id and (user_id or "").strip() != auth_uid.strip():
        raise HTTPException(
            status_code=403, detail="user_id query does not match API key owner"
        )
    if auth_uid and session_id:
        ensure_api_user_owns_session(auth_uid, store, session_id)
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
        "api_keys_active": counts.get("api_keys_active", 0),
        "embedding_backend": settings.embedding_backend,
        "embedding_device": settings.embedding_device,
        "embedding_model": settings.embedding_model,
        "openai_embedding_model": settings.openai_embedding_model,
        "llm_provider": settings.llm_provider,
    }


@router.get("/metrics")
def metrics(
    auth_uid: Annotated[str | None, Depends(verify_api_key)],
):
    """Expose rolling chat/upload latency stats and error counters for operators."""
    _ = auth_uid
    return request_metrics.snapshot()


@router.get("/health")
def health():
    """Liveness probe for load balancers and Docker healthchecks."""
    return {"status": "ok", "service": "cogniflow-api"}

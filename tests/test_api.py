from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from agents.graph_state import agent_state_to_graph
from core.models import AgentState, QueryIntent, RetrievalStrategy


class _FakeOrchestrator:
    def __init__(self):
        self.graph = MagicMock()

    def invoke(self, state: AgentState, **kwargs):
        return AgentState(
            session_id=state.session_id,
            user_id=state.user_id,
            user_query=state.user_query,
            conversation_history=state.conversation_history,
            user_memory_context=state.user_memory_context,
            query_intent=QueryIntent.factual,
            needs_history=False,
            needs_rewrite=False,
            rewritten_query=state.user_query,
            retrieval_strategy=RetrievalStrategy.semantic,
            retrieved_documents=[
                {
                    "id": "c1",
                    "content": "doc",
                    "metadata": {"title": "T", "source": "s"},
                    "distance": 0.2,
                }
            ],
            response="Hello from stub orchestrator.",
            agent_log=[{"node": "query_understanding", "intent": "factual"}],
        )


class _FakeStreamingOrchestrator:
    """Minimal orchestrator for ``/chat/stream`` without real LangGraph or LLM."""

    def prepare_graph_payload(self, state: AgentState) -> dict:
        payload = agent_state_to_graph(state)
        payload.setdefault("memory_updates", [])
        payload.setdefault("agent_log", [])
        payload.setdefault("sub_queries", [])
        payload.setdefault("use_retrieved_context", True)
        payload.setdefault("context_validation_reason", "")
        return payload

    def run_until_before_synthesis(self, graph_state: dict) -> None:
        graph_state["query_intent"] = "general_knowledge"
        graph_state["needs_retrieval"] = False
        graph_state.setdefault("agent_log", []).append(
            {"node": "query_understanding", "intent": "general_knowledge"}
        )

    def iter_streaming_synthesis(self, graph_state: dict):
        yield {"type": "token", "data": "Hel"}
        yield {"type": "token", "data": "lo"}
        yield {
            "type": "complete",
            "response": "Hello stream",
            "synthesized_context": "",
            "retrieved_documents": [],
            "agent_log": [{"node": "context_synthesis"}],
        }

    def finalize_after_synthesis(self, graph_state: dict) -> None:
        graph_state.setdefault("agent_log", []).append(
            {"node": "conversation_summarizer", "skipped": True}
        )


@pytest.fixture()
def client(tmp_path, monkeypatch, clear_singletons):
    pytest.importorskip("multipart", reason="FastAPI File/Form routes require python-multipart")
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "memory.db"))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHAT_RESPONSE_CACHE_BACKEND", "memory")
    # Avoid EmbeddingManager in /chat (put_cached embeds by default); keeps tests fast/offline.
    monkeypatch.setenv("CHAT_EXACT_MESSAGE_CACHE_ENABLED", "false")

    import importlib
    from importlib import reload

    import config as cfg

    reload(cfg)
    import main

    importlib.reload(main)

    from api.deps import get_memory_store, get_orchestrator, get_vector_store

    app = main.app

    store = __import__("core.memory_store", fromlist=["MemoryStore"]).MemoryStore(
        db_path=str(tmp_path / "memory.db")
    )
    fake = _FakeOrchestrator()

    def _mem():
        return store

    def _orch():
        return fake

    fake_vs = MagicMock()
    fake_vs.get_collection_stats = MagicMock(
        return_value={"name": "mock", "count": 0},
    )
    fake_vs.has_document = MagicMock(return_value=False)

    def _vs():
        return fake_vs

    app.dependency_overrides[get_memory_store] = _mem
    app.dependency_overrides[get_orchestrator] = _orch
    app.dependency_overrides[get_vector_store] = _vs

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


def test_health(client: TestClient):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_session_and_chat(client: TestClient):
    r = client.post("/api/v1/sessions", json={"user_id": "alice"})
    assert r.status_code == 200
    sid = r.json()["session_id"]

    r2 = client.post(
        "/api/v1/chat",
        json={"session_id": sid, "user_id": "alice", "message": "What is FastAPI?"},
    )
    assert r2.status_code == 200
    body = r2.json()
    assert "Hello from stub" in body["response"]
    assert body.get("latency_seconds") is not None

    r3 = client.get(f"/api/v1/sessions/{sid}/messages")
    assert r3.status_code == 200
    msgs = r3.json()["messages"]
    assert len(msgs) == 2


def test_stats(client: TestClient):
    r = client.get("/api/v1/stats")
    assert r.status_code == 200
    assert "vector_store" in r.json()


def test_create_and_list_sessions(client: TestClient):
    r1 = client.post("/api/v1/sessions", json={"user_id": "bob"})
    r2 = client.post("/api/v1/sessions", json={"user_id": "bob"})
    assert r1.status_code == 200 and r2.status_code == 200
    lst = client.get("/api/v1/users/bob/sessions")
    assert lst.status_code == 200
    data = lst.json()
    assert data["user_id"] == "bob"
    assert len(data["sessions"]) == 2


def test_chat_nonexistent_session(client: TestClient):
    r = client.post(
        "/api/v1/chat",
        json={
            "session_id": "00000000-0000-0000-0000-000000000000",
            "user_id": "x",
            "message": "hi",
        },
    )
    assert r.status_code == 404


def test_get_messages_nonexistent(client: TestClient):
    r = client.get("/api/v1/sessions/bad-id-123/messages")
    assert r.status_code == 404


def test_upload_requires_session(client: TestClient):
    r = client.post(
        "/api/v1/documents/upload",
        files={"file": ("x.md", b"# hi", "text/markdown")},
    )
    assert r.status_code == 422


def test_upload_unknown_session(client: TestClient):
    r = client.post(
        "/api/v1/documents/upload",
        files={"file": ("x.md", b"# hi", "text/markdown")},
        data={"session_id": "00000000-0000-0000-0000-000000000000"},
    )
    assert r.status_code == 404


def test_upload_unsupported_file(client: TestClient):
    r_sess = client.post("/api/v1/sessions", json={"user_id": "u"})
    sid = r_sess.json()["session_id"]
    r = client.post(
        "/api/v1/documents/upload",
        files={"file": ("t.xyz", b"data", "application/octet-stream")},
        data={"session_id": sid},
    )
    assert r.status_code == 400


def test_upload_empty_file(client: TestClient):
    r_sess = client.post("/api/v1/sessions", json={"user_id": "u"})
    sid = r_sess.json()["session_id"]
    r = client.post(
        "/api/v1/documents/upload",
        files={"file": ("empty.md", b"", "text/markdown")},
        data={"session_id": sid},
    )
    assert r.status_code == 400


def test_upload_duplicate_is_not_reindexed(client: TestClient):
    from api.deps import get_vector_store

    r_sess = client.post("/api/v1/sessions", json={"user_id": "u"})
    sid = r_sess.json()["session_id"]
    fake_vs = MagicMock()
    fake_vs.has_document = MagicMock(return_value=True)
    fake_vs.has_user_document = MagicMock(return_value=False)
    client.app.dependency_overrides[get_vector_store] = lambda: fake_vs
    try:
        r = client.post(
            "/api/v1/documents/upload",
            files={"file": ("dup.md", b"# Same", "text/markdown")},
            data={"session_id": sid},
        )
    finally:
        client.app.dependency_overrides.pop(get_vector_store, None)

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "already_indexed"
    assert body["num_chunks"] == 0


def test_metrics_endpoint(client: TestClient):
    r = client.get("/api/v1/metrics")
    assert r.status_code == 200
    body = r.json()
    for key in (
        "chat_requests_total",
        "upload_requests_total",
        "errors_total",
        "chat_latency_samples",
        "chat_latency_avg_seconds",
        "chat_latency_p95_seconds",
        "last_updated_unix",
    ):
        assert key in body


def test_agent_logs_endpoint(client: TestClient):
    r = client.post("/api/v1/sessions", json={"user_id": "logs-user"})
    assert r.status_code == 200
    sid = r.json()["session_id"]

    r2 = client.post(
        "/api/v1/chat",
        json={
            "session_id": sid,
            "user_id": "logs-user",
            "message": "Hello?",
        },
    )
    assert r2.status_code == 200

    r3 = client.get(f"/api/v1/sessions/{sid}/agent-logs")
    assert r3.status_code == 200
    payload = r3.json()
    assert payload["session_id"] == sid
    assert isinstance(payload["agent_logs"], list)
    assert len(payload["agent_logs"]) >= 1


def test_agent_logs_not_found(client: TestClient):
    r = client.get("/api/v1/sessions/not-a-real-session/agent-logs")
    assert r.status_code == 404


@pytest.fixture()
def stream_client(tmp_path, monkeypatch, clear_singletons):
    """App + deps for streaming tests (fake streaming orchestrator)."""
    pytest.importorskip("multipart", reason="FastAPI File/Form routes require python-multipart")
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "memory.db"))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHAT_RESPONSE_CACHE_BACKEND", "memory")
    monkeypatch.setenv("CHAT_EXACT_MESSAGE_CACHE_ENABLED", "false")

    import importlib

    import config as cfg

    importlib.reload(cfg)
    import main

    importlib.reload(main)

    from api.deps import get_memory_store, get_orchestrator, get_vector_store

    app = main.app
    store = __import__("core.memory_store", fromlist=["MemoryStore"]).MemoryStore(
        db_path=str(tmp_path / "memory.db")
    )
    fake_vs = MagicMock()
    fake_vs.get_collection_stats = MagicMock(return_value={"name": "mock", "count": 0})
    fake_vs.has_document = MagicMock(return_value=False)

    app.dependency_overrides[get_memory_store] = lambda: store
    app.dependency_overrides[get_orchestrator] = lambda: _FakeStreamingOrchestrator()
    app.dependency_overrides[get_vector_store] = lambda: fake_vs

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


def test_chat_stream_yields_tokens_and_done(stream_client: TestClient):
    r_sess = stream_client.post("/api/v1/sessions", json={"user_id": "streamer"})
    sid = r_sess.json()["session_id"]
    with stream_client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"session_id": sid, "user_id": "streamer", "message": "Hi"},
    ) as r:
        assert r.status_code == 200
        raw = b"".join(r.iter_bytes())
    text = raw.decode("utf-8")
    assert "event" in text and "token" in text
    assert "done" in text
    assert "Hello stream" in text


def test_chat_stream_unknown_session_returns_404(stream_client: TestClient):
    r = stream_client.post(
        "/api/v1/chat/stream",
        json={
            "session_id": "00000000-0000-0000-0000-000000000000",
            "user_id": "u",
            "message": "x",
        },
    )
    assert r.status_code == 404


def test_chat_invoke_error_returns_500(client: TestClient, monkeypatch):
    bad = MagicMock()
    bad.invoke.side_effect = RuntimeError("simulated provider failure")

    def _orch():
        return bad

    from api.deps import get_orchestrator

    client.app.dependency_overrides[get_orchestrator] = _orch
    try:
        r_sess = client.post("/api/v1/sessions", json={"user_id": "err-user"})
        sid = r_sess.json()["session_id"]
        r = client.post(
            "/api/v1/chat",
            json={"session_id": sid, "user_id": "err-user", "message": "boom"},
        )
        assert r.status_code == 500
        body = r.json()
        assert "detail" in body
    finally:
        client.app.dependency_overrides.pop(get_orchestrator, None)

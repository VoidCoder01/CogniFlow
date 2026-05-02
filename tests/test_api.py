from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

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


@pytest.fixture()
def client(tmp_path, monkeypatch, clear_singletons):
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "memory.db"))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))

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

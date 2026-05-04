"""API key auth (optional via API_AUTH_ENABLED)."""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def auth_client(tmp_path, monkeypatch, clear_singletons):
    pytest.importorskip("multipart", reason="FastAPI needs python-multipart")
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "memory.db"))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHAT_RESPONSE_CACHE_BACKEND", "memory")
    monkeypatch.setenv("CHAT_EXACT_MESSAGE_CACHE_ENABLED", "false")
    monkeypatch.setenv("API_AUTH_ENABLED", "true")
    monkeypatch.setenv("API_ADMIN_SECRET", "admin-test-secret")

    import config as cfg

    importlib.reload(cfg)
    import main

    importlib.reload(main)
    from api.deps import get_memory_store, get_orchestrator, get_vector_store
    from unittest.mock import MagicMock

    from core.memory_store import MemoryStore
    from tests.test_api import _FakeOrchestrator

    app = main.app
    store = MemoryStore(db_path=str(tmp_path / "memory.db"))

    app.dependency_overrides[get_memory_store] = lambda: store
    app.dependency_overrides[get_orchestrator] = lambda: _FakeOrchestrator()
    fake_vs = MagicMock()
    fake_vs.get_collection_stats = MagicMock(return_value={"name": "m", "count": 0})
    fake_vs.has_document = MagicMock(return_value=False)
    app.dependency_overrides[get_vector_store] = lambda: fake_vs

    with TestClient(app) as c:
        yield c, store

    app.dependency_overrides.clear()


def test_health_skips_auth(auth_client):
    c, _store = auth_client
    r = c.get("/api/v1/health")
    assert r.status_code == 200


def test_chat_requires_key(auth_client):
    c, store = auth_client
    s = store.create_session("user_a")
    r = c.post(
        "/api/v1/chat",
        json={"session_id": s.session_id, "user_id": "user_a", "message": "hi"},
    )
    assert r.status_code == 401


def test_mint_and_chat_with_key(auth_client):
    c, store = auth_client
    r_mint = c.post(
        "/api/v1/users/user_a/api-keys",
        headers={"X-Admin-Secret": "admin-test-secret"},
    )
    assert r_mint.status_code == 200
    key = r_mint.json()["api_key"]
    s = store.create_session("user_a")
    r = c.post(
        "/api/v1/chat",
        json={"session_id": s.session_id, "user_id": "user_a", "message": "hi"},
        headers={"X-API-Key": key},
    )
    assert r.status_code == 200


def test_wrong_user_id_with_key(auth_client):
    c, store = auth_client
    r_mint = c.post(
        "/api/v1/users/user_a/api-keys",
        headers={"X-Admin-Secret": "admin-test-secret"},
    )
    key = r_mint.json()["api_key"]
    s = store.create_session("user_a")
    r = c.post(
        "/api/v1/chat",
        json={"session_id": s.session_id, "user_id": "other", "message": "hi"},
        headers={"X-API-Key": key},
    )
    assert r.status_code == 403

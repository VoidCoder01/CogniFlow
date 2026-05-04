"""api.errors: client-safe messages."""

from __future__ import annotations

import pytest


def test_safe_detail_generic_when_not_exposed(monkeypatch):
    monkeypatch.setenv("EXPOSE_INTERNAL_ERRORS", "false")
    import importlib

    import api.errors as ae
    import config as cfg

    importlib.reload(cfg)
    importlib.reload(ae)
    from api.errors import safe_client_error_detail

    assert "password" not in safe_client_error_detail(
        RuntimeError("internal db password=secret")
    )
    assert "secret" not in safe_client_error_detail(
        RuntimeError("internal db password=secret")
    )


def test_safe_detail_exposes_when_flag_true(monkeypatch):
    monkeypatch.setenv("EXPOSE_INTERNAL_ERRORS", "true")
    import importlib

    import api.errors as ae
    import config as cfg

    importlib.reload(cfg)
    importlib.reload(ae)
    from api.errors import safe_client_error_detail

    assert "visible" in safe_client_error_detail(ValueError("visible traceback bit"))


def test_safe_detail_rate_limit_snippet(monkeypatch):
    monkeypatch.setenv("EXPOSE_INTERNAL_ERRORS", "false")
    import importlib

    import api.errors as ae
    import config as cfg

    importlib.reload(cfg)
    importlib.reload(ae)
    from api.errors import safe_client_error_detail

    msg = safe_client_error_detail(RuntimeError("Error 429 rate limit exceeded"))
    assert "429" in msg


@pytest.fixture()
def fresh_main(monkeypatch, tmp_path):
    pytest.importorskip("multipart", reason="FastAPI Form/File routes need python-multipart")
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "m.db"))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    import importlib
    import sys

    import config as cfg

    importlib.reload(cfg)
    for name in ("api.errors", "api.middleware", "api.routes"):
        mod = sys.modules.get(name)
        if mod is not None:
            importlib.reload(mod)
    if "main" in sys.modules:
        importlib.reload(sys.modules["main"])
    else:
        import main  # noqa: F401

    return sys.modules["main"]


def test_health_returns_request_id_header(fresh_main):
    c = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(fresh_main.app)
    r = c.get("/api/v1/health")
    assert r.status_code == 200
    assert r.headers.get("X-Request-ID")


def test_health_echoes_incoming_request_id(fresh_main):
    TC = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient
    with TC(fresh_main.app) as c:
        r = c.get("/api/v1/health", headers={"X-Request-ID": "client-fixed-id"})
        assert r.headers.get("X-Request-ID") == "client-fixed-id"

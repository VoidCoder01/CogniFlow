from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest

# Ensure tests use isolated stores before application modules read settings.
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-import-only")


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def clear_singletons() -> Generator[None, None, None]:
    from api.deps import clear_app_caches

    clear_app_caches()
    yield
    clear_app_caches()

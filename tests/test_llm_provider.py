"""Tests for chat model factory error paths."""

from __future__ import annotations

import pytest

from config import settings
from core.llm_provider import clear_chat_model_cache, get_chat_model


@pytest.fixture(autouse=True)
def _reset_llm_cache():
    clear_chat_model_cache()
    yield
    clear_chat_model_cache()


def test_unsupported_llm_provider_raises(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "__not_a_provider__")
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        get_chat_model()


@pytest.mark.parametrize(
    ("provider", "attr", "needle"),
    [
        ("openai", "openai_api_key", "OPENAI_API_KEY"),
        ("groq", "groq_api_key", "GROQ_API_KEY"),
        ("anthropic", "anthropic_api_key", "ANTHROPIC_API_KEY"),
        ("openrouter", "openrouter_api_key", "OPENROUTER_API_KEY"),
        ("gemini", "google_api_key", "GOOGLE_API_KEY"),
    ],
)
def test_missing_provider_api_key_raises(monkeypatch, provider, attr, needle):
    monkeypatch.setattr(settings, "llm_provider", provider)
    monkeypatch.setattr(settings, attr, "")
    with pytest.raises(ValueError, match=needle):
        get_chat_model()

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from config import settings

logger = logging.getLogger(__name__)


def _temperature_kwargs() -> dict:
    return {
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
    }


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """
    Return a LangChain chat model for the configured provider.
    Cached so the graph and CLI reuse one client per process.
    """
    provider = settings.llm_provider.lower().strip()
    kw = _temperature_kwargs()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        model = settings.openai_model or "gpt-4o-mini"
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=model,
            **kw,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        model = settings.anthropic_model or "claude-sonnet-4-20250514"
        return ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model=model,
            **kw,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        model = settings.llm_model or "llama-3.1-8b-instant"
        return ChatGroq(api_key=settings.groq_api_key, model=model, **kw)

    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        model = settings.llm_model or "llama3.1"
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=model,
            temperature=settings.llm_temperature,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER={settings.llm_provider!r}; "
        "use openai, anthropic, groq, or ollama"
    )


def clear_chat_model_cache() -> None:
    """Tests / hot-reload: drop cached client."""
    get_chat_model.cache_clear()

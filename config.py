from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment and optional `.env` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM — openai | anthropic | groq | ollama | openrouter | gemini
    llm_provider: str = Field(
        default="gemini",
        description="openai | anthropic | groq | ollama | openrouter | gemini (alias: google)",
    )
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    # OpenRouter (OpenAI-compatible API; get key at https://openrouter.ai/keys)
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "google/gemma-2-9b-it"
    openrouter_http_referer: str = "https://github.com/cogniflow"
    openrouter_app_title: str = "CogniFlow"
    # Google Gemini (Google AI Studio / Generative Language API)
    google_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    # Provider-specific defaults (override with OPENAI_MODEL / ANTHROPIC_MODEL / LLM_MODEL)
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-sonnet-4-20250514"
    llm_model: str = ""  # fallback for groq / ollama when provider-specific model unset
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048

    # LangGraph checkpointing: memory (default) or sqlite (persistent thread state)
    checkpoint_backend: str = Field(
        default="memory",
        description="memory | sqlite",
    )
    checkpoint_sqlite_path: str = "./data/checkpoints.db"

    # Embeddings & vector DB (local = SentenceTransformer; openai = API embeddings)
    embedding_backend: str = Field(
        default="local",
        description="local | openai",
    )
    embedding_device: str = Field(
        default="cpu",
        description="cpu | cuda (local embeddings only)",
    )
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "cogniflow_docs"
    sqlite_db_path: str = "./data/memory.db"

    # Chunking & conversation
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_conversation_length: int = 50
    summary_threshold: int = 10
    summary_max_bullets: int = Field(
        default=6,
        description="Max bullet sentences in rolling summary (compression knob)",
    )
    memory_window_size: int = 5

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    expose_internal_errors: bool = Field(
        default=False,
        description="If true, /chat 500 responses include full exception text (dev only).",
    )


settings = Settings()

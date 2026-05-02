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

    # LLM — use LLM_PROVIDER=openai | anthropic | groq | ollama
    llm_provider: str = Field(
        default="openai",
        description="openai | anthropic | groq | ollama",
    )
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
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

    # Embeddings & stores
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "cogniflow_docs"
    sqlite_db_path: str = "./data/memory.db"

    # Chunking & conversation
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_conversation_length: int = 50
    summary_threshold: int = 10
    memory_window_size: int = 5

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()

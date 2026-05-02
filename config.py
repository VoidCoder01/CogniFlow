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

    # LLM
    llm_provider: str = Field(default="groq", description="groq | openai | ollama")
    groq_api_key: str = ""
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048

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

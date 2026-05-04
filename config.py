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

    # LangGraph checkpointing: sqlite (persistent thread state) or memory (ephemeral)
    checkpoint_backend: str = Field(
        default="sqlite",
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
    chroma_server_host: str = Field(
        default="",
        description=(
            "If set (e.g. `chroma` in Docker), use ChromaDB HttpClient to this host; "
            "otherwise use embedded PersistentClient under CHROMA_PERSIST_DIR."
        ),
    )
    chroma_server_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port for remote Chroma server (CHROMA_SERVER_HOST).",
    )
    sqlite_db_path: str = "./data/memory.db"

    # Chunking & conversation
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_conversation_length: int = 200
    summary_threshold: int = 2
    summary_max_bullets: int = Field(
        default=6,
        description="Max bullet sentences in rolling summary (compression knob)",
    )
    memory_window_size: int = 10
    memory_pruning_strategy: str = Field(
        default="relevance",
        description="relevance | sliding | summary — how to trim user_memory after turns",
    )
    summary_compression_ratio: float = Field(
        default=0.3,
        ge=0.1,
        le=0.8,
        description="Target compression ratio for conversation summaries vs raw message volume",
    )

    # RAG quality gate (cosine-style distance → relevance = 1 - distance, capped [0,1])
    retrieval_min_relevance: float = Field(
        default=0.18,
        ge=0.0,
        le=1.0,
        description=(
            "Chunks below this relevance are dropped. If none remain and this chat has "
            "indexed docs, skip the main LLM and return a fast 'not in your documents' reply."
        ),
    )

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_auth_enabled: bool = Field(
        default=False,
        description="If true, require valid X-API-Key on API routes (except /health).",
    )
    api_admin_secret: str = Field(
        default="",
        description="If set, POST /users/{user_id}/api-keys requires matching X-Admin-Secret to mint keys.",
    )
    expose_internal_errors: bool = Field(
        default=False,
        description=(
            "If true, HTTP 500 and SSE error events include full exception text (dev only). "
            "When false, clients get a generic message; details stay in server logs."
        ),
    )

    # Response cache (exact + optional semantic similarity; cleared on upload for that session)
    chat_exact_message_cache_enabled: bool = Field(
        default=True,
        description="Enable response cache (per session+user); cleared on document upload.",
    )
    chat_exact_message_cache_max_entries: int = Field(default=512, ge=16)
    chat_message_cache_normalize_whitespace: bool = Field(
        default=True,
        description="Collapse repeated spaces/newlines before cache lookup / embedding.",
    )
    chat_response_cache_mode: str = Field(
        default="both",
        description="exact | semantic | both — both = exact string match then embedding similarity.",
    )
    chat_response_cache_min_similarity: float = Field(
        default=0.82,
        ge=0.5,
        le=1.0,
        description="Minimum cosine similarity for semantic cache hits (same session+user).",
    )
    chat_response_cache_backend: str = Field(
        default="sqlite",
        description="sqlite (persistent, survives restarts) | memory (process-local only).",
    )
    chat_response_cache_sqlite_path: str = Field(
        default="./data/response_cache.db",
        description="SQLite path when backend is sqlite (mount with other ./data volumes).",
    )
    chat_response_cache_include_context: bool = Field(
        default=False,
        description=(
            "If true, cache keys include rolling summary + recent messages — safer RAG answers "
            "per thread state; repeats after long chats miss more often."
        ),
    )
    chat_response_cache_context_messages: int = Field(
        default=16,
        ge=0,
        description="Messages from DB tail hashed into cache fingerprint when include_context is true.",
    )


settings = Settings()

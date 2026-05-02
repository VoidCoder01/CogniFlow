from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class QueryIntent(str, Enum):
    factual = "factual"
    follow_up = "follow_up"
    clarification = "clarification"
    comparison = "comparison"
    multi_part = "multi_part"
    greeting = "greeting"
    off_topic = "off_topic"


class RetrievalStrategy(str, Enum):
    semantic = "semantic"
    keyword = "keyword"
    hybrid = "hybrid"
    none = "none"


# ---------------------------------------------------------------------------
# Session & Message models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = Field(default_factory=dict)


class Session(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    messages: list[ChatMessage] = Field(default_factory=list)
    summary: str = ""
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Document models
# ---------------------------------------------------------------------------

class DocumentMetadata(BaseModel):
    source: str
    doc_type: str
    title: str
    section_headers: list[str] = Field(default_factory=list)
    has_code_blocks: bool = False
    version: str = ""
    page_number: Optional[int] = None
    chunk_index: int = 0
    total_chunks: int = 1
    # API uploads: disambiguate same filename (e.g. two README.md) and avoid temp paths in citations
    original_filename: str = ""
    doc_instance_id: str = ""
    # Vector scope: chunks belong to one chat session (or "__global__" for CLI ingest)
    session_id: str = ""
    # Content-level dedupe key (sha256) for "do not re-index same file bytes"
    content_hash: str = ""


class DocumentChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    metadata: DocumentMetadata
    embedding: Optional[list[float]] = None


# ---------------------------------------------------------------------------
# Agent state (shared LangGraph pipeline state)
# ---------------------------------------------------------------------------

class AgentState(BaseModel):
    session_id: str
    user_id: str
    user_query: str
    conversation_history: list[ChatMessage] = Field(default_factory=list)
    user_memory_context: str = ""
    query_intent: Optional[QueryIntent] = None
    needs_history: bool = False
    needs_rewrite: bool = False
    rewritten_query: str = ""
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.semantic
    retrieved_documents: list[dict] = Field(default_factory=list)
    synthesized_context: str = ""
    response: str = ""
    should_summarize: bool = False
    conversation_summary: str = ""
    memory_updates: list[dict] = Field(default_factory=list)
    agent_log: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    user_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    sources: list[dict] = Field(default_factory=list)
    agent_log: list[dict] = Field(default_factory=list)
    latency_seconds: Optional[float] = None
    conversation_summary: str = ""


class SessionCreateRequest(BaseModel):
    user_id: str


class SessionCreateResponse(BaseModel):
    session_id: str
    user_id: str
    created_at: datetime


class DocumentUploadResponse(BaseModel):
    filename: str
    num_chunks: int
    status: str

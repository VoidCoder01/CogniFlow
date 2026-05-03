from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class QueryRouterResult(BaseModel):
    """Structured output from the query-understanding (router) LLM."""

    intent: str = Field(
        description=(
            "One of: factual_doc, multi_part (multiple distinct questions in one message), "
            "general_knowledge, follow_up, meta, preference, "
            "session_recall (recap of what was asked/discussed in the current chat only)"
        )
    )
    needs_retrieval: bool = Field(
        description="True only when the answer should use indexed documents"
    )
    needs_memory: bool = Field(
        description="True when cross-session or stated user context/preferences matter"
    )
    response_style: str = Field(
        default="short",
        description="short | detailed — how verbose the assistant reply should be",
    )

    @field_validator("response_style", mode="before")
    @classmethod
    def _norm_style(cls, v: object) -> str:
        s = str(v or "short").lower().strip()
        return "detailed" if s == "detailed" else "short"


class ContextValidationResult(BaseModel):
    """Whether retrieved chunks are sufficient to ground the reply."""

    use_context: bool = Field(
        description="True if passages are on-topic and strong enough to cite; false to fall back to general knowledge"
    )
    reason: str = Field(default="", description="Short justification for logging")


class RetrievalRoutingResult(BaseModel):
    strategy: str = Field(
        description="semantic | keyword | hybrid | none — retrieval strategy for this turn"
    )
    rationale: str = Field(default="", description="Short reason for logging")


class MemoryItem(BaseModel):
    memory_type: str = Field(description="preference | context | decision | issue")
    content: str
    metadata: dict = Field(default_factory=dict)


class MemoryExtractionResult(BaseModel):
    items: list[MemoryItem] = Field(default_factory=list)


class QueryDecompositionResult(BaseModel):
    sub_queries: list[str] = Field(default_factory=list)

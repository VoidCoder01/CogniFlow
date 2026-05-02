from __future__ import annotations

from pydantic import BaseModel, Field


class QueryUnderstandingResult(BaseModel):
    intent: str = Field(
        description="One of: factual, follow_up, clarification, comparison, multi_part, greeting, off_topic"
    )
    needs_history: bool = Field(description="Whether conversation history is required to answer")
    needs_rewrite: bool = Field(description="Whether the user query should be rewritten with context")


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

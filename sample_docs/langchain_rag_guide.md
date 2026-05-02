# LangChain RAG Notes

## Retrieval-augmented generation

RAG combines a retriever (vector or keyword search) with an LLM. The retriever returns passages; the model answers using those passages as grounding.

## Chunking

Chunk size and overlap affect recall. Technical documentation often benefits from section-aware splitting before recursive character splitting.

## Version

These patterns align with LangChain 0.3-style component APIs.

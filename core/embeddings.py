from __future__ import annotations

import logging
import os

from config import settings

logger = logging.getLogger(__name__)

# Avoid tokenizer fork warnings and oversubscription on laptops
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class EmbeddingManager:
    """
    Text embeddings for ChromaDB.

    - ``local``: SentenceTransformer on ``EMBEDDING_DEVICE`` (default ``cpu`` — no GPU required).
    - ``openai``: OpenAI API — no PyTorch/GPU on this machine; requires ``OPENAI_API_KEY``.
      If you switch backend or model, use a new ``CHROMA_COLLECTION_NAME`` or clear ``CHROMA_PERSIST_DIR``
      and re-run ingestion (vector dimensions must match).
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._backend = settings.embedding_backend.lower().strip()
        if self._backend not in ("local", "openai"):
            raise ValueError(
                f"EMBEDDING_BACKEND must be 'local' or 'openai', got {settings.embedding_backend!r}"
            )
        self._local_model = None
        self._openai_embedder = None
        self._dim_cache: int | None = None

    @property
    def model(self):
        if self._backend != "local":
            raise RuntimeError("SentenceTransformer model is only used when EMBEDDING_BACKEND=local")
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer

            dev = (settings.embedding_device or "cpu").strip().lower()
            logger.info("Loading embedding model %s on device=%s", self.model_name, dev)
            self._local_model = SentenceTransformer(self.model_name, device=dev)
        return self._local_model

    def _openai(self):
        if self._openai_embedder is None:
            from langchain_openai import OpenAIEmbeddings

            if not settings.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required when EMBEDDING_BACKEND=openai "
                    "(remote embeddings; no local GPU needed)."
                )
            self._openai_embedder = OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                model=settings.openai_embedding_model,
            )
        return self._openai_embedder

    def embed_text(self, text: str) -> list[float]:
        if self._backend == "openai":
            vec = self._openai().embed_query(text)
            return list(vec)
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self._backend == "openai":
            rows = self._openai().embed_documents(texts)
            return [list(r) for r in rows]
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    @property
    def dimension(self) -> int:
        if self._dim_cache is not None:
            return self._dim_cache
        if self._backend == "openai":
            self._dim_cache = len(self.embed_text("."))
            return self._dim_cache
        self._dim_cache = self.model.get_sentence_embedding_dimension()
        return self._dim_cache

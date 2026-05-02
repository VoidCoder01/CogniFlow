from __future__ import annotations

from sentence_transformers import SentenceTransformer

from config import settings


class EmbeddingManager:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_text(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

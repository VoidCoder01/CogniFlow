from __future__ import annotations

import os
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings
from core.embeddings import EmbeddingManager
from core.models import DocumentChunk, DocumentMetadata


class VectorStore:
    """Chroma-backed dense / lexical retrieval with session+user scoped filters."""

    def __init__(
        self,
        persist_dir: str = None,
        collection_name: str = None,
    ):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection_name

        os.makedirs(self.persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_manager = EmbeddingManager()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_documents(self, chunks: list[DocumentChunk]):
        """Embed chunk texts and upsert into the configured collection."""
        ids = [chunk.id for chunk in chunks]
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_manager.embed_texts(texts)

        metadatas = []
        for chunk in chunks:
            m: DocumentMetadata = chunk.metadata
            metadatas.append(
                {
                    "source": m.source,
                    "doc_type": m.doc_type,
                    "title": m.title,
                    "section_headers": "|".join(m.section_headers),
                    "has_code_blocks": str(m.has_code_blocks),
                    "version": m.version,
                    "page_number": m.page_number if m.page_number is not None else 0,
                    "chunk_index": m.chunk_index,
                    "total_chunks": m.total_chunks,
                    "original_filename": m.original_filename or "",
                    "doc_instance_id": m.doc_instance_id or "",
                    "session_id": m.session_id or "",
                    "user_id": m.user_id or "",
                    "content_hash": m.content_hash or "",
                }
            )

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            self.collection.add(
                ids=ids[i : i + batch_size],
                documents=texts[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

    # ------------------------------------------------------------------
    # Retrieval scope (session + same user for cross-session RAG)
    # ------------------------------------------------------------------

    @staticmethod
    def scope_filter(session_id: str, user_id: str) -> dict[str, Any] | None:
        sid = (session_id or "").strip()
        uid = (user_id or "").strip()
        if sid and uid:
            return {"$or": [{"session_id": sid}, {"user_id": uid}]}
        if sid:
            return {"session_id": sid}
        if uid:
            return {"user_id": uid}
        return None

    def count_chunks_for_scope(
        self, session_id: str | None = None, user_id: str | None = None
    ) -> int:
        filt = self.scope_filter(session_id or "", user_id or "")
        if filt is None:
            return int(self.collection.count())
        res = self.collection.get(where=filt)
        ids = res.get("ids") if res else None
        return len(ids) if ids else 0

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        """Embedding-based similarity search over chunk bodies."""
        query_embedding = self.embedding_manager.embed_text(query)
        kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if filter_metadata:
            kwargs["where"] = filter_metadata
        results = self.collection.query(**kwargs)
        return self._format_results(results)

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        """Chroma full-text style retrieval using raw query tokens."""
        kwargs = dict(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if filter_metadata:
            kwargs["where"] = filter_metadata
        results = self.collection.query(**kwargs)
        return self._format_results(results)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        """RRF-style fusion of semantic and keyword rankings."""
        keyword_weight = 1.0 - semantic_weight
        fetch_k = top_k * 2

        semantic_results = self.semantic_search(query, fetch_k, filter_metadata)
        keyword_results = self.keyword_search(query, fetch_k, filter_metadata)

        k = 60
        scores: dict[str, float] = {}
        merged: dict[str, dict] = {}

        for rank, doc in enumerate(semantic_results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + semantic_weight / (k + rank + 1)
            merged[doc_id] = doc

        for rank, doc in enumerate(keyword_results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + keyword_weight / (k + rank + 1)
            if doc_id not in merged:
                merged[doc_id] = doc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [merged[doc_id] for doc_id, _ in ranked[:top_k]]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_results(self, results: dict) -> list[dict]:
        if not results or not results.get("ids") or not results["ids"][0]:
            return []
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        return [
            {
                "id": ids[i],
                "content": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i],
            }
            for i in range(len(ids))
        ]

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------

    def get_collection_stats(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> dict:
        """Chunk count: whole collection, or scoped to session, user, or union (both set)."""
        sid = (session_id or "").strip()
        uid = (user_id or "").strip()
        if not sid and not uid:
            return {"name": self.collection_name, "count": self.collection.count()}
        if sid and uid:
            n = self.count_chunks_for_scope(session_id=sid, user_id=uid)
            return {
                "name": self.collection_name,
                "count": n,
                "session_id": sid,
                "user_id": uid,
                "scope": "session_or_user",
            }
        if sid:
            res = self.collection.get(where={"session_id": sid})
            ids = res.get("ids") if res else None
            n = len(ids) if ids else 0
            return {"name": self.collection_name, "count": n, "session_id": sid}
        res = self.collection.get(where={"user_id": uid})
        ids = res.get("ids") if res else None
        n = len(ids) if ids else 0
        return {"name": self.collection_name, "count": n, "user_id": uid}

    def has_document(self, session_id: str, content_hash: str) -> bool:
        sid = (session_id or "").strip()
        digest = (content_hash or "").strip()
        if not sid or not digest:
            return False
        res = self.collection.get(
            where={"$and": [{"session_id": sid}, {"content_hash": digest}]},
            limit=1,
        )
        ids = res.get("ids") if res else None
        return bool(ids)

    def has_user_document(self, user_id: str, content_hash: str) -> bool:
        """Same bytes already indexed for this user (any session)."""
        uid = (user_id or "").strip()
        digest = (content_hash or "").strip()
        if not uid or not digest:
            return False
        res = self.collection.get(
            where={"$and": [{"user_id": uid}, {"content_hash": digest}]},
            limit=1,
        )
        ids = res.get("ids") if res else None
        return bool(ids)

    def delete_collection(self):
        self._client.delete_collection(self.collection_name)
        self.collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

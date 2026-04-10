from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata,
            "embedding": embedding,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_embedding = self._embedding_fn(query)
        scored = []
        for record in records:
            score = _dot(query_embedding, record["embedding"])
            scored.append({
                "content": record["content"],
                "metadata": record["metadata"],
                "score": score,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        for doc in docs:
            record = self._make_record(doc)
            self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        if metadata_filter is None:
            return self.search(query, top_k)

        filtered = [
            r for r in self._store
            if all(r["metadata"].get(k) == v for k, v in metadata_filter.items())
        ]
        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        before = len(self._store)
        self._store = [r for r in self._store if r["id"] != doc_id]
        return len(self._store) < before

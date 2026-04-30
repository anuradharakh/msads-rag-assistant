from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.embeddings import EmbeddingClient
from src.vector_store import ChromaVectorStore


class Retriever:
    def __init__(self) -> None:
        self.embedder = EmbeddingClient()
        self.store = ChromaVectorStore()

    def search(
        self,
        query: str,
        n_results: int = 5,
        content_type: Optional[str] = None,
        program_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.embed_batch([query])[0]

        where = self._build_where_filter(
            content_type=content_type,
            program_type=program_type,
        )

        raw = self.store.query(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where,
        )

        results: List[Dict[str, Any]] = []

        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for doc_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            results.append(
                {
                    "id": doc_id,
                    "distance": distance,
                    "score": 1 - distance if isinstance(distance, (int, float)) else None,
                    "metadata": metadata,
                    "document": document,
                }
            )

        return results

    def _build_where_filter(
        self,
        content_type: Optional[str],
        program_type: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        filters = []

        if content_type:
            filters.append({"content_type": content_type})

        if program_type:
            filters.append({"program_type": program_type})

        if not filters:
            return None

        if len(filters) == 1:
            return filters[0]

        return {"$and": filters}
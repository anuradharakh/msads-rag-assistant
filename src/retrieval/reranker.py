from __future__ import annotations

from typing import Any, Dict, List

from sentence_transformers import CrossEncoder

from src.config import settings


class Reranker:
    """
    Cross-encoder reranker.

    Embedding search is fast but approximate.
    Reranking is slower but more accurate because it scores:
        (query, document)
    pairs directly.
    """

    def __init__(self) -> None:
        self.model_name = settings.reranker_model
        self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if not results:
            return []

        pairs = [
            [query, result.get("document", "")]
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for result, rerank_score in zip(results, scores):
            updated = dict(result)
            updated["rerank_score"] = float(rerank_score)
            reranked.append(updated)

        reranked.sort(
            key=lambda item: item.get("rerank_score", float("-inf")),
            reverse=True,
        )

        return reranked[:top_k]
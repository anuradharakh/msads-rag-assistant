from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer

from src.config import settings


class EmbeddingClient:
    """
    Local embedding client using BAAI/bge-base-en-v1.5.

    BGE works best when retrieval text is prefixed with:
    "Represent this sentence for searching relevant passages: "
    """

    def __init__(self) -> None:
        self.model_name = settings.embedding_model
        self.model = SentenceTransformer(self.model_name)

    def _format_for_embedding(self, text: str) -> str:
        return f"Represent this sentence for searching relevant passages: {text.strip()}"

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        cleaned = [self._format_for_embedding(t) for t in texts if t and t.strip()]

        if not cleaned:
            return []

        embeddings = self.model.encode(
            cleaned,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()
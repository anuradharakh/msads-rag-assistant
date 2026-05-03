from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.config import settings
from src.generation.llm import LocalLLMClient
from src.generation.prompts import SYSTEM_PROMPT, build_rag_prompt
from src.retrieval.retriever import Retriever


class RAGChain:
    def __init__(self) -> None:
        self.retriever = Retriever()
        self.llm = LocalLLMClient()

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        content_type: Optional[str] = None,
        program_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        retrieved = self.retriever.search(
            query=question,
            n_results=top_k or settings.rag_top_k,
            content_type=content_type,
            program_type=program_type,
        )

        filtered = self._filter_low_score(retrieved)

        if not filtered:
            return {
                "question": question,
                "answer": "I don't have enough information in the indexed MSADS sources to answer that.",
                "sources": [],
                "retrieved_chunks": retrieved,
            }

        prompt = build_rag_prompt(question, filtered)

        answer = self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
        )

        return {
            "question": question,
            "answer": answer,
            "sources": self._build_sources(filtered),
            "retrieved_chunks": filtered,
        }

    def _filter_low_score(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered = []

        for result in results:
            score = result.get("score")

            if score is None:
                filtered.append(result)
                continue

            if score >= settings.rag_min_score:
                filtered.append(result)

        return filtered

    def _build_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources = []

        for idx, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {})

            sources.append(
                {
                    "source_id": f"Source {idx}",
                    "url": metadata.get("url"),
                    "page_title": metadata.get("page_title"),
                    "section_title": metadata.get("section_title"),
                    "content_type": metadata.get("content_type"),
                    "score": chunk.get("score"),
                }
            )

        return sources
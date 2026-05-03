from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.config import settings
from src.generation.llm import LocalLLMClient
from src.generation.prompts import SYSTEM_PROMPT, build_rag_prompt
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever


class RAGChain:
    def __init__(self) -> None:
        self.retriever = Retriever()
        self.llm = LocalLLMClient()
        self.reranker = Reranker() if settings.use_reranker else None

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        content_type: Optional[str] = None,
        program_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        initial_k = settings.rerank_initial_top_k if settings.use_reranker else (top_k or settings.rag_top_k)
        final_k = top_k or settings.rerank_final_top_k or settings.rag_top_k

        retrieved = self.retriever.search(
            query=question,
            n_results=initial_k,
            content_type=content_type,
            program_type=program_type,
        )

        retrieved = self._sort_by_score(retrieved)
        retrieved = self._dedupe_results(retrieved)

        filtered = self._filter_low_score(retrieved)

        if self.reranker:
            filtered = self.reranker.rerank(
                query=question,
                results=filtered,
                top_k=final_k,
            )
        else:
            filtered = filtered[:final_k]

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

    def _sort_by_score(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            results,
            key=lambda item: item.get("score") if item.get("score") is not None else float("-inf"),
            reverse=True,
        )

    def _dedupe_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: set[Tuple[str, str]] = set()
        deduped = []

        for result in results:
            metadata = result.get("metadata", {})
            url = metadata.get("url", "")
            section = metadata.get("section_title", "")
            key = (url, section)

            if key in seen:
                continue

            seen.add(key)
            deduped.append(result)

        return deduped

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
                    "rerank_score": chunk.get("rerank_score"),
                }
            )

        return sources
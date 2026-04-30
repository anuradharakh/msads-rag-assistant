from __future__ import annotations

from typing import List

from src.models import PageSection
from src.utils import (
    approx_token_count,
    clean_text,
    make_summary,
    sha256_hash,
    split_sentences,
)


class SectionChunker:
    def __init__(
        self,
        max_tokens: int = 690,
        overlap_sentences: int = 1,
    ) -> None:
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences

    def chunk_sections(self, sections: List[PageSection]) -> List[PageSection]:
        chunked: List[PageSection] = []

        for section in sections:
            if section.token_count <= self.max_tokens:
                section.chunk_id = section.doc_id
                section.parent_doc_id = None
                section.chunk_index = 0
                section.chunk_count = 1
                section.is_chunk = False
                chunked.append(section)
                continue

            chunks = self._split_section_text(section.content_clean or section.content)

            if not chunks:
                chunked.append(section)
                continue

            for idx, text in enumerate(chunks):
                chunked.append(self._build_chunk(section, text, idx, len(chunks)))

        return chunked

    def final_dedupe_chunks(self, chunks: List[PageSection]) -> List[PageSection]:
        seen = set()
        final: List[PageSection] = []

        for chunk in chunks:
            fp = sha256_hash(clean_text(chunk.content_clean or chunk.content).lower())
            if fp in seen:
                continue

            seen.add(fp)
            final.append(chunk)

        return final

    def _split_section_text(self, text: str) -> List[str]:
        sentences = split_sentences(text)
        chunks: List[str] = []
        current: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = approx_token_count(sentence)

            if sentence_tokens > self.max_tokens:
                if current:
                    chunks.append(" ".join(current).strip())
                    current = []
                    current_tokens = 0

                chunks.extend(self._split_long_text(sentence))
                continue

            if current and current_tokens + sentence_tokens > self.max_tokens:
                chunks.append(" ".join(current).strip())

                current = current[-self.overlap_sentences:] if self.overlap_sentences else []
                current_tokens = approx_token_count(" ".join(current)) if current else 0

            current.append(sentence)
            current_tokens += sentence_tokens

        if current:
            chunks.append(" ".join(current).strip())

        return [clean_text(chunk) for chunk in chunks if clean_text(chunk)]

    def _split_long_text(self, text: str) -> List[str]:
        words = text.split()
        max_words = max(80, int(self.max_tokens * 0.55))
        chunks = []

        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words]).strip()
            if chunk:
                chunks.append(chunk)

        return chunks

    def _build_chunk(
        self,
        section: PageSection,
        text: str,
        chunk_index: int,
        chunk_count: int,
    ) -> PageSection:
        chunk_doc_id = sha256_hash(
            f"{section.doc_id}|chunk|{chunk_index}|{text}".lower()
        )

        payload = section.model_dump()
        payload.update(
            {
                "doc_id": chunk_doc_id,
                "parent_doc_id": section.doc_id,
                "chunk_id": chunk_doc_id,
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "content": text,
                "content_clean": clean_text(text),
                "content_summary": make_summary(text),
                "token_count": approx_token_count(text),
                "char_count": len(text),
                "is_chunk": True,
            }
        )

        return PageSection(**payload)
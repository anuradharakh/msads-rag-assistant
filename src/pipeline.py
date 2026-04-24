from __future__ import annotations

from pathlib import Path
from typing import List

from src.config import settings
from src.crawler import WebCrawler
from src.extractor import ContentExtractor
from src.models import PageSection, RawPage
from src.utils import (
    approx_token_count,
    clean_text,
    ensure_dirs,
    make_summary,
    sha256_hash,
    split_sentences,
    write_jsonl,
)


class IngestionPipeline:
    def __init__(self) -> None:
        self.crawler = WebCrawler()
        self.extractor = ContentExtractor()

    def _deduplicate_sections(self, sections: List[PageSection]) -> List[PageSection]:
        seen = {}
        deduped: List[PageSection] = []

        for section in sections:
            key = sha256_hash(
                f"{section.url}|{section.section_title}|{section.content_clean or section.content}".lower()
            )
            near_key = sha256_hash(clean_text(section.content).lower())

            if key in seen:
                continue

            if near_key in seen and section.url == seen[near_key].url:
                section.duplicate_of = seen[near_key].doc_id
                continue

            seen[key] = section
            seen[near_key] = section
            deduped.append(section)

        return deduped

    def _chunk_sections(
        self,
        sections: List[PageSection],
        max_tokens: int = 690,
        overlap_sentences: int = 1,
    ) -> List[PageSection]:
        chunked: List[PageSection] = []

        for section in sections:
            if section.token_count <= max_tokens:
                section.chunk_id = section.doc_id
                section.parent_doc_id = None
                section.chunk_index = 0
                section.chunk_count = 1
                section.is_chunk = False
                chunked.append(section)
                continue

            # Keep FAQ records intact when possible. FAQ chunks must preserve Q/A fields.
            sentences = split_sentences(section.content_clean or section.content)
            chunks: List[str] = []
            current: List[str] = []
            current_tokens = 0

            for sentence in sentences:
                t = approx_token_count(sentence)

                # Hard fallback for a single sentence that is too long.
                if t > max_tokens:
                    if current:
                        chunks.append(" ".join(current).strip())
                        current = []
                        current_tokens = 0
                    chunks.extend(self._split_long_text(sentence, max_tokens=max_tokens))
                    continue

                if current and current_tokens + t > max_tokens:
                    chunks.append(" ".join(current).strip())
                    current = current[-overlap_sentences:] if overlap_sentences else []
                    current_tokens = approx_token_count(" ".join(current)) if current else 0

                current.append(sentence)
                current_tokens += t

            if current:
                chunks.append(" ".join(current).strip())

            if not chunks:
                chunked.append(section)
                continue

            for idx, text in enumerate(chunks):
                chunk_doc_id = sha256_hash(f"{section.doc_id}|chunk|{idx}|{text}".lower())
                payload = section.model_dump()
                payload.update({
                    "doc_id": chunk_doc_id,
                    "parent_doc_id": section.doc_id,
                    "chunk_id": chunk_doc_id,
                    "chunk_index": idx,
                    "chunk_count": len(chunks),
                    "content": text,
                    "content_clean": clean_text(text),
                    "content_summary": make_summary(text),
                    "token_count": approx_token_count(text),
                    "char_count": len(text),
                    "is_chunk": True,
                })

                # Preserve structured fields for specialized records.
                if section.content_type == "faq":
                    payload["question"] = section.question
                    payload["answer"] = section.answer
                    payload["faq_category"] = section.faq_category

                if section.content_type == "course":
                    payload["course_name"] = section.course_name
                    payload["course_type"] = section.course_type
                    payload["course_names"] = section.course_names or ([section.course_name] if section.course_name else [])

                if section.content_type == "faculty_bio":
                    payload["faculty_name"] = section.faculty_name
                    payload["academic_title"] = section.academic_title
                    payload["industry_role"] = section.industry_role

                chunked.append(PageSection(**payload))

        return chunked

    def _split_long_text(self, text: str, max_tokens: int) -> List[str]:
        words = text.split()
        # Approximate safe word count; approx_token_count is conservative.
        max_words = max(80, int(max_tokens * 0.55))
        chunks = []

        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words]).strip()
            if chunk:
                chunks.append(chunk)

        return chunks

    def _final_dedupe_chunks(self, chunks: List[PageSection]) -> List[PageSection]:
        seen = set()
        final: List[PageSection] = []

        for chunk in chunks:
            fp = sha256_hash(clean_text(chunk.content_clean or chunk.content).lower())
            if fp in seen:
                continue
            seen.add(fp)
            final.append(chunk)

        return final

    def _save_raw_pages(self, raw_pages: List[RawPage]) -> None:
        if not settings.save_raw_html:
            return

        for raw_page in raw_pages:
            file_name = f"{sha256_hash(str(raw_page.url))}.html"
            file_path = Path(settings.raw_html_dir) / file_name

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(raw_page.html)

    def run(self) -> str:
        print("[INFO] Creating output directories...")
        ensure_dirs([settings.output_dir, settings.raw_html_dir, settings.log_dir])

        print("[INFO] Crawling seed URLs...")
        raw_pages = self.crawler.crawl(settings.seed_urls)
        print(f"[INFO] Raw pages crawled: {len(raw_pages)}")

        print("[INFO] Saving raw HTML pages...")
        self._save_raw_pages(raw_pages)

        all_sections: List[PageSection] = []
        for raw_page in raw_pages:
            all_sections.extend(self.extractor.extract(raw_page))

        print(f"[INFO] Sections before dedup: {len(all_sections)}")
        all_sections = self._deduplicate_sections(all_sections)
        print(f"[INFO] Sections after dedup: {len(all_sections)}")

        all_chunks = self._chunk_sections(all_sections, max_tokens=690)
        all_chunks = self._final_dedupe_chunks(all_chunks)
        print(f"[INFO] Records after chunking + final dedup: {len(all_chunks)}")

        output_path = f"{settings.output_dir}/msads_processed_data.jsonl"
        write_jsonl([section.model_dump(mode="json") for section in all_chunks], output_path)

        faq_path = f"{settings.output_dir}/msads_faq_items.jsonl"
        write_jsonl(
            [
                s.model_dump(mode="json")
                for s in all_chunks
                if s.content_type == "faq" and s.question and s.answer
            ],
            faq_path,
        )

        print(f"[INFO] Output written to: {output_path}")
        print(f"[INFO] FAQ output written to: {faq_path}")
        return output_path

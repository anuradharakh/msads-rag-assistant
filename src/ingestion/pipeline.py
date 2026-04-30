from __future__ import annotations

from pathlib import Path
from typing import List

from src.config import settings
from src.ingestion.crawler import WebCrawler
from src.ingestion.extractor import ContentExtractor
from src.models import PageSection, RawPage
from src.processing.chunking import SectionChunker
from src.utils import clean_text, ensure_dirs, sha256_hash, write_jsonl


class IngestionPipeline:
    def __init__(self) -> None:
        self.crawler = WebCrawler()
        self.extractor = ContentExtractor()
        self.chunker = SectionChunker(max_tokens=690, overlap_sentences=1)

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

        all_chunks = self.chunker.chunk_sections(all_sections)
        all_chunks = self.chunker.final_dedupe_chunks(all_chunks)

        print(f"[INFO] Records after chunking + final dedup: {len(all_chunks)}")

        output_path = f"{settings.output_dir}/msads_processed_data.jsonl"
        write_jsonl([section.model_dump(mode="json") for section in all_chunks], output_path)

        faq_path = f"{settings.output_dir}/msads_faq_items.jsonl"
        write_jsonl(
            [
                section.model_dump(mode="json")
                for section in all_chunks
                if section.content_type == "faq" and section.question and section.answer
            ],
            faq_path,
        )

        print(f"[INFO] Output written to: {output_path}")
        print(f"[INFO] FAQ output written to: {faq_path}")

        return output_path
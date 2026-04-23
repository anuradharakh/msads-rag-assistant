from pathlib import Path
from typing import List

from src.config import settings
from src.crawler import WebCrawler
from src.extractor import ContentExtractor
from src.models import PageSection, RawPage
from src.utils import ensure_dirs, sha256_hash, write_jsonl


class IngestionPipeline:
    def __init__(self) -> None:
        self.crawler = WebCrawler()
        self.extractor = ContentExtractor()

    def _deduplicate_sections(self, sections: List[PageSection]) -> List[PageSection]:
        seen = set()
        deduped = []

        for section in sections:
            content_fingerprint = sha256_hash(section.content.strip().lower())
            if content_fingerprint in seen:
                continue
            seen.add(content_fingerprint)
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
        ensure_dirs([
            settings.output_dir,
            settings.raw_html_dir,
            settings.log_dir
        ])

        print("[INFO] Crawling seed URLs...")
        raw_pages = self.crawler.crawl(settings.seed_urls)
        print(f"[INFO] Raw pages crawled: {len(raw_pages)}")

        print("[INFO] Saving raw HTML pages...")
        self._save_raw_pages(raw_pages)

        all_sections: List[PageSection] = []
        for raw_page in raw_pages:
            sections = self.extractor.extract(raw_page)
            all_sections.extend(sections)

        print(f"[INFO] Sections before dedup: {len(all_sections)}")
        all_sections = self._deduplicate_sections(all_sections)
        print(f"[INFO] Sections after dedup: {len(all_sections)}")

        output_path = f"{settings.output_dir}/canonical_sections.jsonl"
        write_jsonl([section.model_dump(mode="json") for section in all_sections], output_path)

        print(f"[INFO] Output written to: {output_path}")
        return output_path
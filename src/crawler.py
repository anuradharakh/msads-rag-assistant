from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import List, Set

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.models import RawPage
from src.utils import canonicalize_url, is_allowed_url, normalize_url

SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".zip", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".mp4",
)


class WebCrawler:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.user_agent})

    @retry(stop=stop_after_attempt(settings.max_retries), wait=wait_exponential(multiplier=1, min=1, max=8))
    def fetch_page(self, url: str) -> RawPage:
        response = self.session.get(
            url,
            timeout=settings.request_timeout_seconds,
            verify=settings.verify_ssl,
            allow_redirects=True,
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type:
            raise ValueError(f"Skipping non-HTML content: {url} ({content_type})")

        soup = BeautifulSoup(response.text, "lxml")
        title = soup.title.get_text(" ", strip=True) if soup.title else None
        canonical = None
        canonical_tag = soup.find("link", rel=lambda v: v and "canonical" in v)
        if canonical_tag and canonical_tag.get("href"):
            canonical = canonicalize_url(normalize_url(str(response.url), canonical_tag["href"]) or str(response.url))
        meta_desc = soup.find("meta", attrs={"name": "description"})

        return RawPage(
            url=canonicalize_url(url),
            final_url=canonicalize_url(str(response.url)),
            html=response.text,
            fetched_at=datetime.now(timezone.utc),
            status_code=response.status_code,
            page_title=title,
            canonical_url=canonical or canonicalize_url(str(response.url)),
            meta_description=meta_desc.get("content", "").strip() if meta_desc else None,
            last_modified=response.headers.get("Last-Modified"),
            etag=response.headers.get("ETag"),
        )

    def extract_links(self, base_url: str, html: str) -> List[str]:
        soup = BeautifulSoup(html, "lxml")
        links: List[str] = []
        for a in soup.find_all("a", href=True):
            normalized = normalize_url(base_url, a["href"])
            if not normalized:
                continue
            if normalized.lower().endswith(SKIP_EXTENSIONS):
                continue
            if is_allowed_url(normalized, settings.allowed_domains, settings.allowed_path_keywords):
                links.append(normalized)
        return list(dict.fromkeys(links))

    def crawl(self, seed_urls: List[str]) -> List[RawPage]:
        visited: Set[str] = set()
        queue = deque(canonicalize_url(url) for url in seed_urls)
        results: List[RawPage] = []

        while queue and len(results) < settings.max_pages:
            url = queue.popleft()
            normalized_url = canonicalize_url(url)
            if normalized_url in visited:
                continue
            visited.add(normalized_url)
            try:
                page = self.fetch_page(normalized_url)
                results.append(page)
                for link in self.extract_links(str(page.final_url or page.url), page.html):
                    canonical_link = canonicalize_url(link)
                    if canonical_link not in visited:
                        queue.append(canonical_link)
            except Exception as exc:
                print(f"[WARN] Failed to crawl {normalized_url}: {exc}")
        return results

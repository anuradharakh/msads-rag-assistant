from collections import deque
from datetime import datetime, timezone
from typing import List, Set, Tuple

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.models import RawPage
from src.utils import is_allowed_url, normalize_url


SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif",
    ".svg", ".zip", ".doc", ".docx", ".xls", ".xlsx"
)

class WebCrawler:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": settings.user_agent
        })

    @retry(stop=stop_after_attempt(settings.max_retries), wait=wait_exponential(multiplier=1, min=1, max=8))
    def fetch_page(self, url: str) -> RawPage:
        response = self.session.get(
            url,
            timeout=settings.request_timeout_seconds,
            verify=settings.verify_ssl
        )

        content_type = response.headers.get("Content-Type", "")

        if "text/html" not in content_type:
            raise ValueError(f"Skipping non-HTML content: {url} ({content_type})")
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")
        title = soup.title.get_text(strip=True) if soup.title else None

        return RawPage(
            url=url,
            html=response.text,
            fetched_at=datetime.now(timezone.utc),
            status_code=response.status_code,
            page_title=title
        )

    def extract_links(self, base_url: str, html: str) -> List[str]:
        soup = BeautifulSoup(html, "lxml")
        links = []

        for a in soup.find_all("a", href=True):
            normalized = normalize_url(base_url, a["href"])
            if not normalized:
                continue

            # Skip non-HTML file types
            if normalized.lower().endswith(SKIP_EXTENSIONS):
                continue

            if is_allowed_url(
                normalized,
                allowed_domains=settings.allowed_domains,
                allowed_path_keywords=settings.allowed_path_keywords
            ):
                links.append(normalized)

        return list(dict.fromkeys(links))

    def crawl(self, seed_urls: List[str]) -> List[RawPage]:
        visited: Set[str] = set()
        queue = deque(seed_urls)
        results: List[RawPage] = []

        while queue and len(results) < settings.max_pages:
            url = queue.popleft()
            normalized_url = url.rstrip("/")

            if normalized_url in visited:
                continue

            visited.add(normalized_url)

            try:
                page = self.fetch_page(normalized_url)
                results.append(page)

                discovered_links = self.extract_links(normalized_url, page.html)
                for link in discovered_links:
                    if link not in visited:
                        queue.append(link)

            except Exception as exc:
                print(f"[WARN] Failed to crawl {normalized_url}: {exc}")

        return results
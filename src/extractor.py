from datetime import datetime, timezone
from typing import List, Tuple

from bs4 import BeautifulSoup, Tag

from src.config import settings
from src.models import PageSection, RawPage
from src.utils import clean_text, classify_content_type, classify_program_type, sha256_hash


REMOVE_SELECTORS = [
    "nav",
    "footer",
    "script",
    "style",
    "noscript",
    "form",
    "svg",
    "aside"
]


class ContentExtractor:
    def __init__(self) -> None:
        pass

    def _remove_noise(self, soup: BeautifulSoup) -> None:
        for selector in REMOVE_SELECTORS:
            for tag in soup.select(selector):
                tag.decompose()

        for tag in soup.find_all(class_=lambda value: value and any(
            junk in " ".join(value).lower()
            for junk in ["menu", "nav", "footer", "sidebar", "breadcrumb", "share", "social"]
        )):
            tag.decompose()

    def _extract_title(self, soup: BeautifulSoup) -> str:
        h1 = soup.find("h1")
        if h1:
            text = clean_text(h1.get_text(" ", strip=True))
            if text:
                return text

        if soup.title:
            return clean_text(soup.title.get_text(" ", strip=True))

        return "Untitled Page"

    def _clean_section_text(self, section_title: str, text: str) -> str:
        text = clean_text(text)

        junk_phrases = [
            "Learn More",
            "Read More",
            "Click Here",
            "Loading…",
            "click here to download the PDF",
        ]
        for phrase in junk_phrases:
            text = text.replace(phrase, "").strip()

        if text.lower().endswith(section_title.lower()):
            text = text[:-len(section_title)].strip()

        return clean_text(text)

    def _extract_from_container(self, heading: Tag) -> str:
        """
        Try to extract section content from the closest meaningful parent container.
        This helps for card/grid layouts where content is nested inside the same wrapper.
        """
        parent = heading.parent
        if not parent or not isinstance(parent, Tag):
            return ""

        # Copy the parent subtree so we can safely modify it
        parent_clone = BeautifulSoup(str(parent), "lxml")

        # Find the cloned heading text
        cloned_heading = None
        for candidate in parent_clone.find_all(["h2", "h3"]):
            if clean_text(candidate.get_text(" ", strip=True)) == clean_text(heading.get_text(" ", strip=True)):
                cloned_heading = candidate
                break

        if not cloned_heading:
            return ""

        # Remove all headings from the cloned container so we keep only content
        for h in parent_clone.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            h.decompose()

        text = clean_text(parent_clone.get_text(" ", strip=True))
        return text

    def _extract_from_siblings(self, heading: Tag) -> str:
        """
        Fallback extraction using next siblings until the next heading.
        """
        buffer = []
        current = heading.find_next_sibling()

        while current and current.name not in {"h2", "h3"}:
            if isinstance(current, Tag):
                text = clean_text(current.get_text(separator=" ", strip=True))
                if text:
                    buffer.append(text)
            current = current.find_next_sibling()

        return clean_text(" ".join(buffer))

    def _extract_sections(self, soup: BeautifulSoup) -> List[Tuple[str, str]]:
        sections: List[Tuple[str, str]] = []

        headings = soup.find_all(["h2", "h3"])
        if not headings:
            body_text = clean_text(soup.get_text(" ", strip=True))
            if len(body_text) >= settings.min_text_length:
                sections.append(("Main Content", body_text))
            return sections

        seen = set()

        for heading in headings:
            section_title = clean_text(heading.get_text(" ", strip=True))
            if not section_title:
                continue

            # First try container-based extraction
            section_text = self._extract_from_container(heading)

            # If container extraction is too weak or too broad, fall back to siblings
            if len(section_text) < settings.min_text_length:
                section_text = self._extract_from_siblings(heading)

            section_text = self._clean_section_text(section_title, section_text)

            if len(section_text) < settings.min_text_length:
                continue

            fingerprint = f"{section_title}|{section_text}"
            if fingerprint in seen:
                continue
            seen.add(fingerprint)

            sections.append((section_title, section_text))

        return sections

    def extract(self, raw_page: RawPage) -> List[PageSection]:
        soup = BeautifulSoup(raw_page.html, "lxml")
        self._remove_noise(soup)

        page_title = self._extract_title(soup)
        extracted_sections = self._extract_sections(soup)

        page_sections: List[PageSection] = []

        for section_title, content in extracted_sections:
            content_type = classify_content_type(
                text=content,
                page_title=page_title,
                section_title=section_title,
                url=str(raw_page.url)
            )

            program_type = classify_program_type(
                text=content,
                page_title=page_title,
                section_title=section_title,
                url=str(raw_page.url)
            )

            doc_id = sha256_hash(f"{raw_page.url}|{section_title}|{content}")

            page_sections.append(
                PageSection(
                    doc_id=doc_id,
                    url=raw_page.url,
                    page_title=page_title,
                    section_title=section_title,
                    content=content,
                    content_type=content_type,
                    program_type=program_type,
                    last_scraped_at=datetime.now(timezone.utc)
                )
            )

        return page_sections
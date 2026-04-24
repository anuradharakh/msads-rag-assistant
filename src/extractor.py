from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, Tag

from src.config import settings
from src.models import PageSection, RawPage
from src.utils import (
    approx_token_count,
    canonicalize_url,
    clean_text,
    classify_content_type,
    classify_program_type,
    extract_bullets,
    extract_entities,
    extract_keywords,
    infer_course_type,
    infer_faq_category,
    infer_modality,
    infer_program_tags,
    make_summary,
    sha256_hash,
    _looks_like_person_name,
)

REMOVE_SELECTORS = [
    "nav", "footer", "script", "style", "noscript", "form", "svg", "aside",
    "[aria-hidden='true']", ".screen-reader-text", ".breadcrumb", ".breadcrumbs",
    ".share", ".social", ".menu", ".navbar", ".site-footer", ".site-header",
]

SPECIALIZED_EXTRACTORS = [
    ("faq", "_extract_faq_sections"),
    ("course-progressions", "_extract_course_sections"),
    ("instructors-staff", "_extract_faculty_sections"),
]

COURSE_TITLE_KEYWORDS = [
    "machine learning", "data engineering", "time series", "statistical models",
    "leadership and consulting", "capstone project", "generative ai", "nlp",
    "bayesian", "reinforcement learning", "mlops", "machine learning operations",
    "data visualization", "marketing analytics", "computer vision", "supply chain",
    "quantitative finance", "real time intelligent", "optimization and simulation",
    "healthcare", "causal models", "advanced machine learning", "applied generative",
    "next-gen nlp", "deep reinforcement", "r for data science", "python for data science",
    "introduction to statistical", "advanced linear algebra", "cloud computing",
    "big data", "data science capstone", "data science for",
]


class ContentExtractor:
    def _remove_noise(self, soup: BeautifulSoup) -> None:
        for selector in REMOVE_SELECTORS:
            for tag in soup.select(selector):
                tag.decompose()

        for tag in soup.find_all(class_=lambda value: value and any(
            junk in " ".join(value).lower()
            for junk in ["menu", "nav", "footer", "sidebar", "breadcrumb", "share", "social", "cookie"]
        )):
            tag.decompose()

    def _extract_title(self, soup: BeautifulSoup, raw_page: RawPage) -> str:
        h1 = soup.find("h1")
        if h1:
            text = clean_text(h1.get_text(" ", strip=True))
            if text:
                return text
        if raw_page.page_title:
            return clean_text(raw_page.page_title)
        if soup.title:
            return clean_text(soup.title.get_text(" ", strip=True))
        return "Untitled Page"

    def _routing_url(self, raw_page: RawPage) -> str:
        """
        CRITICAL:
        Route extraction by the originally requested URL FIRST.

        Why:
        UChicago can expose canonical/final URLs that collapse specialized pages
        like /faqs/ back to the parent program URL. If we route by final_url or
        canonical_url first, the FAQ extractor never runs and FAQ output is 0 bytes.
        """
        return canonicalize_url(str(raw_page.url))

    def _citation_url(self, raw_page: RawPage) -> str:
        """
        Preserve the originally requested page URL in each record.
        This keeps /faqs/, /course-progressions/, and /instructors-staff/
        visible in the JSONL output and avoids losing specialized-page identity.
        """
        return canonicalize_url(str(raw_page.url))

    def _canonical_url(self, raw_page: RawPage) -> str:
        return canonicalize_url(str(raw_page.canonical_url or raw_page.final_url or raw_page.url))

    def _heading_path_for(self, heading: Tag) -> List[str]:
        levels: Dict[int, str] = {}
        current_level = int(heading.name[1]) if heading.name and heading.name.startswith("h") else 7

        for prev in heading.find_all_previous(["h1", "h2", "h3", "h4", "h5", "h6"]):
            level = int(prev.name[1])
            text = clean_text(prev.get_text(" ", strip=True))
            if not text:
                continue
            if level < current_level and level not in levels:
                levels[level] = text

        levels[current_level] = clean_text(heading.get_text(" ", strip=True))
        return [levels[k] for k in sorted(levels)]

    def _extract_between_headings(self, heading: Tag) -> str:
        parts: List[str] = []
        current_level = int(heading.name[1]) if heading.name and heading.name.startswith("h") else 6

        for sib in heading.find_next_siblings():
            if isinstance(sib, Tag) and sib.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                sib_level = int(sib.name[1])
                if sib_level <= current_level:
                    break

            if isinstance(sib, Tag):
                text = clean_text(sib.get_text(" ", strip=True))
                if text:
                    parts.append(text)

        return clean_text(" ".join(parts))

    def _extract_sections(self, soup: BeautifulSoup) -> List[Tuple[str, str, dict]]:
        sections: List[Tuple[str, str, dict]] = []
        headings = soup.find_all(["h1", "h2", "h3", "h4"])

        if not headings:
            body_text = clean_text(soup.get_text(" ", strip=True))
            if len(body_text) >= settings.min_text_length:
                sections.append(("Main Content", body_text, {
                    "heading_path": ["Main Content"],
                    "heading_level": 1,
                }))
            return sections

        seen = set()
        for heading in headings:
            section_title = clean_text(heading.get_text(" ", strip=True))
            if not section_title or len(section_title) > 160:
                continue

            section_text = self._extract_between_headings(heading)
            if len(section_text) < settings.min_text_length:
                continue

            fingerprint = sha256_hash(f"{section_title}|{section_text}".lower())
            if fingerprint in seen:
                continue
            seen.add(fingerprint)

            sections.append((section_title, section_text, {
                "heading_path": self._heading_path_for(heading),
                "heading_level": int(heading.name[1]),
                "anchor_id": heading.get("id"),
            }))

        return sections

    def _extract_faq_sections(self, soup: BeautifulSoup) -> List[Tuple[str, str, dict]]:
        """
        Robust FAQ extraction.

        Supports:
        - <dt>/<dd> FAQ accordions
        - question headings followed by answer blocks
        - accordion buttons with aria-controls / expanded regions
        - generic question text with sibling answer content

        Also excludes CTA headings like "Ready to Apply?"
        """
        results: List[Tuple[str, str, dict]] = []
        current_category = "General"
        seen_questions = set()

        question_tags = soup.find_all(["h2", "h3", "h4", "h5", "dt", "button", "summary"])

        for tag in question_tags:
            tag_text = clean_text(tag.get_text(" ", strip=True))
            if not tag_text:
                continue

            # Category headings
            if tag.name in ("h2", "h3") and "?" not in tag_text and len(tag_text) < 90:
                current_category = tag_text
                continue

            if not self._looks_like_real_question(tag_text):
                continue

            question = tag_text
            q_norm = question.lower().strip()
            if q_norm in seen_questions:
                continue

            answer = self._extract_answer_for_question(tag)
            answer = clean_text(answer)

            if len(answer) < 25:
                continue

            if self._is_cta_question(question, answer):
                continue

            seen_questions.add(q_norm)

            category = infer_faq_category(question, answer) or current_category
            content = f"Question: {question}\nAnswer: {answer}"

            results.append((question[:140], content, {
                "faq_category": category,
                "question": question,
                "answer": answer,
                "heading_path": ["FAQs", category, question],
                "heading_level": 4,
                "retrieval_priority": 95,
            }))

        if not results:
            results = self._extract_faq_from_visible_text(soup)

        return _dedupe_tuple_records(results)

    def _extract_faq_from_visible_text(self, soup: BeautifulSoup) -> List[Tuple[str, str, dict]]:
        """
        Last-resort FAQ extractor for pages where the accordion DOM does not expose
        clean sibling relationships. It scans visible text for question lines and
        uses the following text span as the answer until the next question.
        """
        text = soup.get_text("\n", strip=True)
        lines = [clean_text(line) for line in text.split("\n")]
        lines = [line for line in lines if line and len(line) > 3]

        question_indexes = []
        for idx, line in enumerate(lines):
            if self._looks_like_real_question(line):
                question_indexes.append(idx)

        records: List[Tuple[str, str, dict]] = []
        for pos, start_idx in enumerate(question_indexes):
            end_idx = question_indexes[pos + 1] if pos + 1 < len(question_indexes) else min(len(lines), start_idx + 8)
            question = lines[start_idx]
            answer_lines = []
            for line in lines[start_idx + 1:end_idx]:
                if self._looks_like_real_question(line):
                    break
                if not self._is_noise_line(line):
                    answer_lines.append(line)

            answer = clean_text(" ".join(answer_lines))
            if len(answer) < 25:
                continue
            if self._is_cta_question(question, answer):
                continue

            category = infer_faq_category(question, answer)
            content = f"Question: {question}\nAnswer: {answer}"
            records.append((question[:140], content, {
                "faq_category": category,
                "question": question,
                "answer": answer,
                "heading_path": ["FAQs", category, question],
                "heading_level": 4,
                "retrieval_priority": 95,
            }))

        return records

    def _is_noise_line(self, line: str) -> bool:
        lower = line.lower()
        noise = [
            "master's in applied data science", "learn more", "read more",
            "start my app", "apply today", "schedule an appointment",
            "university of chicago", "data science institute",
        ]
        return any(n in lower for n in noise)

    def _looks_like_real_question(self, text: str) -> bool:
        text = clean_text(text)
        if "?" not in text:
            return False
        if len(text) < 18 or len(text) > 260:
            return False
        cta_phrases = [
            "ready to apply", "want more information", "have questions",
            "start your application", "learn more", "get in touch",
        ]
        return not any(p in text.lower() for p in cta_phrases)

    def _is_cta_question(self, question: str, answer: str) -> bool:
        blob = f"{question} {answer}".lower()
        cta_terms = [
            "start my app", "schedule an appointment", "get in touch",
            "submit your request", "download the pdf", "learn more",
        ]
        return any(t in blob for t in cta_terms) and len(answer) < 180

    def _extract_answer_for_question(self, tag: Tag) -> str:
        # 1) <dt> paired with <dd>
        if tag.name == "dt":
            dd = tag.find_next_sibling("dd")
            if dd:
                return clean_text(dd.get_text(" ", strip=True))

        # 2) <summary> paired with details body
        if tag.name == "summary" and tag.parent and tag.parent.name == "details":
            clone = BeautifulSoup(str(tag.parent), "lxml")
            summary = clone.find("summary")
            if summary:
                summary.decompose()
            return clean_text(clone.get_text(" ", strip=True))

        # 3) Button controls accordion content
        controls = tag.get("aria-controls")
        if controls:
            controlled = tag.find_parent().find(id=controls) if tag.find_parent() else None
            if controlled:
                return clean_text(controlled.get_text(" ", strip=True))

        # 4) Next sibling block or siblings until next question/category
        parts: List[str] = []
        for sib in tag.find_next_siblings():
            if isinstance(sib, Tag):
                sib_text = clean_text(sib.get_text(" ", strip=True))
                if not sib_text:
                    continue

                if sib.name in ("h2", "h3", "h4", "h5", "dt", "summary"):
                    break

                if sib.name == "button" and "?" in sib_text:
                    break

                parts.append(sib_text)

        # 5) If no direct siblings, try parent container minus the question text.
        if not parts and tag.parent:
            parent_clone = BeautifulSoup(str(tag.parent), "lxml")
            first_question = parent_clone.find(tag.name)
            if first_question:
                first_question.decompose()
            text = clean_text(parent_clone.get_text(" ", strip=True))
            if text and text.lower() != clean_text(tag.get_text(" ", strip=True)).lower():
                parts.append(text)

        return clean_text(" ".join(parts))

    def _extract_course_sections(self, soup: BeautifulSoup) -> List[Tuple[str, str, dict]]:
        results: List[Tuple[str, str, dict]] = []
        current_course_type = "elective"
        current_heading_path = ["Course Progressions"]
        seen_course_names = set()

        for tag in soup.find_all(["h2", "h3", "h4"]):
            heading_text = clean_text(tag.get_text(" ", strip=True))
            if not heading_text:
                continue

            lower = heading_text.lower()

            if _is_course_section_label(lower):
                current_course_type = infer_course_type(heading_text)
                current_heading_path = self._heading_path_for(tag)
                continue

            if not _is_course_title(lower):
                continue

            norm_name = lower.strip()
            if norm_name in seen_course_names:
                continue
            seen_course_names.add(norm_name)

            description = self._extract_between_headings(tag)
            if len(description) < 40:
                description = heading_text

            content = f"Course: {heading_text}\nCourse type: {current_course_type}\n\n{description}"
            results.append((heading_text, content, {
                "course_name": heading_text,
                "course_type": current_course_type,
                "course_names": [heading_text],
                "heading_path": current_heading_path + [heading_text],
                "heading_level": int(tag.name[1]),
                "anchor_id": tag.get("id"),
                "retrieval_priority": 90,
            }))

        return _dedupe_tuple_records(results)

    def _extract_faculty_sections(self, soup: BeautifulSoup) -> List[Tuple[str, str, dict]]:
        results: List[Tuple[str, str, dict]] = []
        cards: List[Tag] = []

        for sel in [
            "div.people-grid__item",
            "article",
            "div[class*='person']",
            "div[class*='faculty']",
            "div[class*='instructor']",
            "div[class*='team-member']",
        ]:
            cards = soup.select(sel)
            if cards:
                break

        for card in cards:
            parsed = self._parse_faculty_card(card)
            if parsed:
                results.append(parsed)

        if results:
            return _dedupe_tuple_records(results)

        for tag in soup.find_all(["h2", "h3", "h4"]):
            name = clean_text(tag.get_text(" ", strip=True))
            if not _looks_like_person_name(name):
                continue

            paras: List[str] = []
            for sib in tag.find_next_siblings():
                if isinstance(sib, Tag) and sib.name in ("h2", "h3", "h4"):
                    break
                if isinstance(sib, Tag):
                    t = clean_text(sib.get_text(" ", strip=True))
                    if t:
                        paras.append(t)

            if not paras:
                continue

            acad_title, industry_role = _split_role_line(paras[0])
            bio = clean_text(" ".join(paras[1:]))
            content = _build_faculty_content(name, acad_title, industry_role, bio)

            if len(content) < 60:
                continue

            results.append((name, content, {
                "faculty_name": name,
                "academic_title": acad_title,
                "industry_role": industry_role,
                "heading_path": ["Faculty, Instructors & Staff", name],
                "retrieval_priority": 80,
            }))

        return _dedupe_tuple_records(results)

    def _parse_faculty_card(self, card: Tag) -> Optional[Tuple[str, str, dict]]:
        name_tag = card.find(["h2", "h3", "h4", "strong"])
        name = clean_text(name_tag.get_text(" ", strip=True)) if name_tag else ""

        if not _looks_like_person_name(name):
            return None

        paras = [clean_text(p.get_text(" ", strip=True)) for p in card.find_all("p")]
        paras = [p for p in paras if p]
        role_line = paras[0] if paras else ""
        bio = clean_text(" ".join(paras[1:]))
        acad_title, industry_role = _split_role_line(role_line)
        content = _build_faculty_content(name, acad_title, industry_role, bio)

        if len(content) < 60:
            return None

        return (name, content, {
            "faculty_name": name,
            "academic_title": acad_title,
            "industry_role": industry_role,
            "heading_path": ["Faculty, Instructors & Staff", name],
            "retrieval_priority": 80,
        })

    def extract(self, raw_page: RawPage) -> List[PageSection]:
        soup = BeautifulSoup(raw_page.html, "lxml")
        self._remove_noise(soup)

        page_title = self._extract_title(soup, raw_page)
        route_url = self._routing_url(raw_page).lower()
        title_blob = f"{page_title} {raw_page.page_title or ''}".lower()

        extracted: List[Tuple[str, str, dict]] = []

        # Route specialized pages using requested/final URL + title, not canonical_url.
        if "faq" in route_url or "faq" in title_blob:
            extracted = self._extract_faq_sections(soup)
            print(f"[INFO] FAQ extractor routed: {route_url} -> {len(extracted)} FAQ records")
        elif "course-progressions" in route_url or "course progression" in title_blob:
            extracted = self._extract_course_sections(soup)
        elif "instructors-staff" in route_url or "faculty" in title_blob or "instructor" in title_blob:
            extracted = self._extract_faculty_sections(soup)

        if not extracted:
            # Opportunistic course extraction on pages that contain course headings.
            course_sections = self._extract_course_sections(soup)
            generic_sections = self._extract_sections(soup)
            extracted = course_sections + generic_sections if course_sections else generic_sections

        return [
            self._build_section(raw_page, page_title, title, content, dict(extras))
            for title, content, extras in extracted
        ]

    def _build_section(
        self,
        raw_page: RawPage,
        page_title: str,
        section_title: str,
        content: str,
        extras: dict,
    ) -> PageSection:
        route_url = self._citation_url(raw_page)
        canonical_url = self._canonical_url(raw_page)
        content_clean = clean_text(content)

        heading_path = extras.pop("heading_path", [page_title, section_title])

        # Pop fields that are also explicitly passed below. This prevents:
        # TypeError: PageSection() got multiple values for keyword argument ...
        extra_keywords = extras.pop("keywords", [])
        extra_entities = extras.pop("entities", [])
        extra_dates = extras.pop("dates", [])
        extra_emails = extras.pop("emails", [])
        extra_course_names = extras.pop("course_names", [])
        extra_tuition_mentions = extras.pop("tuition_mentions", [])

        content_type = classify_content_type(content_clean, page_title, section_title, route_url)

        # Hard override for true FAQ records because classifier priority may otherwise
        # mark visa/deadline/tuition FAQ answers as those types.
        if extras.get("question") and extras.get("answer"):
            content_type = "faq"

        if extras.get("course_name"):
            content_type = "course"

        if extras.get("faculty_name"):
            content_type = "faculty_bio"

        program_type = classify_program_type(content_clean, page_title, section_title, route_url)
        entities = extract_entities(content_clean)
        doc_id = sha256_hash(f"{route_url}|{section_title}|{content_clean}".lower())

        return PageSection(
            doc_id=doc_id,
            parent_doc_id=None,
            chunk_id=doc_id,
            url=route_url,
            canonical_url=canonical_url,
            page_title=page_title,
            section_title=section_title,
            heading_path=heading_path,
            content=content_clean,
            content_clean=content_clean,
            content_summary=make_summary(content_clean),
            bullet_points=extract_bullets(content),
            token_count=approx_token_count(content_clean),
            char_count=len(content_clean),
            content_type=content_type,
            program_type=program_type,
            program_tags=infer_program_tags(content_clean, route_url),
            modality=infer_modality(content_clean, route_url),
            keywords=list(dict.fromkeys(extra_keywords + extract_keywords(f"{page_title} {section_title} {content_clean}"))),
            entities=list(dict.fromkeys(extra_entities + entities["entities"])),
            dates=list(dict.fromkeys(extra_dates + entities["dates"])),
            emails=list(dict.fromkeys(extra_emails + entities["emails"])),
            course_names=list(dict.fromkeys(extra_course_names + entities["course_names"])),
            tuition_mentions=list(dict.fromkeys(extra_tuition_mentions + entities["tuition_mentions"])),
            last_scraped_at=datetime.now(timezone.utc),
            page_last_modified=raw_page.last_modified,
            **extras,
        )


def _dedupe_tuple_records(records: List[Tuple[str, str, dict]]) -> List[Tuple[str, str, dict]]:
    seen = set()
    out = []

    for title, content, extras in records:
        fp = sha256_hash(f"{title}|{content}".lower())
        if fp in seen:
            continue
        seen.add(fp)
        out.append((title, content, extras))

    return out


def _is_course_section_label(lower_text: str) -> bool:
    labels = [
        "core course", "elective course", "sample elective", "capstone",
        "foundational", "noncredit", "career seminar", "independent study",
    ]
    return any(lbl in lower_text for lbl in labels) and len(lower_text) < 80


def _is_course_title(lower_text: str) -> bool:
    if len(lower_text) < 8 or len(lower_text) > 140 or lower_text.endswith("?"):
        return False

    blocked = [
        "application", "deadline", "tuition", "financial aid", "program overview",
        "start your application", "schedule", "faculty", "staff",
    ]
    if any(b in lower_text for b in blocked):
        return False

    return any(kw in lower_text for kw in COURSE_TITLE_KEYWORDS)


def _split_role_line(line: str) -> Tuple[str, str]:
    line = clean_text(line)

    if ";" in line:
        left, right = [p.strip() for p in line.split(";", 1)]
        return left, right

    acad_keywords = ["professor", "instructor", "lecturer", "clinical", "adjunct", "senior"]
    if any(kw in line.lower() for kw in acad_keywords):
        return line, ""

    return "", line


def _build_faculty_content(name: str, acad_title: str, industry_role: str, bio: str) -> str:
    parts = [f"Faculty: {name}"]

    if acad_title:
        parts.append(f"Academic role: {acad_title}")

    if industry_role:
        parts.append(f"Industry role: {industry_role}")

    if bio:
        parts.append(bio)

    return "\n".join(parts)

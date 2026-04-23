import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import urljoin, urlparse, urldefrag

from src.models import ContentType, ProgramType


def ensure_dirs(paths: Iterable[str]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = normalize_whitespace(text)

    # remove UI junk
    junk_phrases = ["Learn More", "Click here", "Read more"]
    for phrase in junk_phrases:
        text = text.replace(phrase, "")

    return text.strip()


def normalize_url(base_url: str, href: str) -> Optional[str]:
    if not href:
        return None

    absolute = urljoin(base_url, href)
    absolute, _ = urldefrag(absolute)

    parsed = urlparse(absolute)
    if parsed.scheme not in {"http", "https"}:
        return None

    return absolute.rstrip("/")


def is_allowed_url(url: str, allowed_domains: List[str], allowed_path_keywords: List[str]) -> bool:
    parsed = urlparse(url)

    if parsed.netloc not in allowed_domains:
        return False

    path = parsed.path.lower()
    return any(keyword in path for keyword in allowed_path_keywords)


def sha256_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def classify_content_type(text: str, page_title: str, section_title: str, url: str):
    title_blob = f"{page_title} {section_title} {url}".lower()
    text_blob = text[:2000].lower()

    if "capstone" in section_title.lower():
        return "capstone"

    if (
        "capstone" in title_blob and
        any(k in text_blob for k in ["project sponsors", "research-focused team", "teams comprised of four students", "real business problem"])
    ):
        return "capstone"

    if any(k in title_blob + " " + text_blob for k in ["online", "in-person", "full-time", "part-time", "working professionals"]):
        return "program_format"

    if any(k in title_blob + " " + text_blob for k in ["admission", "apply", "application", "deadline", "recommendation", "transcript"]):
        return "admissions"

    if any(k in title_blob + " " + text_blob for k in ["curriculum", "course", "elective", "core courses"]):
        return "curriculum"

    if any(k in title_blob + " " + text_blob for k in ["faculty", "instructor", "professor"]):
        return "faculty"

    if any(k in title_blob + " " + text_blob for k in ["career", "outcome", "employment"]):
        return "career"

    return "unknown"


def classify_program_type(text: str, page_title: str, section_title: str, url: str) -> ProgramType:
    blob = f"{page_title} {section_title} {url} {text[:1000]}".lower()

    has_online = "online" in blob
    has_in_person = "in-person" in blob or "in person" in blob

    if has_online and not has_in_person:
        return "online"
    if has_in_person and not has_online:
        return "in_person"
    if has_online and has_in_person:
        return "general"

    return "general"


def write_jsonl(records: List[dict], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
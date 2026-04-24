from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse, urldefrag, unquote

from src.models import ContentType, ProgramType

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "you", "your", "will",
    "have", "has", "into", "our", "their", "students", "program", "master", "masters",
    "data", "science", "applied", "university", "chicago", "course", "courses",
}

UI_JUNK_PATTERNS = [
    r"\bLearn More\b", r"\bRead More\b", r"\bClick Here\b", r"\bApply today\b",
    r"\bLoading…\b", r"\bShare this\b", r"\bBack to top\b",
]


def ensure_dirs(paths: Iterable[str]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def clean_text(text: str) -> str:
    text = (text or "").replace("\xa0", " ").replace("\u200b", "")
    text = normalize_whitespace(text)
    for pattern in UI_JUNK_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return normalize_whitespace(text)


def normalize_url(base_url: str, href: str) -> Optional[str]:
    if not href:
        return None
    absolute = urljoin(base_url, href.strip())
    absolute, _ = urldefrag(absolute)
    parsed = urlparse(absolute)
    if parsed.scheme not in {"http", "https"}:
        return None
    # Drop tracking params and decode accidental %20 path artifacts.
    kept_query = [(k, v) for k, v in parse_qsl(parsed.query) if not k.lower().startswith("utm_")]
    clean_path = re.sub(r"/%20/?$", "/", unquote(parsed.path)).rstrip("/")
    normalized = parsed._replace(path=clean_path, query=urlencode(kept_query), fragment="")
    return urlunparse(normalized).rstrip("/")


def canonicalize_url(url: str) -> str:
    return normalize_url(url, url) or url.rstrip("/")


def is_allowed_url(url: str, allowed_domains: List[str], allowed_path_keywords: List[str]) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower().replace("www.", "")
    allowed = {d.lower().replace("www.", "") for d in allowed_domains}
    if host not in allowed:
        return False
    path = parsed.path.lower()
    return any(keyword.lower() in path for keyword in allowed_path_keywords)


def sha256_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def approx_token_count(text: str) -> int:
    # Conservative approximation without adding a tokenizer dependency.
    return max(1, int(len(re.findall(r"\w+|[^\w\s]", text or "")) / 0.75))


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", clean_text(text))
    return [p.strip() for p in parts if p.strip()]


def make_summary(text: str, max_chars: int = 420) -> str:
    sentences = split_sentences(text)
    summary = " ".join(sentences[:2]) if sentences else clean_text(text)
    return summary[:max_chars].rstrip()


def extract_bullets(text: str) -> List[str]:
    bullets = []
    for line in re.split(r"[\n\r]+", text or ""):
        line = line.strip(" -•\t")
        if 8 <= len(line) <= 220:
            bullets.append(clean_text(line))
    return list(dict.fromkeys(bullets))[:12]


def extract_keywords(text: str, limit: int = 18) -> List[str]:
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text or "")]
    freq = {}
    for w in words:
        if w in STOPWORDS or len(w) < 4:
            continue
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq, key=lambda w: (-freq[w], w))
    return ranked[:limit]


def extract_entities(text: str) -> dict:
    text = text or ""
    emails = re.findall(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+", text)
    dates = re.findall(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
        r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+\d{4}\b|"
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\bAutumn\s+\d{4}\b",
        text,
        flags=re.IGNORECASE,
    )
    tuition = re.findall(r"\$\s?\d[\d,]*(?:\.\d+)?", text)
    course_names = re.findall(
        r"\b(?:Machine Learning I{0,2}|Time Series Analysis and Forecasting|Statistical Models for Data Science|"
        r"Data Engineering Platforms for Analytics|Big Data and Cloud Computing|Leadership and Consulting for Data Science|"
        r"Data Science Capstone Project|Python for Data Science|R for Data Science|Advanced Linear Algebra for Machine Learning|"
        r"Generative AI[:\w\s\-]*|Machine Learning Operations|Data Visualization Techniques)\b",
        text,
    )
    titlecase_entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}\b", text)
    return {
        "emails": list(dict.fromkeys(emails)),
        "dates": list(dict.fromkeys(dates)),
        "tuition_mentions": list(dict.fromkeys(tuition)),
        "course_names": list(dict.fromkeys(course_names)),
        "entities": list(dict.fromkeys(titlecase_entities))[:25],
    }


def infer_modality(text: str, url: str = "") -> List[str]:
    blob = f"{url} {text}".lower()
    out = []
    if "online" in blob:
        out.append("online")
    if "in-person" in blob or "in person" in blob or "nbc tower" in blob or "gleacher" in blob:
        out.append("in_person")
    if "full-time" in blob or "full time" in blob:
        out.append("full_time")
    if "part-time" in blob or "part time" in blob:
        out.append("part_time")
    return list(dict.fromkeys(out))


def infer_program_tags(text: str, url: str = "") -> List[str]:
    blob = f"{url} {text}".lower()
    tags = infer_modality(text, url)
    if "mba" in blob or "booth" in blob or "joint degree" in blob:
        tags.append("mba_ms")
    if "thesis" in blob or "18-course" in blob or "18 course" in blob or "2-year" in blob:
        tags.append("thesis_track")
    if "stem" in blob or "opt" in blob:
        tags.append("stem_opt")
    if "visa" in blob or "international" in blob:
        tags.append("international")
    return list(dict.fromkeys(tags))


def classify_content_type(text: str, page_title: str, section_title: str, url: str) -> ContentType:
    title_blob = f"{page_title} {section_title} {url}".lower()
    text_blob = (text or "")[:2500].lower()
    combined = f"{title_blob} {text_blob}"

    # Only true FAQ records should be labeled faq. Avoid CTA headings such as
    # "Ready to Apply?" on non-FAQ pages polluting the FAQ export.
    if "/faqs" in url.lower() or text_blob.startswith("question:") or text_blob.startswith("q:"):
        return "faq"
    if "tuition" in combined or "financial aid" in combined or "fees" in combined or re.search(r"\$\s?\d", combined):
        return "tuition"
    if "deadline" in combined or "application round" in combined or "apply by" in combined:
        return "deadline"
    if "stem" in combined and "opt" in combined:
        return "stem_opt"
    if any(k in combined for k in ["visa", "toefl", "ielts", "cpt", "international student"]):
        return "visa"
    if "capstone" in combined:
        return "capstone"
    if "course:" in text_blob or "course-progressions" in url.lower() or any(k in section_title.lower() for k in ["machine learning", "time series", "statistical models", "data engineering", "generative ai", "nlp", "visualization", "capstone project", "linear algebra", "python for data science", "r for data science"]):
        if any(k in combined for k in ["machine learning", "time series", "statistical models", "data engineering", "generative ai", "nlp", "visualization", "capstone project", "linear algebra", "python for data science", "r for data science"]):
            return "course"
        return "curriculum"
    if ("instructors-staff" in url.lower() or text_blob.startswith("faculty:")) and _looks_like_person_name(section_title):
        return "faculty_bio"
    if any(k in combined for k in ["admission", "apply", "application", "recommendation", "transcript", "gre", "gmat"]):
        return "admissions"
    if any(k in combined for k in ["curriculum", "core courses", "elective", "foundational", "course progression"]):
        return "curriculum"
    if any(k in combined for k in ["faculty", "instructor", "professor"]):
        return "faculty"
    if any(k in combined for k in ["career", "employment", "outcome", "internship", "employer"]):
        return "career"
    if any(k in combined for k in ["online", "in-person", "full-time", "part-time", "program format"]):
        return "program_format"
    if any(k in combined for k in ["mba", "booth", "joint degree"]):
        return "mba_ms"
    return "overview" if len(text_blob) > 250 else "unknown"


def classify_program_type(text: str, page_title: str, section_title: str, url: str) -> ProgramType:
    blob = f"{page_title} {section_title} {url} {(text or '')[:1200]}".lower()
    if "mba" in blob or "booth" in blob or "joint degree" in blob:
        return "mba_ms"
    if "thesis" in blob or "18-course" in blob or "18 course" in blob or "2-year" in blob:
        return "thesis_track"
    has_online = "online" in blob
    has_in_person = "in-person" in blob or "in person" in blob or "nbc tower" in blob
    if has_online and not has_in_person:
        return "online"
    if has_in_person and not has_online:
        return "in_person"
    return "general"

FAQ_CATEGORY_KEYWORDS = {
    "Application Process": ["admission", "apply", "recommend", "transcript", "gre", "gmat", "deadline"],
    "International Students": ["visa", "toefl", "ielts", "international", "opt", "stem", "cpt"],
    "Online Program": ["online program", "synchronous", "asynchronous", "immersion weekend"],
    "In-Person Program": ["in-person program", "nbc tower", "gleacher"],
    "MBA/MS Program": ["mba", "booth", "joint degree"],
    "2-Year Thesis Track": ["2-year", "thesis track", "18 course", "18-course", "thesis"],
    "Curriculum": ["course", "elective", "core", "curriculum", "capstone"],
    "Tuition & Financial Aid": ["tuition", "fees", "scholarship", "financial aid", "cost"],
    "Career": ["career", "job", "employer", "internship", "placement"],
}

def infer_faq_category(question: str, answer: str) -> str:
    blob = f"{question} {answer}".lower()
    for category, keywords in FAQ_CATEGORY_KEYWORDS.items():
        if any(kw in blob for kw in keywords):
            return category
    return "General"

COURSE_TYPE_KEYWORDS = {
    "career_seminar": ["career seminar"],
    "independent_study": ["independent study"],
    "foundational": ["foundational", "noncredit", "pre-quarter"],
    "capstone": ["capstone"],
    "core": ["core course", "required course"],
    "elective": ["elective", "sample elective"],
}

def infer_course_type(context_heading: str) -> str:
    lower = (context_heading or "").lower()
    for ctype, keywords in COURSE_TYPE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return ctype
    return "elective"


def _looks_like_person_name(text: str) -> bool:
    text = clean_text(text)
    if len(text) > 80 or len(text) < 4 or "?" in text:
        return False
    words = text.split()
    if len(words) < 2 or len(words) > 6:
        return False
    lower = text.lower()
    non_name_tokens = ["course", "program", "curriculum", "apply", "tuition", "capstone", "career", "overview", "faculty", "instructor", "staff", "online", "in-person", "mba"]
    return text[0].isupper() and not any(tok in lower for tok in non_name_tokens)


def write_jsonl(records: List[dict], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

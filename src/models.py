from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


ContentType = Literal[
    "overview",
    "admissions",
    "deadline",
    "tuition",
    "curriculum",
    "course",
    "faculty",
    "faculty_bio",
    "faq",
    "capstone",
    "career",
    "program_format",
    "visa",
    "stem_opt",
    "mba_ms",
    "contact",
    "unknown",
]

ProgramType = Literal[
    "general",
    "online",
    "in_person",
    "mba_ms",
    "thesis_track",
    "unknown",
]

CourseType = Literal[
    "core",
    "elective",
    "capstone",
    "foundational",
    "career_seminar",
    "independent_study",
    "unknown",
]


class RawPage(BaseModel):
    url: HttpUrl
    final_url: Optional[HttpUrl] = None
    html: str
    fetched_at: datetime
    status_code: int
    page_title: Optional[str] = None
    canonical_url: Optional[str] = None
    meta_description: Optional[str] = None
    last_modified: Optional[str] = None
    etag: Optional[str] = None


class PageSection(BaseModel):
    # Stable identifiers
    doc_id: str
    parent_doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    chunk_index: int = 0
    chunk_count: int = 1

    # Source/citation metadata
    url: HttpUrl
    canonical_url: Optional[str] = None
    source_domain: str = "datascience.uchicago.edu"
    source: str = "official_uchicago_site"
    page_title: str
    section_title: str
    heading_path: List[str] = Field(default_factory=list)
    heading_level: Optional[int] = None
    anchor_id: Optional[str] = None

    # Text fields
    content: str
    content_clean: Optional[str] = None
    content_summary: Optional[str] = None
    bullet_points: List[str] = Field(default_factory=list)
    token_count: int = 0
    char_count: int = 0

    # Labels
    content_type: ContentType
    program_type: ProgramType
    program_tags: List[str] = Field(default_factory=list)
    modality: List[str] = Field(default_factory=list)
    degree_track: Optional[str] = None
    audience_type: Optional[str] = None

    # FAQ fields
    faq_category: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None

    # Course fields
    course_name: Optional[str] = None
    course_type: Optional[CourseType] = None

    # Faculty fields
    faculty_name: Optional[str] = None
    academic_title: Optional[str] = None
    industry_role: Optional[str] = None

    # Structured retrieval features
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    dates: List[str] = Field(default_factory=list)
    emails: List[str] = Field(default_factory=list)
    course_names: List[str] = Field(default_factory=list)
    tuition_mentions: List[str] = Field(default_factory=list)

    # Quality/governance metadata
    retrieval_priority: int = 50
    is_chunk: bool = False
    duplicate_of: Optional[str] = None
    last_scraped_at: datetime
    page_last_modified: Optional[str] = None

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Content cannot be empty")
        return value

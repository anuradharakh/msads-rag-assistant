from pydantic import BaseModel, HttpUrl, field_validator
from typing import Optional, Literal
from datetime import datetime


ContentType = Literal[
    "overview",
    "admissions",
    "curriculum",
    "faculty",
    "faq",
    "capstone",
    "career",
    "program_format",
    "unknown"
]

ProgramType = Literal[
    "general",
    "online",
    "in_person",
    "unknown"
]


class PageSection(BaseModel):
    doc_id: str
    url: HttpUrl
    page_title: str
    section_title: str
    content: str
    content_type: ContentType
    program_type: ProgramType
    source: str = "official_uchicago_site"
    last_scraped_at: datetime

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Content cannot be empty")
        return value


class RawPage(BaseModel):
    url: HttpUrl
    html: str
    fetched_at: datetime
    status_code: int
    page_title: Optional[str] = None
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    app_name: str = "msads-rag-ingestion"

    seed_urls: List[str] = Field(
        default_factory=lambda: [
            "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/"
        ]
    )

    allowed_domains: List[str] = Field(
        default_factory=lambda: [
            "datascience.uchicago.edu"
        ]
    )

    allowed_path_keywords: List[str] = Field(
        default_factory=lambda: [
            "ms-in-applied-data-science",
            "masters-programs",
            "faculty",
            "admissions",
            "capstone",
            "faq",
            "online",
            "in-person",
            "curriculum",
            "courses",
            "apply"
        ]
    )

    request_timeout_seconds: int = 20
    max_pages: int = 100
    user_agent: str = (
        "Mozilla/5.0 (compatible; MSADS-RAG-Bot/1.0; +https://example.com/bot)"
    )

    output_dir: str = "data/processed"
    raw_html_dir: str = "data/raw"
    log_dir: str = "data/logs"

    save_raw_html: bool = True
    min_text_length: int = 120
    max_retries: int = 3
    verify_ssl: bool = True


settings = Settings()
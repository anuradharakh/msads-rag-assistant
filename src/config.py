from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from dotenv import load_dotenv
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    app_name: str = "msads-rag-ingestion"

    seed_urls: List[str] = Field(
        default_factory=lambda: [
            "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/",
            "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/faqs/",
            "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/course-progressions/",
            "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/instructors-staff/"
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
            "faqs",
            "online",
            "in-person",
            "curriculum",
            "courses",
            "course-progressions",   
            "instructors-staff",     
            "apply",
            "how-to-apply",
            "tuition",
            "our-students",
            "career-outcomes",
            "international-students"
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

    save_raw_html: bool = False
    min_text_length: int = 120
    max_retries: int = 3
    verify_ssl: bool = True

    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_batch_size: int = 32
    chroma_dir: str = "data/vector_store/chroma"
    chroma_collection_name: str = "msads_knowledge_base"
    processed_data_path: str = "data/processed/msads_processed_data.jsonl"

    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:7b"
    rag_top_k: int = 5
    rag_min_score: float = 0.35

    # RAG / Generation
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:7b"
    rag_top_k: int = 5
    rag_min_score: float = 0.35

    # Reranking
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_initial_top_k: int = 20
    rerank_final_top_k: int = 5
    use_reranker: bool = True

settings = Settings()
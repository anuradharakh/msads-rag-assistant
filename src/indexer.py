from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.config import settings
from src.embeddings import EmbeddingClient
from src.utils import clean_text, ensure_dirs
from src.vector_store import ChromaVectorStore


ALLOWED_METADATA_TYPES = (str, int, float, bool)


class VectorIndexer:
    def __init__(self) -> None:
        self.embedder = EmbeddingClient()
        self.store = ChromaVectorStore()

    def _load_records(self, path: str) -> List[Dict[str, Any]]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Processed data file not found: {path}. Run Phase 1 first: python -m src.main"
            )

        records: List[Dict[str, Any]] = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        return records

    def _make_document_text(self, record: Dict[str, Any]) -> str:
        """
        Text sent to embedding model.

        We include titles and content type because this improves retrieval context.
        """
        parts = [
            f"Page title: {record.get('page_title', '')}",
            f"Section title: {record.get('section_title', '')}",
            f"Content type: {record.get('content_type', '')}",
            f"Program type: {record.get('program_type', '')}",
            "",
            record.get("content_clean") or record.get("content") or "",
        ]

        return clean_text("\n".join(parts))

    def _clean_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Chroma metadata values must be simple scalar types.
        Lists are converted to comma-separated strings.
        """
        keep_keys = [
            "url",
            "canonical_url",
            "source",
            "page_title",
            "section_title",
            "content_type",
            "program_type",
            "chunk_index",
            "chunk_count",
            "is_chunk",
            "last_scraped_at",
            "page_last_modified",
            "retrieval_priority",
            "faq_category",
            "question",
            "course_name",
            "course_type",
            "faculty_name",
            "academic_title",
            "industry_role",
            "token_count",
            "char_count",
        ]

        metadata: Dict[str, Any] = {}

        for key in keep_keys:
            value = record.get(key)

            if value is None:
                continue

            if isinstance(value, ALLOWED_METADATA_TYPES):
                metadata[key] = value
            elif isinstance(value, list):
                metadata[key] = ", ".join(str(v) for v in value)

        # Useful list fields flattened for filtering/debugging
        for key in ["program_tags", "modality", "keywords", "dates", "emails", "course_names"]:
            value = record.get(key)
            if isinstance(value, list) and value:
                metadata[key] = ", ".join(str(v) for v in value)

        return metadata

    def _batch(
        self,
        records: List[Dict[str, Any]],
        batch_size: int,
    ) -> Iterable[List[Dict[str, Any]]]:
        for i in range(0, len(records), batch_size):
            yield records[i:i + batch_size]

    def build_index(self, input_path: str | None = None) -> None:
        ensure_dirs([settings.chroma_dir])

        path = input_path or settings.processed_data_path
        records = self._load_records(path)

        print(f"[INFO] Loaded processed records: {len(records)}")
        indexed_count = 0

        for batch in self._batch(records, settings.embedding_batch_size):
            ids: List[str] = []
            documents: List[str] = []
            metadatas: List[Dict[str, Any]] = []

            for record in batch:
                doc_id = record.get("chunk_id") or record.get("doc_id")
                document = self._make_document_text(record)

                if not doc_id:
                    continue

                if len(document) < 40:
                    continue

                ids.append(str(doc_id))
                documents.append(document)
                metadatas.append(self._clean_metadata(record))

            if not documents:
                continue

            embeddings = self.embedder.embed_batch(documents)

            self.store.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            indexed_count += len(ids)
            print(f"[INFO] Indexed records so far: {indexed_count}")

        print(f"[INFO] Final Chroma collection count: {self.store.count()}")
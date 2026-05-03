# MSADS RAG Assistant

A production-oriented Retrieval-Augmented Generation (RAG) chatbot for answering questions about the University of Chicago MS in Applied Data Science (MSADS) program using grounded, structured data from official sources.


## Overview

MSADS RAG Assistant is a GenAI chatbot system designed to:

- Retrieve accurate information from official program data
- Avoid hallucinations by grounding responses in real content
- Provide explainable answers with source traceability

This project is built as a **production-grade RAG pipeline**, not just a prototype.


## System Architecture

```text
Official MSADS Website
        ↓
Phase 1: Ingestion (crawl + extract + clean)
        ↓
Structured Sections + Metadata
        ↓
Phase 2: Chunking + Embeddings
        ↓
Chroma Vector Database
        ↓
Phase 3: Semantic Retrieval (Top-K)
        ↓
Phase 4: Answer Generation (LLM)

```

## Project Structure
```
src/
├── ingestion/
│   ├── crawler.py
│   ├── extractor.py
│   └── pipeline.py
├── processing/
│   └── chunking.py
├── retrieval/
│   ├── embeddings.py
│   ├── indexer.py
│   ├── retriever.py
│   └── vector_store.py
├── scripts/
│   ├── check_index.py
│   ├── query_index.py
│   ├── retrieval_baseline.py
│   └── retrieval_eval.py
├── config.py
├── models.py
├── main.py
└── utils.py
```

## Tech Stack
- Python
- BeautifulSoup
- Pydantic
- Sentence Transformers
- BAAI/bge-base-en-v1.5
- ChromaDB
- Local vector search


## Data Model

Each knowledge record includes:

- `doc_id`
- `chunk_id`
- `url`
- `canonical_url`
- `page_title`
- `section_title`
- `content`
- `content_type`
- `program_type`
- `token_count`
- `keywords`
- `entities`
- `last_scraped_at`

## SetUp
```
python -m src.main
python -m src.build_index
python -m src.scripts.check_index
python -m src.scripts.retrieval_baseline
python -m src.scripts.rag_smoke_test
python -m src.scripts.chat_cli
python -m src.scripts.evaluate_rag
streamlit run app.py
```


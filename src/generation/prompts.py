from __future__ import annotations

from typing import Dict, List


SYSTEM_PROMPT = """
You are MSADS RAG Assistant, a helpful AI assistant for the University of Chicago
MS in Applied Data Science program.

Rules:
1. Answer only using the provided context.
2. Do not invent deadlines, requirements, tuition, policies, or program details.
3. If the context does not contain the answer, say:
   "I don't have enough information in the indexed MSADS sources to answer that."
4. Cite sources using [Source 1], [Source 2], etc.
5. Be concise, accurate, and helpful.
6. For admissions, visa, tuition, deadlines, and policy questions, be extra careful.
""".strip()


def build_context(retrieved_chunks: List[Dict]) -> str:
    blocks = []

    for idx, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadata", {})
        document = chunk.get("document", "")

        blocks.append(
            f"""
[Source {idx}]
URL: {metadata.get("url", "Unknown URL")}
Page title: {metadata.get("page_title", "Unknown page")}
Section title: {metadata.get("section_title", "Unknown section")}
Content type: {metadata.get("content_type", "unknown")}

{document}
""".strip()
        )

    return "\n\n".join(blocks)


def build_rag_prompt(question: str, retrieved_chunks: List[Dict]) -> str:
    context = build_context(retrieved_chunks)

    return f"""
Use the context below to answer the user question.

CONTEXT:
{context}

USER QUESTION:
{question}

Answer requirements:
- Give a direct answer first.
- Use citations like [Source 1].
- Do not use outside knowledge.
- If the answer is not clearly present in the context, say you do not have enough information.
- Do not mention internal retrieval, embeddings, vector databases, or implementation details.

FINAL ANSWER:
""".strip()
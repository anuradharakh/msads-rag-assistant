from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.retriever import Retriever


@dataclass
class EvalCase:
    query: str
    expected_content_type: Optional[str] = None
    expected_keywords: Optional[List[str]] = None


EVAL_CASES = [
    EvalCase(
        query="Is GRE required for admission?",
        expected_content_type="faq",
        expected_keywords=["gre", "gmat", "not required"],
    ),
    EvalCase(
        query="What is the capstone project?",
        expected_content_type="capstone",
        expected_keywords=["capstone", "project", "real"],
    ),
    EvalCase(
        query="Is the online program synchronous or asynchronous?",
        expected_keywords=["synchronous", "asynchronous", "online"],
    ),
    EvalCase(
        query="What are the application deadlines?",
        expected_keywords=["deadline", "June 23, 2026", "Autumn 2026"],
    ),
    EvalCase(
        query="Is the in-person program STEM OPT eligible?",
        expected_keywords=["STEM", "OPT", "in-person"],
    ),
    EvalCase(
        query="How many courses are required?",
        expected_keywords=["12", "courses", "core", "elective"],
    ),
    EvalCase(
        query="What is the 2-year thesis track?",
        expected_keywords=["2-year", "thesis", "18"],
    ),
]


def contains_expected_keyword(text: str, keywords: List[str]) -> bool:
    lower = text.lower()
    return any(keyword.lower() in lower for keyword in keywords)


def evaluate_case(retriever: Retriever, case: EvalCase, top_k: int = 5) -> dict:
    results = retriever.search(case.query, n_results=top_k)

    combined_text = " ".join(
        [
            result.get("document", "")
            + " "
            + " ".join(str(v) for v in result.get("metadata", {}).values())
            for result in results
        ]
    )

    content_type_hit = True
    if case.expected_content_type:
        content_type_hit = any(
            result.get("metadata", {}).get("content_type") == case.expected_content_type
            for result in results
        )

    keyword_hit = True
    if case.expected_keywords:
        keyword_hit = contains_expected_keyword(combined_text, case.expected_keywords)

    success = bool(results) and content_type_hit and keyword_hit

    return {
        "query": case.query,
        "success": success,
        "content_type_hit": content_type_hit,
        "keyword_hit": keyword_hit,
        "top_result_title": results[0]["metadata"].get("section_title") if results else None,
        "top_result_type": results[0]["metadata"].get("content_type") if results else None,
        "top_result_url": results[0]["metadata"].get("url") if results else None,
    }


def main() -> None:
    retriever = Retriever()

    passed = 0
    total = len(EVAL_CASES)

    print("\nRetrieval Baseline Evaluation")
    print("=" * 100)

    for case in EVAL_CASES:
        result = evaluate_case(retriever, case)

        if result["success"]:
            passed += 1

        print("\nQuery:", result["query"])
        print("Success:", result["success"])
        print("Content type hit:", result["content_type_hit"])
        print("Keyword hit:", result["keyword_hit"])
        print("Top title:", result["top_result_title"])
        print("Top type:", result["top_result_type"])
        print("Top URL:", result["top_result_url"])

    print("\n" + "=" * 100)
    print(f"Passed: {passed}/{total}")
    print(f"Baseline retrieval accuracy: {passed / total:.2%}")


if __name__ == "__main__":
    main()
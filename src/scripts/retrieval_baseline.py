from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.retriever import Retriever


DEFAULT_QUERIES = [
    "What is the capstone project?",
    "Is GRE required for admission?",
    "What are the application deadlines?",
    "Is the online program synchronous or asynchronous?",
    "How many courses are required?",
    "Is the in-person program STEM OPT eligible?",
    "What is the 2-year thesis track?",
    "Who teaches machine learning?",
]


def similarity_from_distance(distance: Any) -> Optional[float]:
    if isinstance(distance, (int, float)):
        return round(1 - distance, 4)
    return None


def print_result(result: Dict[str, Any], rank: int) -> None:
    metadata = result.get("metadata", {})
    document = result.get("document", "")
    distance = result.get("distance")

    print("\n" + "-" * 100)
    print(f"Rank          : {rank}")
    print(f"Similarity    : {similarity_from_distance(distance)}")
    print(f"Distance      : {distance}")
    print(f"Content Type  : {metadata.get('content_type')}")
    print(f"Program Type  : {metadata.get('program_type')}")
    print(f"Page Title    : {metadata.get('page_title')}")
    print(f"Section Title : {metadata.get('section_title')}")
    print(f"URL           : {metadata.get('url')}")
    print("-" * 100)
    print(document[:1400])


def run_query(
    query: str,
    top_k: int = 5,
    content_type: Optional[str] = None,
    program_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    retriever = Retriever()

    results = retriever.search(
        query=query,
        n_results=top_k,
        content_type=content_type,
        program_type=program_type,
    )

    print("\n" + "=" * 100)
    print(f"QUERY: {query}")
    print(f"TOP_K: {top_k}")
    if content_type:
        print(f"FILTER content_type={content_type}")
    if program_type:
        print(f"FILTER program_type={program_type}")
    print("=" * 100)

    if not results:
        print("No results found. Check index with: python -m src.scripts.check_index")
        return []

    for idx, result in enumerate(results, start=1):
        print_result(result, idx)

    return results


def main() -> None:
    for query in DEFAULT_QUERIES:
        run_query(query=query, top_k=5)

    run_query(
        query="Is GRE required?",
        top_k=5,
        content_type="faq",
    )


if __name__ == "__main__":
    main()
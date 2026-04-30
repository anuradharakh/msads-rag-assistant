from __future__ import annotations

from src.scripts.retrieval_baseline import run_query


def main() -> None:
    print("MSADS Semantic Retrieval Baseline")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Ask a question: ").strip()

        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        if not query:
            continue

        content_type = input("Optional content_type filter, or Enter: ").strip() or None
        program_type = input("Optional program_type filter, or Enter: ").strip() or None

        run_query(
            query=query,
            top_k=5,
            content_type=content_type,
            program_type=program_type,
        )


if __name__ == "__main__":
    main()
from src.retrieval.retriever import Retriever


def print_results(query: str, results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print(f"QUERY: {query}")
    print("=" * 100)

    for idx, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print(f"\nResult {idx}")
        print(f"Score: {result.get('score')}")
        print(f"Distance: {result.get('distance')}")
        print(f"Title: {metadata.get('section_title')}")
        print(f"Type: {metadata.get('content_type')}")
        print(f"Program: {metadata.get('program_type')}")
        print(f"URL: {metadata.get('url')}")
        print("-" * 80)
        print(result["document"][:700])


def main() -> None:
    retriever = Retriever()

    test_queries = [
        "What is the capstone project?",
        "Is the online program synchronous or asynchronous?",
        "What are the application deadlines?",
        "Is GRE required?",
        "How many courses are required?",
        "Is the in-person program STEM OPT eligible?",
        "What is the 2-year thesis track?",
        "Who teaches machine learning?",
    ]

    for query in test_queries:
        results = retriever.search(query, n_results=5)
        print_results(query, results)

    print("\n" + "=" * 100)
    print("FILTERED SEARCH EXAMPLE: content_type='faq'")
    print("=" * 100)

    filtered_results = retriever.search(
        "Is GRE required?",
        n_results=3,
        content_type="faq",
    )
    print_results("Is GRE required? [FAQ only]", filtered_results)


if __name__ == "__main__":
    main()
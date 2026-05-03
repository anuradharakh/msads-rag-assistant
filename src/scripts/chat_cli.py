from __future__ import annotations

from src.generation.rag_chain import RAGChain


def print_sources(sources: list[dict]) -> None:
    if not sources:
        return

    print("\nSources:")
    for source in sources:
        print(
            f"- [{source['source_id']}] "
            f"{source.get('section_title')} | "
            f"{source.get('content_type')} | "
            f"{source.get('url')}"
        )


def main() -> None:
    rag = RAGChain()

    print("MSADS RAG Assistant")
    print("Model: qwen2.5:7b")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        if not question:
            continue

        result = rag.answer(question)

        print("\nAssistant:")
        print(result["answer"])
        print_sources(result["sources"])
        print()


if __name__ == "__main__":
    main()
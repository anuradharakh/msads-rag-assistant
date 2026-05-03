from __future__ import annotations

from src.generation.rag_chain import RAGChain


TEST_QUESTIONS = [
    "What is the capstone project?",
    "Is GRE required for admission?",
    "What are the application deadlines?",
    "Is the online program synchronous or asynchronous?",
    "Is the in-person program STEM OPT eligible?",
    "What is the 2-year thesis track?",
    "What scholarships are available?",
    "Does the online program provide visa sponsorship?",
]


def main() -> None:
    rag = RAGChain()

    for question in TEST_QUESTIONS:
        print("\n" + "=" * 100)
        print(f"QUESTION: {question}")
        print("=" * 100)

        result = rag.answer(question)

        print("\nANSWER:")
        print(result["answer"])

        print("\nSOURCES:")
        for source in result["sources"]:
            print(source)


if __name__ == "__main__":
    main()
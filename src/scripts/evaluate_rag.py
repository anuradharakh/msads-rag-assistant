from __future__ import annotations

from typing import Dict, List

from src.generation.rag_chain import RAGChain


TEST_SET = [
    {
        "question": "Is GRE required?",
        "expected": "GRE not required",
    },
    {
        "question": "What is the capstone project?",
        "expected": "capstone project real-world data science",
    },
    {
        "question": "Is the online program synchronous or asynchronous?",
        "expected": "both synchronous and asynchronous",
    },
    {
        "question": "Is the in-person program STEM OPT eligible?",
        "expected": "STEM OPT eligible",
    },
    {
        "question": "Does the online program provide visa sponsorship?",
        "expected": "no visa sponsorship",
    },
]


def score_answer(answer: str, expected: str) -> float:
    answer = answer.lower()
    expected_tokens = expected.lower().split()

    hits = sum(token in answer for token in expected_tokens)
    return hits / len(expected_tokens)


def score_groundedness(answer: str) -> float:
    return 1.0 if "[Source" in answer else 0.0


def score_retrieval(chunks: List[Dict]) -> float:
    if not chunks:
        return 0.0

    # use rerank_score if available
    scores = [
        c.get("rerank_score") or c.get("score") or 0
        for c in chunks
    ]

    return sum(scores) / len(scores)


def evaluate():
    rag = RAGChain()

    results = []

    for item in TEST_SET:
        question = item["question"]
        expected = item["expected"]

        output = rag.answer(question)

        answer = output["answer"]
        chunks = output["retrieved_chunks"]

        correctness = score_answer(answer, expected)
        groundedness = score_groundedness(answer)
        retrieval = score_retrieval(chunks)

        final_score = (
            0.5 * correctness +
            0.2 * groundedness +
            0.3 * retrieval
        )

        results.append({
            "question": question,
            "correctness": correctness,
            "groundedness": groundedness,
            "retrieval": retrieval,
            "final_score": final_score,
        })

        print("\n" + "=" * 80)
        print("QUESTION:", question)
        print("ANSWER:", answer[:200])
        print(f"Correctness: {correctness:.2f}")
        print(f"Groundedness: {groundedness:.2f}")
        print(f"Retrieval: {retrieval:.2f}")
        print(f"Final Score: {final_score:.2f}")

    avg_score = sum(r["final_score"] for r in results) / len(results)

    print("\n" + "=" * 80)
    print("OVERALL SCORE:", round(avg_score, 3))


if __name__ == "__main__":
    evaluate()
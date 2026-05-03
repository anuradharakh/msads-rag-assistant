from __future__ import annotations

from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference
)

from src.generation.rag_chain import RAGChain
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EVAL_QUESTIONS = [
    {
        "question": "Is GRE required for admission?",
        "reference": "The GRE/GMAT is not required for admission, but applicants may submit scores if they choose.",
    },
    {
        "question": "What is the capstone project?",
        "reference": "The capstone project is a required applied project involving real-world data science problems.",
    },
    {
        "question": "Is the online program synchronous or asynchronous?",
        "reference": "The online program includes both synchronous and asynchronous components.",
    },
    {
        "question": "Is the in-person program STEM OPT eligible?",
        "reference": "The full-time in-person MS in Applied Data Science program is STEM/OPT eligible.",
    },
    {
        "question": "Does the online program provide visa sponsorship?",
        "reference": "The online program is not eligible for visa sponsorship.",
    },
]


def build_eval_dataset() -> Dataset:
    rag = RAGChain()
    rows = []

    for item in EVAL_QUESTIONS:
        output = rag.answer(item["question"])

        rows.append(
            {
                "question": item["question"],
                "answer": output["answer"],
                "contexts": [
                    chunk.get("document", "")
                    for chunk in output.get("retrieved_chunks", [])
                    if chunk.get("document")
                ],
                "reference": item["reference"],
            }
        )

    return Dataset.from_list(rows)


def main() -> None:
    dataset = build_eval_dataset()

    judge = LangchainLLMWrapper(
        ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0,
        )
    )

    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithoutReference()
        ],
        llm=judge,
    )

    df = result.to_pandas()
    print(df)

    output_path = "data/processed/ragas_eval_results.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved RAGAS results to: {output_path}")


if __name__ == "__main__":
    main()
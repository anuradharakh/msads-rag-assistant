from __future__ import annotations

import ollama

from src.config import settings


class LocalLLMClient:
    def __init__(self) -> None:
        self.model = settings.llm_model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_ctx": 8192,
                },
            )

            return response["message"]["content"].strip()

        except Exception as exc:
            raise RuntimeError(
                "Failed to generate answer with Ollama. "
                "Make sure Ollama is running and qwen2.5:7b is pulled. "
                f"Original error: {exc}"
            ) from exc
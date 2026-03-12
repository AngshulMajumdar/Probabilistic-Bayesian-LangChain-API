from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class GeminiResponse:
    content: str


class GeminiColabLLM:
    """Thin adapter for Google Gemini. Meant for Colab or any environment with
    google-generativeai installed and credentials configured.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: str | None = None):
        try:
            import google.generativeai as genai
        except Exception as exc:
            raise RuntimeError(
                "google-generativeai is not installed. Add it to your environment to use Gemini."
            ) from exc

        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY before using the Gemini backend.")

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)

    def invoke(self, prompt: str) -> GeminiResponse:
        out = self._model.generate_content(prompt)
        text = getattr(out, "text", None)
        if text is None:
            text = str(out)
        return GeminiResponse(content=text)

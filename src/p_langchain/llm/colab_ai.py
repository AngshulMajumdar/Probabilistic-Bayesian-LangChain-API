# p_langchain/llm/colab_ai.py
from __future__ import annotations

from typing import Optional, Any
from .base import BaseLLM


class ColabGeminiLLM(BaseLLM):
    """
    Colab Gemini adapter (NO KEYS).

    Colab's google.colab.ai API changes often. This adapter prioritizes:
      1) prompt-only calls (positional), which are most stable
      2) prompt-only calls (keyword)
      3) minimal kwargs only if accepted

    If it still fails, we throw an error that includes which entry points exist,
    so we can patch in 30 seconds.
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model
        try:
            from google.colab import ai  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Failed to import google.colab.ai. "
                "Make sure you're in Google Colab and AI is enabled."
            ) from e
        self._ai = ai

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        return self._call_ai(prompt)

    def _call_ai(self, prompt: str) -> str:
        ai = self._ai

        # Candidate entry points to try (in order)
        fns = []
        for name in ("generate_text", "generate", "chat"):
            fn = getattr(ai, name, None)
            if callable(fn):
                fns.append((name, fn))

        if not fns:
            raise RuntimeError("google.colab.ai has no callable generate_text/generate/chat functions.")

        last_err: Optional[Exception] = None

        for name, fn in fns:
            # ---- Most stable: positional prompt only ----
            try:
                resp = fn(prompt)  # many versions accept a single positional prompt
                return _resp_to_text(resp)
            except Exception as e:
                last_err = e

            # ---- Next: keyword prompt= ----
            try:
                resp = fn(prompt=prompt)
                return _resp_to_text(resp)
            except Exception as e:
                last_err = e

            # ---- Chat-style fallback ----
            if name == "chat":
                try:
                    resp = fn(messages=[{"role": "user", "content": prompt}])
                    return _resp_to_text(resp)
                except Exception as e:
                    last_err = e

        # If we got here, nothing worked.
        # Include available callable names to patch quickly.
        avail = [n for n, _ in fns]
        raise RuntimeError(
            "Could not call google.colab.ai using prompt-only patterns. "
            f"Available entry points tried: {avail}. "
            f"Last error: {last_err}"
        )


def _resp_to_text(resp: Any) -> str:
    """Convert common response shapes to plain text."""
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        for k in ("text", "content", "output", "response"):
            v = resp.get(k)
            if isinstance(v, str):
                return v
        c = resp.get("candidates")
        if isinstance(c, list) and c:
            c0 = c[0]
            if isinstance(c0, str):
                return c0
            if isinstance(c0, dict):
                for k in ("text", "content"):
                    v = c0.get(k)
                    if isinstance(v, str):
                        return v
    for attr in ("text", "content", "output"):
        if hasattr(resp, attr) and isinstance(getattr(resp, attr), str):
            return getattr(resp, attr)
    return str(resp)

# p_langchain/llm/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseLLM(ABC):
    """
    Minimal LLM interface for our runtime.

    Contract:
      - generate(prompt) -> str
      - generate_n(prompt, n, ...) -> list[str] (default: calls generate() n times)
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        raise NotImplementedError

    def generate_n(
        self,
        prompt: str,
        n: int,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        if n <= 0:
            return []
        return [
            self.generate(prompt, temperature=temperature, max_tokens=max_tokens, **kwargs)
            for _ in range(n)
        ]

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from b_langchain.agents.bayesian_lc import BayesianLangChain, BayesianLCConfig
from b_langchain.llm.local_free import LocalHeuristicLLM

from bayesian_prob_langchain_api.services.tools import EchoTool, CalculatorTool, RetrieverTool


DEFAULT_DOCS = [
    {"id": "doc1", "text": "Bayesian LangChain uses a particle based orchestration runtime for tool selection."},
    {"id": "doc2", "text": "The probabilistic LangChain demo includes retrieval augmented generation and tool calling patterns."},
    {"id": "doc3", "text": "Gemini can be plugged in as a backend, but the runtime itself is backend agnostic."},
]


@dataclass
class ServiceContainer:
    docs: List[Dict[str, Any]] = field(default_factory=lambda: list(DEFAULT_DOCS))

    def __post_init__(self) -> None:
        self.tools: Dict[str, Any] = {
            "echo": EchoTool(),
            "calculator": CalculatorTool(),
            "retriever": RetrieverTool(self.docs),
        }

    def _gemini_available(self) -> bool:
        try:
            import importlib.util
            import os
            has_pkg = importlib.util.find_spec("google.generativeai") is not None
            has_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
            return has_pkg and has_key
        except Exception:
            return False

    @property
    def available_backends(self) -> List[str]:
        backends = ["local"]
        if self._gemini_available():
            backends.append("gemini")
        return backends

    def get_llm(self, backend: str):
        key = str(backend).strip().lower()
        if key == "local":
            return LocalHeuristicLLM(default_tool="echo")
        if key == "gemini":
            try:
                from b_langchain.llm.gemini_colab import GeminiColabLLM
            except Exception as exc:
                raise ValueError(
                    "unsupported_backend: gemini (optional dependency or credentials unavailable)"
                ) from exc
            return GeminiColabLLM()
        raise ValueError(f"unsupported_backend: {backend}")

    def make_agent(self, backend: str = "local") -> BayesianLangChain:
        llm = self.get_llm(backend)
        return BayesianLangChain(
            llm=llm,
            tools=self.tools,
            cfg=BayesianLCConfig(
                n_particles=8,
                max_steps=2,
                ess_resample_frac=0.5,
                temperature=1.0,
                tool_cost=0.1,
            ),
        )

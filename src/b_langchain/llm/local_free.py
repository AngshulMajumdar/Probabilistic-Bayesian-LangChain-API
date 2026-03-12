from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import List


@dataclass
class LLMResponse:
    content: str


class LocalHeuristicLLM:
    """A tiny offline fallback that emits the JSON protocol expected by JSONActionProposer.

    It is not a general language model. It is a deterministic local policy used to
    exercise the orchestration stack end to end without external services.
    """

    def __init__(self, default_tool: str = "echo"):
        self.default_tool = default_tool

    def _extract_tools(self, prompt: str) -> List[str]:
        m = re.search(r"Available tools:\s*\[(.*?)\]", prompt, flags=re.S)
        if not m:
            return []
        raw = m.group(1).strip()
        if not raw:
            return []
        parts = [p.strip().strip("'\"") for p in raw.split(",")]
        return [p for p in parts if p]

    def _extract_user(self, prompt: str) -> str:
        m = re.search(r"User:\s*(.*?)\nState:", prompt, flags=re.S)
        if m:
            return m.group(1).strip()
        return prompt.strip()

    def invoke(self, prompt: str) -> LLMResponse:
        tools = self._extract_tools(prompt)
        user_msg = self._extract_user(prompt)
        lower = user_msg.lower()

        # Prefer direct answers for greetings / meta questions.
        if any(tok in lower for tok in ["hello", "hi", "who are you", "what can you do"]):
            payload = {"kind": "ANSWER", "text": "I can choose tools and return structured actions for testing."}
            return LLMResponse(json.dumps(payload))

        chosen_tool = None
        if "calculator" in tools and re.search(r"\d+\s*[-+*/]\s*\d+", lower):
            expr = re.search(r"(\d+\s*[-+*/]\s*\d+(?:\s*[-+*/]\s*\d+)*)", user_msg)
            payload = {"kind": "TOOL", "name": "calculator", "input": {"expression": expr.group(1) if expr else user_msg}}
            return LLMResponse(json.dumps(payload))

        if "retriever" in tools and any(tok in lower for tok in ["document", "doc", "rag", "retrieve", "search", "about"]):
            payload = {"kind": "TOOL", "name": "retriever", "input": {"query": user_msg}}
            return LLMResponse(json.dumps(payload))

        if tools:
            chosen_tool = self.default_tool if self.default_tool in tools else tools[0]
        if chosen_tool is None:
            payload = {"kind": "CLARIFY", "question": "No tools available. Please configure the backend."}
        else:
            payload = {"kind": "TOOL", "name": chosen_tool, "input": {"text": user_msg}}
        return LLMResponse(json.dumps(payload))

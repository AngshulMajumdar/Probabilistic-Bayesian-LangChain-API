from __future__ import annotations

"""
LangChain wrapper for ColabGeminiLLM
(Pydantic-compatible version)
"""

from typing import Any, List, Optional

from pydantic import Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from p_langchain.llm.colab_ai import ColabGeminiLLM


class ColabGeminiChatModel(BaseChatModel):
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=512)

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._raw = ColabGeminiLLM()

    @property
    def _llm_type(self) -> str:
        return "colab_gemini_chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:

        # Flatten structured messages to text
        parts = []
        for m in messages:
            if isinstance(m, SystemMessage):
                role = "SYSTEM"
            elif isinstance(m, HumanMessage):
                role = "USER"
            else:
                role = "ASSISTANT"

            parts.append(f"{role}:\n{m.content}")

        prompt = "\n\n".join(parts)

        # IMPORTANT:
        # If your ColabGeminiLLM uses a different method name, change this line only.
        text = self._raw.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=text)
                )
            ]
        )

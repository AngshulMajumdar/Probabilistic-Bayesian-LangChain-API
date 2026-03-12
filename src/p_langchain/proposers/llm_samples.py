# p_langchain/proposers/llm_samples.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any

from p_langchain.core.types import Hypothesis, TraceEvent
from p_langchain.llm.base import BaseLLM
from .base import Proposer


PromptFn = Callable[[Hypothesis], str]


@dataclass
class LLMSamplesProposer(Proposer):
    """
    Expands a hypothesis by sampling N continuations from an LLM.

    - prompt_fn: maps hypothesis -> prompt string
    - n: number of samples
    - temperature: sampling temperature
    - max_tokens: optional cap
    - store_key: where to store raw generations in child.artifacts
    - write_to_state_key: optionally write generation into child.state[...]
    """
    llm: BaseLLM
    prompt_fn: PromptFn
    n: int = 3
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    store_key: str = "llm_generation"
    write_to_state_key: str = "llm_text"
    extra_llm_kwargs: Optional[Dict[str, Any]] = None

    def propose(self, h: Hypothesis) -> List[Hypothesis]:
        prompt = self.prompt_fn(h)
        kw = dict(self.extra_llm_kwargs or {})

        texts = self.llm.generate_n(
            prompt=prompt,
            n=self.n,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kw,
        )

        children: List[Hypothesis] = []
        for i, txt in enumerate(texts):
            child = h.copy()
            child.artifacts[self.store_key] = {
                "prompt": prompt,
                "index": i,
                "text": txt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            child.state[self.write_to_state_key] = txt

            # add a small trace (don't store huge prompt)
            child.trace.append(_trace("propose.llm_sample", idx=i, text_len=len(txt), prompt_len=len(prompt)))
            children.append(child)

        return children


def _trace(kind: str, **data: Any) -> TraceEvent:
    return TraceEvent(kind=kind, message="", data=dict(data))

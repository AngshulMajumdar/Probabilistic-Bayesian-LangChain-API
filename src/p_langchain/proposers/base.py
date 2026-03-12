# p_langchain/proposers/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from p_langchain.core.types import Hypothesis


class Proposer(ABC):
    """
    Proposer expands a hypothesis into a list of child hypotheses.
    Examples:
      - LLM sampling proposer (N continuations)
      - Prompt-variant proposer (paraphrase / perturb prompt)
      - Tool-branch proposer (simulate tool outcomes)
    """

    @abstractmethod
    def propose(self, h: Hypothesis) -> List[Hypothesis]:
        raise NotImplementedError

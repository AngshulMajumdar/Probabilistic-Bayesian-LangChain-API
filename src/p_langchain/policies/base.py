# p_langchain/policies/base.py
from __future__ import annotations

"""
Policy interface.

A "policy" decides what the system should do *after* we have a belief distribution.

Key idea to explain in interview:
---------------------------------
The executor produces a posterior distribution over hypotheses.
A policy looks at that distribution and decides:
  - ASK the user a clarifying question (if uncertainty is high), or
  - PROCEED with the best hypothesis (if confidence is high), or
  - STOP (if budget exhausted / max steps / etc.).

We keep policies separate so:
  - the probabilistic inference part stays reusable,
  - the decision logic stays swappable.
"""

from abc import ABC, abstractmethod

from p_langchain.core.types import Belief
from p_langchain.runtime.pchain import Action


class Policy(ABC):
    """Base interface for all policies."""

    @abstractmethod
    def decide(self, belief: Belief) -> Action:
        """
        Inspect belief (hypotheses with logw + normalized probs in state["_p"])
        and return an Action (Ask / Proceed / Stop).
        """
        raise NotImplementedError

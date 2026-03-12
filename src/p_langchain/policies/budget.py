# p_langchain/policies/budget.py
from __future__ import annotations

"""
Budget tracking policy.

Why this exists (interview explanation):
---------------------------------------
LLM calls are expensive. Even in Colab AI (no keys), we still want to show
"enterprise grade discipline": cap how many calls / steps we allow.

We track a simple budget object and can wrap another policy:
  - If budget is exceeded -> STOP
  - Else delegate to wrapped policy

This is intentionally minimal for the demo.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from p_langchain.core.types import Belief
from p_langchain.runtime.pchain import Action, StopAction
from .base import Policy


@dataclass
class Budget:
    """
    A simple budget tracker.

    You can track anything you like. For now we include:
      - llm_calls: total LLM calls allowed
      - steps: total executor steps allowed (optional)
    """
    max_llm_calls: int = 50
    max_steps: int = 20

    used_llm_calls: int = 0
    used_steps: int = 0

    def consume_llm_calls(self, n: int = 1) -> None:
        self.used_llm_calls += int(n)

    def consume_step(self, n: int = 1) -> None:
        self.used_steps += int(n)

    def ok(self) -> bool:
        return (self.used_llm_calls <= self.max_llm_calls) and (self.used_steps <= self.max_steps)

    def remaining(self) -> Dict[str, int]:
        return {
            "llm_calls": self.max_llm_calls - self.used_llm_calls,
            "steps": self.max_steps - self.used_steps,
        }


@dataclass
class BudgetPolicy(Policy):
    """
    Wrap any policy with a budget check.
    If budget exceeded -> STOP.
    Otherwise -> delegate to inner policy.
    """
    inner: Policy
    budget: Budget = field(default_factory=Budget)
    reason: str = "budget_exhausted"

    def decide(self, belief: Belief) -> Action:
        if not self.budget.ok():
            return StopAction(
                reason=self.reason,
                result=None,
                data={"budget": self.budget.__dict__, "remaining": self.budget.remaining()},
            )
        return self.inner.decide(belief)

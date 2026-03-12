# p_langchain/policies/stop.py
from __future__ import annotations

"""
Stop policy.

Purpose:
--------
Sometimes you want to stop early (or stop always) based on simple conditions:
  - if belief is empty
  - if top hypothesis probability is already high enough
  - if log-weight gap is big enough
  - etc.

For the Friday demo we may not rely heavily on this (AskOrProceed is the star),
but it is useful as a plug-in component.
"""

from dataclasses import dataclass
from typing import Optional

from p_langchain.core.types import Belief
from p_langchain.runtime.pchain import Action, ProceedAction, StopAction
from .base import Policy


@dataclass
class StopPolicy(Policy):
    """
    A simple stopping policy.

    - If belief is empty -> STOP.
    - If best probability >= p_stop -> PROCEED.
    - Else -> PROCEED (default) or STOP depending on configuration.

    This is intentionally conservative: it doesn't "ask", it only stops/proceeds.
    """
    p_key: str = "_p"
    p_stop: float = 0.90
    stop_if_uncertain: bool = False  # if True and not confident, STOP instead of PROCEED
    reason: str = "stop_policy"

    def decide(self, belief: Belief) -> Action:
        if belief.empty():
            return StopAction(reason="empty_belief")

        best = belief.best()
        if best is None:
            return StopAction(reason="no_best")

        p = best.state.get(self.p_key, None)
        try:
            p = float(p) if p is not None else None
        except Exception:
            p = None

        if p is not None and p >= self.p_stop:
            # confident enough, proceed
            # Note: PChain will attach PosteriorResult if needed.
            return ProceedAction(result=None, data={"p_best": p})  # type: ignore

        if self.stop_if_uncertain:
            return StopAction(reason=self.reason, data={"p_best": p})

        return ProceedAction(result=None, data={"p_best": p})  # type: ignore

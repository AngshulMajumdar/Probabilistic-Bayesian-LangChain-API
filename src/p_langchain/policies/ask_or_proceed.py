# p_langchain/policies/ask_or_proceed.py
from __future__ import annotations

"""
Ask-or-Proceed policy (the "probabilistic" demo killer).

Interview explanation (keep this narrative):
--------------------------------------------
The executor produces a *distribution* over hypotheses.
If the distribution is spread out, we are uncertain -> ask a clarification.
If the distribution is concentrated on one hypothesis, we are confident -> proceed.

We quantify "spread out" using Shannon entropy:
  H(p) = - sum_i p_i log p_i

- High entropy => many plausible interpretations => ASK a question
- Low entropy  => one interpretation dominates => PROCEED

This is a principled way to decide ask-vs-act rather than heuristics.
"""

from dataclasses import dataclass
from typing import Optional

from p_langchain.core.types import Belief
from p_langchain.core.utils import entropy
from p_langchain.runtime.pchain import Action, AskAction, ProceedAction
from .base import Policy


@dataclass
class AskOrProceedPolicy(Policy):
    """
    Decide whether to ask a clarification or proceed.

    Parameters:
    ----------
    p_key:
      where normalized probabilities are stored in hypothesis.state (executor uses "_p")
    entropy_threshold:
      if entropy >= threshold => ask
      (tune this in demos; ~0.7-1.2 works depending on how many hyps you keep)
    min_p_best:
      if best hypothesis prob is below this, also ask (extra safety)
    question_key:
      if the best hypothesis already contains a suggested question in state[question_key],
      we use it; otherwise we use default_question.
    default_question:
      fallback clarification question
    """
    p_key: str = "_p"
    entropy_threshold: float = 0.9
    min_p_best: float = 0.55

    question_key: str = "clarifying_question"
    default_question: str = "Could you clarify what you mean (a bit more detail will help)?"

    def decide(self, belief: Belief) -> Action:
        if belief.empty():
            # If no candidates survive, safest is to ask user for clarification.
            return AskAction(question=self.default_question, data={"reason": "empty_belief"})

        # Ensure probabilities exist (executor usually already normalized).
        # If not present, normalize on the fly.
        if belief.hyps and self.p_key not in belief.hyps[0].state:
            belief = belief.normalize(key=self.p_key)

        ps = belief.probs(self.p_key)
        H = entropy(ps)

        best = belief.best()
        p_best = float(best.state.get(self.p_key, 0.0)) if best is not None else 0.0

        # Decision rule:
        #  - ask if entropy high (uncertainty)
        #  - or if p_best too low (no dominant mode)
        if (H >= self.entropy_threshold) or (p_best < self.min_p_best):
            # If the best hypothesis already proposes a question, use it.
            q = None
            if best is not None:
                qv = best.state.get(self.question_key, None)
                if isinstance(qv, str) and qv.strip():
                    q = qv.strip()
            if q is None:
                q = self.default_question

            return AskAction(
                question=q,
                data={
                    "reason": "high_uncertainty",
                    "entropy": H,
                    "p_best": p_best,
                    "num_hypotheses": len(belief.hyps),
                },
            )

        # Otherwise proceed. PChain will attach the PosteriorResult.
        return ProceedAction(
            result=None,  # type: ignore
            data={
                "reason": "confident",
                "entropy": H,
                "p_best": p_best,
                "num_hypotheses": len(belief.hyps),
            },
        )

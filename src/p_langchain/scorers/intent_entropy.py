# p_langchain/scorers/intent_entropy.py
from __future__ import annotations

"""
Intent entropy scorer.

Purpose (demo 1: ambiguous intent):
-----------------------------------
We want multiple hypotheses that correspond to different "intents"
(e.g., 'summarize', 'write email', 'explain', 'code', etc.).

This scorer:
  1) reads an intent label from each hypothesis (state[intent_key])
  2) aggregates probability mass per intent across the belief
  3) computes entropy over intents
  4) stores summary fields into each hypothesis.state (so policy can use it)

BUT WAIT: scorers operate on a single hypothesis, not the whole belief.
So how do we compute entropy over intents?

Answer:
-------
We implement this as a *per-hypothesis* scorer that:
  - rewards hypotheses that specify an intent (so they survive),
and we compute belief-level intent entropy in the demo/policy layer.

To keep the package clean and simple:
  - This file provides helper functions to compute intent distribution and entropy.
  - The AskOrProceedPolicy already computes entropy over hypotheses.
For intent-level entropy (grouping by intent), we compute it in the demo and
store 'clarifying_question' in the best hypothesis when needed.

This keeps executor generic.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from p_langchain.core.types import Belief, Hypothesis
from p_langchain.core.utils import entropy
from .base import Scorer, ScoreResult
from p_langchain.core.types import TraceEvent


def intent_distribution(belief: Belief, intent_key: str = "intent", p_key: str = "_p") -> Dict[str, float]:
    """
    Aggregate probability mass per intent label.
    Returns dict intent -> mass (sums to ~1 if belief probabilities sum to 1).
    """
    if belief.empty():
        return {}
    # ensure normalized
    if belief.hyps and p_key not in belief.hyps[0].state:
        belief = belief.normalize(key=p_key)

    dist: Dict[str, float] = {}
    for h in belief.hyps:
        intent = h.state.get(intent_key, None)
        if intent is None:
            intent = "UNKNOWN"
        intent = str(intent)
        p = float(h.state.get(p_key, 0.0))
        dist[intent] = dist.get(intent, 0.0) + p
    return dist


def intent_entropy(belief: Belief, intent_key: str = "intent", p_key: str = "_p") -> float:
    """
    Compute entropy over aggregated intent masses (not per-hypothesis entropy).
    This is often more meaningful for "ask vs proceed".
    """
    dist = intent_distribution(belief, intent_key=intent_key, p_key=p_key)
    ps = list(dist.values())
    return entropy(ps)


@dataclass
class IntentLabelPresenceScorer(Scorer):
    """
    Cheap scorer to encourage hypotheses to explicitly set an intent label.

    If your proposer (or step) produces hypotheses with intent labels, this helps.
    If not, you can skip this scorer.

    Typical use:
      - You create hypotheses with different intents (via prompt variants or LLM prompt).
      - Reward those that contain state["intent"].
    """
    intent_key: str = "intent"
    present_reward: float = 0.5
    missing_penalty: float = -0.5
    name: str = "intent_presence"

    def score(self, h: Hypothesis) -> ScoreResult:
        has = self.intent_key in h.state and str(h.state.get(self.intent_key, "")).strip() != ""
        if has:
            ev = TraceEvent(kind=f"score.{self.name}.present", message="intent present", data={"key": self.intent_key})
            return ScoreResult(score_delta=self.present_reward, meta={"has_intent": True}, event=ev)
        ev = TraceEvent(kind=f"score.{self.name}.missing", message="intent missing", data={"key": self.intent_key})
        return ScoreResult(score_delta=self.missing_penalty, meta={"has_intent": False}, event=ev)

# p_langchain/scorers/critique.py
from __future__ import annotations

"""
LLM-based critique scorer.

What it does (interview-ready):
-------------------------------
Given a hypothesis with some candidate text (state["llm_text"]),
we ask the LLM to "grade" it on a few axes like:
  - Is it consistent with the user's intent?
  - Is it internally consistent?
  - Is it well-formed / safe?
Then convert that critique into a numeric score delta.

Why we still keep it:
---------------------
Rule-based checks (consistency) + schema checks (JSON validity) are cheap,
but they miss semantic issues.
This scorer is an optional semantic filter.

Design choice:
--------------
We return a *small* score adjustment (not huge), so cheap checks dominate,
and critique just helps break ties / remove nonsense.

We keep the critique prompt minimal and robust.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from p_langchain.core.types import Hypothesis, TraceEvent
from p_langchain.llm.base import BaseLLM
from .base import Scorer, ScoreResult


_CRITIQUE_PROMPT = """You are a strict evaluator.
Given the USER REQUEST and a CANDIDATE OUTPUT, judge the candidate.

Return ONLY valid JSON with these keys:
  - supported: number in [0,1]  (does it follow the user request?)
  - consistent: number in [0,1] (is it self-consistent / non-contradictory?)
  - clarity: number in [0,1]    (is it clear and usable?)
  - notes: short string

USER REQUEST:
{user_request}

CANDIDATE OUTPUT:
{candidate}
"""


def _safe_json_loads(text: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return True, obj, None
        return False, None, "critique returned non-dict json"
    except Exception as e:
        return False, None, str(e)


def _clip01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


@dataclass
class CritiqueScorer(Scorer):
    """
    LLM-based critique scorer.

    Required:
      - llm: an LLM adapter (ColabGeminiLLM)
      - user_request_key: where the user's request lives in h.state
      - candidate_key: which text to critique (default "llm_text")

    Scoring:
      score_delta = scale * (0.5*supported + 0.3*consistent + 0.2*clarity - 0.5)
    So:
      - average ~0 gives near 0 delta
      - very good (~1) gives positive delta
      - very bad (~0) gives negative delta
    """
    llm: BaseLLM
    user_request_key: str = "user_request"
    candidate_key: str = "llm_text"
    temperature: float = 0.2
    max_tokens: Optional[int] = 256
    scale: float = 4.0
    name: str = "critique"

    def score(self, h: Hypothesis) -> ScoreResult:
        user_req = h.state.get(self.user_request_key, "")
        cand = h.state.get(self.candidate_key, "")

        user_req = user_req if isinstance(user_req, str) else str(user_req)
        cand = cand if isinstance(cand, str) else str(cand)

        prompt = _CRITIQUE_PROMPT.format(user_request=user_req, candidate=cand)

        raw = self.llm.generate(prompt, temperature=self.temperature, max_tokens=self.max_tokens)

        ok, obj, err = _safe_json_loads(raw if isinstance(raw, str) else str(raw))
        if not ok or obj is None:
            ev = TraceEvent(
                kind=f"score.{self.name}.parse_fail",
                message="critique json parse failed",
                data={"error": err, "raw_len": len(str(raw))},
            )
            # Small penalty: critique failure shouldn't kill the run.
            return ScoreResult(score_delta=-1.0, meta={"ok": False, "error": err}, event=ev)

        supported = _clip01(obj.get("supported", 0.0))
        consistent = _clip01(obj.get("consistent", 0.0))
        clarity = _clip01(obj.get("clarity", 0.0))
        notes = obj.get("notes", "")

        # Weighted average in [0,1]
        agg = 0.5 * supported + 0.3 * consistent + 0.2 * clarity

        # Center around 0.5 so "meh" ~0 delta.
        delta = self.scale * (agg - 0.5)

        ev = TraceEvent(
            kind=f"score.{self.name}.ok",
            message="critique scored",
            data={
                "supported": supported,
                "consistent": consistent,
                "clarity": clarity,
                "agg": agg,
                "delta": delta,
            },
        )

        return ScoreResult(
            score_delta=float(delta),
            meta={"ok": True, "supported": supported, "consistent": consistent, "clarity": clarity, "notes": notes},
            event=ev,
        )

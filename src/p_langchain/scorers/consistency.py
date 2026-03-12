# p_langchain/scorers/consistency.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

from p_langchain.core.types import Hypothesis, TraceEvent
from .base import Scorer, ScoreResult


RuleResult = Tuple[bool, str, Dict[str, Any]]  # (ok?, reason, meta)


def _rule_required_keys(required: Iterable[str]):
    req = list(required)

    def rule(h: Hypothesis) -> RuleResult:
        missing = [k for k in req if k not in h.state]
        if missing:
            return (False, f"missing_required_keys:{missing}", {"missing": missing})
        return (True, "required_keys_ok", {"required": req})

    return rule


def _rule_no_contradiction(flag_key: str, forbidden_keys_if_true: Iterable[str]):
    forb = list(forbidden_keys_if_true)

    def rule(h: Hypothesis) -> RuleResult:
        if bool(h.state.get(flag_key, False)):
            present = [k for k in forb if k in h.state and h.state.get(k) not in (None, "")]
            if present:
                return (
                    False,
                    f"contradiction:{flag_key}=True but has {present}",
                    {"flag_key": flag_key, "present": present},
                )
        return (True, "no_contradiction_ok", {"flag_key": flag_key})

    return rule


def _rule_nonempty_state_key(key: str):
    def rule(h: Hypothesis) -> RuleResult:
        v = h.state.get(key, None)
        if v is None:
            return (False, f"missing:{key}", {"key": key})
        if isinstance(v, str) and len(v.strip()) == 0:
            return (False, f"empty_string:{key}", {"key": key})
        return (True, f"nonempty_ok:{key}", {"key": key})

    return rule


@dataclass
class ConsistencyScorer(Scorer):
    """
    Cheap, rule-based consistency checks.

    It DOES NOT enforce JSON validity. It enforces internal logic constraints on h.state.
    Use it to prune obviously bad branches early.

    Scoring:
      - If any rule fails: add fail_penalty (negative)
      - If all pass: add pass_reward (small positive, can be 0)
    """
    rules: List[Any] = field(default_factory=list)
    fail_penalty: float = -5.0
    pass_reward: float = 0.0
    name: str = "consistency"

    def score(self, h: Hypothesis) -> ScoreResult:
        failures: List[Dict[str, Any]] = []
        passes = 0

        for r in self.rules:
            ok, reason, meta = r(h)
            if ok:
                passes += 1
            else:
                failures.append({"reason": reason, "meta": meta})

        if failures:
            ev = TraceEvent(
                kind=f"score.{self.name}.fail",
                message="consistency failed",
                data={"failures": failures, "passes": passes, "num_rules": len(self.rules)},
            )
            return ScoreResult(score_delta=self.fail_penalty, meta={"failures": failures}, event=ev)

        ev = TraceEvent(
            kind=f"score.{self.name}.pass",
            message="consistency passed",
            data={"passes": passes, "num_rules": len(self.rules)},
        )
        return ScoreResult(score_delta=self.pass_reward, meta={"passes": passes}, event=ev)


# -----------------------------
# Ready-made rule bundles
# -----------------------------

def default_rules_for_demo() -> List[Any]:
    """
    Sensible defaults for our early demos.
    You can adjust later in demos based on chosen 'state' contract.
    """
    rules: List[Any] = []
    # We expect proposer writes sampled text into state["llm_text"]
    rules.append(_rule_required_keys(["llm_text"]))
    rules.append(_rule_nonempty_state_key("llm_text"))

    # If a hypothesis says it needs clarification, it shouldn't also claim a final answer.
    rules.append(_rule_no_contradiction("needs_clarification", ["final_answer"]))

    return rules

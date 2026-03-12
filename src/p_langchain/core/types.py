# p_langchain/core/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math


@dataclass
class TraceEvent:
    """A lightweight, structured event for debugging/explainability."""
    kind: str
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Hypothesis:
    """
    One candidate world-state in the beam.
    - state: arbitrary dict (intent, parsed_json, tool_calls, etc.)
    - logw: unnormalized log weight / score accumulator
    - trace: list of TraceEvent for auditability
    - artifacts: extra objects (raw LLM outputs, intermediate parses, etc.)
    """
    state: Dict[str, Any] = field(default_factory=dict)
    logw: float = 0.0
    trace: List[TraceEvent] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def with_event(self, kind: str, message: str = "", **data: Any) -> "Hypothesis":
        """Return a shallow-copied hypothesis with one more trace event."""
        h = Hypothesis(
            state=dict(self.state),
            logw=float(self.logw),
            trace=list(self.trace),
            artifacts=dict(self.artifacts),
        )
        h.trace.append(TraceEvent(kind=kind, message=message, data=dict(data)))
        return h

    def copy(self) -> "Hypothesis":
        return Hypothesis(
            state=dict(self.state),
            logw=float(self.logw),
            trace=list(self.trace),
            artifacts=dict(self.artifacts),
        )


@dataclass
class Belief:
    """
    A belief is just a collection of hypotheses.
    Convention:
      - hyps[i].logw are unnormalized log-weights
      - normalize() returns probabilities in hyps[i].state["_p"]
    """
    hyps: List[Hypothesis] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.hyps)

    def empty(self) -> bool:
        return len(self.hyps) == 0

    def topk(self, k: int) -> "Belief":
        if k <= 0:
            return Belief([])
        return Belief(sorted(self.hyps, key=lambda h: h.logw, reverse=True)[:k])

    def best(self) -> Optional[Hypothesis]:
        if not self.hyps:
            return None
        return max(self.hyps, key=lambda h: h.logw)

    def normalize(self, key: str = "_p") -> "Belief":
        """
        Adds normalized probabilities into each hypothesis.state[key].
        Returns a new Belief (does not mutate original hyps).
        """
        if not self.hyps:
            return Belief([])

        logws = [h.logw for h in self.hyps]
        m = max(logws)
        exps = [math.exp(lw - m) for lw in logws]
        z = sum(exps)
        if z == 0.0 or not math.isfinite(z):
            # fallback: uniform
            p = 1.0 / len(self.hyps)
            out = []
            for h in self.hyps:
                hh = h.copy()
                hh.state[key] = p
                out.append(hh)
            return Belief(out)

        out = []
        for h, e in zip(self.hyps, exps):
            hh = h.copy()
            hh.state[key] = e / z
            out.append(hh)
        return Belief(out)

    def probs(self, key: str = "_p") -> List[float]:
        """Convenience: get probabilities from normalized belief; if missing, return NaNs."""
        ps = []
        for h in self.hyps:
            v = h.state.get(key, float("nan"))
            ps.append(float(v) if v is not None else float("nan"))
        return ps


@dataclass
class PosteriorResult:
    """Returned by the runtime at the end of a run."""
    best: Optional[Hypothesis]
    posterior: Belief
    trace_summary: List[TraceEvent] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "best_state": None if self.best is None else self.best.state,
            "best_logw": None if self.best is None else self.best.logw,
            "num_hypotheses": len(self.posterior.hyps),
            "trace_summary": [
                {"kind": e.kind, "message": e.message, "data": e.data}
                for e in self.trace_summary
            ],
        }

# p_langchain/runtime/executor.py
from __future__ import annotations

"""
BeamExecutor = the probabilistic "engine room".

Mental model (very interview-friendly):
--------------------------------------
We maintain multiple candidate futures (hypotheses) in parallel.
Each hypothesis has:
  - a mutable 'state' dict (what we believe so far),
  - a log-weight 'logw' (how good / likely / high-quality it is),
  - a 'trace' for explainability.

At each step we:
  1) EXPAND: propose children hypotheses (e.g., sample N LLM continuations)
  2) SCORE: run one or more scorers (schema/consistency/critique, etc.)
  3) PRUNE: keep only top-K hypotheses (beam search)
  4) NORMALIZE: turn log-weights into probabilities (a belief distribution)

This is "probabilistic" because:
  - we keep a distribution over hypotheses (not a single greedy path),
  - we can measure uncertainty (entropy over probabilities),
  - policies can decide: ask clarification vs proceed vs stop based on uncertainty.

We deliberately keep this engine independent from LangChain/LangGraph internals.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from p_langchain.core.types import Belief, Hypothesis, PosteriorResult, TraceEvent
from p_langchain.core.utils import normalize_logweights
from p_langchain.proposers.base import Proposer
from p_langchain.scorers.base import Scorer, ScoreResult


@dataclass
class BeamConfig:
    """
    All knobs in one place so the demo is easy to explain.

    beam_size:
      how many hypotheses we keep after pruning each step.
    max_steps:
      hard cap on steps (safety to avoid infinite loops).
    dedup_key:
      optional: a key in h.state used for deduplication (keep best per key).
      Example: if many hypotheses propose the same 'final_answer', keep best one.
    """
    beam_size: int = 5
    max_steps: int = 3
    dedup_key: Optional[str] = None


@dataclass
class BeamExecutor:
    """
    The executor only knows:
      - proposers: how to expand hypotheses
      - scorers: how to evaluate hypotheses

    It does NOT know what the 'task' is. The task lives in:
      - your prompt_fn used by LLM proposer
      - your state layout (what keys you store in h.state)
      - your policy (ask/proceed/stop) that will wrap this later

    This makes it modular and reusable.
    """
    proposers: List[Proposer]
    scorers: List[Scorer]
    config: BeamConfig = field(default_factory=BeamConfig)

    def run(self, init: Hypothesis) -> PosteriorResult:
        """
        Run the beam loop starting from a single initial hypothesis.

        Returns:
          PosteriorResult(best, posterior, trace_summary)

        trace_summary is a short list of high-level events useful for printing.
        Each hypothesis also carries its own detailed trace.
        """
        # Start belief with a single hypothesis.
        belief = Belief([init])

        summary: List[TraceEvent] = []
        summary.append(TraceEvent(kind="executor.start", message="start", data={
            "beam_size": self.config.beam_size,
            "max_steps": self.config.max_steps,
            "num_proposers": len(self.proposers),
            "num_scorers": len(self.scorers),
        }))

        # Iterate steps
        for step in range(self.config.max_steps):
            if belief.empty():
                summary.append(TraceEvent(kind="executor.empty", message="no hypotheses left", data={"step": step}))
                break

            # ---------------------------
            # 1) EXPAND
            # ---------------------------
            expanded: List[Hypothesis] = []
            for h in belief.hyps:
                children = [h]  # default: keep itself if no proposer expands it
                for proposer in self.proposers:
                    # Apply proposer to every current hypothesis.
                    # Each proposer returns a list of hypotheses (children).
                    # Important: proposers should COPY hypotheses rather than mutate shared objects.
                    out = proposer.propose(h)
                    children = out  # Typically you want proposer to define new children.
                    # If you later want "chain" proposers, you can combine lists here.
                expanded.extend(children)

            summary.append(TraceEvent(
                kind="executor.expand",
                message="expanded",
                data={"step": step, "before": len(belief.hyps), "after": len(expanded)},
            ))

            # ---------------------------
            # 2) SCORE
            # ---------------------------
            scored: List[Hypothesis] = []
            for h in expanded:
                hh = h.copy()
                total_delta = 0.0
                meta_bundle: Dict[str, Any] = {}

                # Apply each scorer and accumulate log-score deltas.
                # This is like adding log-likelihood terms.
                for scorer in self.scorers:
                    res: ScoreResult = scorer.score(hh)
                    total_delta += float(res.score_delta)
                    if res.meta:
                        # store under scorer name if possible
                        key = scorer.__class__.__name__
                        meta_bundle[key] = res.meta
                    if res.event is not None:
                        hh.trace.append(res.event)

                hh.logw += total_delta
                # store one summary meta record for debugging (optional)
                hh.artifacts["score_meta"] = meta_bundle
                scored.append(hh)

            summary.append(TraceEvent(
                kind="executor.score",
                message="scored",
                data={"step": step, "num_scored": len(scored)},
            ))

            # ---------------------------
            # 3) OPTIONAL DEDUPLICATION
            # ---------------------------
            if self.config.dedup_key is not None:
                scored = self._dedup_keep_best(scored, key=self.config.dedup_key)
                summary.append(TraceEvent(
                    kind="executor.dedup",
                    message="deduplicated",
                    data={"step": step, "dedup_key": self.config.dedup_key, "num_left": len(scored)},
                ))

            # ---------------------------
            # 4) PRUNE (beam)
            # ---------------------------
            scored.sort(key=lambda x: x.logw, reverse=True)
            pruned = scored[: max(1, int(self.config.beam_size))]

            summary.append(TraceEvent(
                kind="executor.prune",
                message="pruned",
                data={
                    "step": step,
                    "kept": len(pruned),
                    "best_logw": pruned[0].logw if pruned else None,
                    "worst_logw": pruned[-1].logw if pruned else None,
                },
            ))

            # ---------------------------
            # 5) UPDATE BELIEF
            # ---------------------------
            belief = Belief(pruned)

            # We often want normalized probabilities for policies (entropy etc.).
            # We do not overwrite logw; we store probabilities into state["_p"].
            belief = belief.normalize(key="_p")

            summary.append(TraceEvent(
                kind="executor.belief",
                message="belief normalized",
                data={
                    "step": step,
                    "probs": belief.probs("_p"),
                },
            ))

        best = belief.best()
        summary.append(TraceEvent(
            kind="executor.done",
            message="done",
            data={"best_logw": None if best is None else best.logw, "num_final": len(belief.hyps)},
        ))
        return PosteriorResult(best=best, posterior=belief, trace_summary=summary)

    @staticmethod
    def _dedup_keep_best(hyps: List[Hypothesis], key: str) -> List[Hypothesis]:
        """
        Deduplicate hypotheses by h.state[key] and keep only the highest logw for each value.

        Useful when many branches converge to same final output.
        """
        best_map: Dict[Any, Hypothesis] = {}
        for h in hyps:
            v = h.state.get(key, None)
            if v not in best_map or h.logw > best_map[v].logw:
                best_map[v] = h
        return list(best_map.values())

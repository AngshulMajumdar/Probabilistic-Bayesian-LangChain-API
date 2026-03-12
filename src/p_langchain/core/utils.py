# p_langchain/core/utils.py
from __future__ import annotations

import math
import random
from typing import Iterable, List, Optional, Sequence

# -----------------------------
# Numerics
# -----------------------------

def logsumexp(logws: Sequence[float]) -> float:
    """Stable log(sum(exp(logws))). Returns -inf for empty input."""
    if not logws:
        return float("-inf")
    m = max(logws)
    if not math.isfinite(m):
        return m
    s = 0.0
    for lw in logws:
        s += math.exp(lw - m)
    return m + math.log(s) if s > 0.0 else float("-inf")


def normalize_logweights(logws: Sequence[float]) -> List[float]:
    """
    Convert unnormalized log-weights to probabilities.
    Returns a list of probs summing to 1. If all weights are -inf, returns uniform.
    """
    if not logws:
        return []
    lse = logsumexp(logws)
    if not math.isfinite(lse):
        # uniform fallback
        n = len(logws)
        return [1.0 / n] * n
    ps = [math.exp(lw - lse) for lw in logws]
    z = sum(ps)
    if z <= 0.0 or not math.isfinite(z):
        n = len(logws)
        return [1.0 / n] * n
    return [p / z for p in ps]


def entropy(probs: Sequence[float], eps: float = 1e-12) -> float:
    """Shannon entropy in nats."""
    h = 0.0
    for p in probs:
        p = float(p)
        if p <= 0.0:
            continue
        h -= p * math.log(max(p, eps))
    return h


# -----------------------------
# Randomness helpers
# -----------------------------

def set_seed(seed: Optional[int]) -> None:
    """Seed Python's random module (kept minimal; no numpy dependency)."""
    if seed is None:
        return
    random.seed(int(seed))


def sample_indices_from_probs(probs: Sequence[float], k: int) -> List[int]:
    """
    Sample k indices with replacement according to probs.
    Minimal helper used by some proposers/policies if needed.
    """
    if k <= 0:
        return []
    # random.choices expects weights, not necessarily normalized
    return random.choices(range(len(probs)), weights=list(probs), k=k)

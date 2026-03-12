# p_langchain/scorers/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from p_langchain.core.types import Hypothesis, TraceEvent


@dataclass
class ScoreResult:
    """
    score_delta: add to hypothesis.logw (can be negative)
    meta: optional structured info (e.g., reasons, violations, parsed fields)
    event: optional trace event to attach
    """
    score_delta: float
    meta: Dict[str, Any] = field(default_factory=dict)
    event: Optional[TraceEvent] = None


class Scorer(ABC):
    """
    A scorer evaluates a hypothesis and returns a score delta (log-space).
    It should be cheap when possible; LLM-based scorers should be optional.
    """

    @abstractmethod
    def score(self, h: Hypothesis) -> ScoreResult:
        raise NotImplementedError

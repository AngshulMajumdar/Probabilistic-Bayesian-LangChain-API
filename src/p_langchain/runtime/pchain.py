# p_langchain/runtime/pchain.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

from p_langchain.core.types import Belief, Hypothesis, PosteriorResult
from p_langchain.runtime.executor import BeamConfig, BeamExecutor
from p_langchain.proposers.base import Proposer
from p_langchain.scorers.base import Scorer


@dataclass
class AskAction:
    question: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProceedAction:
    result: Optional[PosteriorResult] = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StopAction:
    reason: str
    result: Optional[PosteriorResult] = None
    data: Dict[str, Any] = field(default_factory=dict)


Action = Union[AskAction, ProceedAction, StopAction]


class Policy(Protocol):
    def decide(self, belief: Belief) -> Action:
        ...


StepFn = Callable[[Hypothesis], Hypothesis]


@dataclass
class PChain:
    proposers: List[Proposer]
    scorers: List[Scorer]
    policy: Policy
    beam_config: BeamConfig = field(default_factory=BeamConfig)
    step_fn: Optional[StepFn] = None

    def run(self, init_state: Optional[Dict[str, Any]] = None, init_logw: float = 0.0) -> Action:
        init = Hypothesis(state=dict(init_state or {}), logw=float(init_logw))

        if self.step_fn is not None:
            init = self.step_fn(init)

        executor = BeamExecutor(
            proposers=self.proposers,
            scorers=self.scorers,
            config=self.beam_config,
        )

        posterior = executor.run(init)
        action = self.policy.decide(posterior.posterior)

        # ✅ Bulletproof: attach posterior if the action can carry it.
        # This avoids any isinstance/module-identity weirdness.
        if hasattr(action, "result") and getattr(action, "result") is None:
            setattr(action, "result", posterior)

        return action

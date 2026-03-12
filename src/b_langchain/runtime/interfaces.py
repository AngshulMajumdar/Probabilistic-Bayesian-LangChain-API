from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class Action:
    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)


class ActionProposer(Protocol):
    def propose(self, user_msg: str, particle: Any) -> List[Action]:
        ...


class Observer(Protocol):
    def observe(self, action: Action) -> Optional[Observation]:
        ...


class TransitionModel(Protocol):
    def transition(self, state: Dict[str, Any], action: Action, obs: Optional[Observation]) -> Dict[str, Any]:
        ...


class ObsLikelihood(Protocol):
    def loglik(self, prev_state: Dict[str, Any], action: Action, obs: Optional[Observation], next_state: Dict[str, Any]) -> float:
        ...


class CostModel(Protocol):
    def cost(self, prev_state: Dict[str, Any], action: Action, obs: Optional[Observation], next_state: Dict[str, Any]) -> float:
        ...

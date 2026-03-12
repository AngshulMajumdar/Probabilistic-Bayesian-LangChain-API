from __future__ import annotations

from typing import Dict, List, Optional

from b_langchain.core.types import Particle
from b_langchain.runtime.interfaces import (
    Action, Observation, ActionProposer, TransitionModel, Observer, ObsLikelihood, CostModel
)

# ---------- Minimal defaults (to verify pipeline) ----------


class SimpleProposer(ActionProposer):
    def __init__(self, tool_name: str = "echo"):
        self.tool_name = tool_name

    def propose(self, user_msg: str, particle: Particle) -> List[Action]:
        # Always propose a tool call.
        return [Action(kind="TOOL", payload={"name": self.tool_name, "input": {"text": user_msg}})]


class DictToolObserver(Observer):
    def __init__(self, tools: Dict[str, object]):
        self.tools = tools

    def observe(self, action: Action) -> Optional[Observation]:
        if action.kind != "TOOL":
            return None
        name = action.payload.get("name", "")
        inp = action.payload.get("input", {})
        tool = self.tools.get(name)
        if tool is None:
            return Observation(kind="TOOL_RESULT", payload={"name": name, "ok": False, "error": "tool_not_found"})
        try:
            out = tool.invoke(inp)
            return Observation(kind="TOOL_RESULT", payload={"name": name, "ok": True, "output": out})
        except Exception as e:
            return Observation(kind="TOOL_RESULT", payload={"name": name, "ok": False, "error": f"{type(e).__name__}: {e}"})


class IdentityTransition(TransitionModel):
    def transition(self, state: Dict, action: Action, obs: Optional[Observation]) -> Dict:
        s = dict(state)
        s["last_action"] = action.kind
        s["last_payload"] = dict(action.payload)
        if obs is not None:
            s["last_obs"] = {"kind": obs.kind, "payload": dict(obs.payload)}
        return s


class SimpleObsLikelihood(ObsLikelihood):
    def __init__(self, tool_bonus: float = 2.0, tool_fail_penalty: float = 3.0):
        self.tool_bonus = tool_bonus
        self.tool_fail_penalty = tool_fail_penalty

    def loglik(self, prev_state: Dict, action: Action, obs: Optional[Observation], next_state: Dict) -> float:
        if action.kind == "TOOL":
            if obs and obs.payload.get("ok") is True:
                return +self.tool_bonus
            return -self.tool_fail_penalty
        return 0.0


class SimpleCost(CostModel):
    def __init__(self, tool_cost: float = 0.2):
        self.tool_cost = tool_cost

    def cost(self, prev_state: Dict, action: Action, obs: Optional[Observation], next_state: Dict) -> float:
        return self.tool_cost if action.kind == "TOOL" else 0.0

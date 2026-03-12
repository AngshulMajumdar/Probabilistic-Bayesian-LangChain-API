from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import re

from b_langchain.runtime.smc_agent import SMCAgent, SMCConfig
from b_langchain.runtime.interfaces import Action, Observation
from b_langchain.core.types import Particle


# ---------------------------
# Tool Observer (LangChain Tool API: tool.invoke(dict))
# ---------------------------

class LangChainToolObserver:
    def __init__(self, tools: Dict[str, Any]):
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


# ---------------------------
# Proposer (uses an LLM to propose actions in JSON)
# ---------------------------

class JSONActionProposer:
    def __init__(self, llm: Any, tool_names: List[str]):
        self.llm = llm
        self.tool_names = tool_names

    def _extract_json(self, txt: str) -> Optional[Dict[str, Any]]:
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    def propose(self, user_msg: str, particle: Particle) -> List[Action]:
        prompt = (
            "Return ONLY JSON. Choose ONE action.\n"
            "Allowed:\n"
            "1) {\"kind\":\"TOOL\",\"name\":\"<tool>\",\"input\":{...}}\n"
            "2) {\"kind\":\"ANSWER\",\"text\":\"...\"}\n"
            "3) {\"kind\":\"CLARIFY\",\"question\":\"...\"}\n\n"
            f"Available tools: {self.tool_names}\n"
            f"User: {user_msg}\n"
            f"State: {particle.state}\n"
        )
        out = self.llm.invoke(prompt).content
        j = self._extract_json(out)
        if not j or "kind" not in j:
            return [Action(kind="CLARIFY", payload={"question": "Can you clarify?"})]

        k = j.get("kind", "")
        if k == "TOOL":
            name = j.get("name", "")
            inp = j.get("input", {})
            return [Action(kind="TOOL", payload={"name": name, "input": inp})]
        if k == "ANSWER":
            return [Action(kind="ANSWER", payload={"text": j.get("text", "")})]
        if k == "CLARIFY":
            return [Action(kind="CLARIFY", payload={"question": j.get("question", "")})]
        return [Action(kind="CLARIFY", payload={"question": "Can you clarify?"})]


# ---------------------------
# Transition + Likelihood + Cost
# ---------------------------

class SimpleTransition:
    def transition(self, state: Dict[str, Any], action: Action, obs: Optional[Observation]) -> Dict[str, Any]:
        s = dict(state)
        s["last_action"] = action.kind
        s["last_payload"] = dict(action.payload)
        if obs is not None:
            s["last_obs"] = {"kind": obs.kind, "payload": dict(obs.payload)}
        return s


class ToolSuccessLikelihood:
    def __init__(self, tool_bonus: float = 2.0, tool_fail_penalty: float = 3.0, answer_penalty: float = 0.25):
        self.tool_bonus = tool_bonus
        self.tool_fail_penalty = tool_fail_penalty
        self.answer_penalty = answer_penalty

    def loglik(self, prev_state: Dict[str, Any], action: Action, obs: Optional[Observation], next_state: Dict[str, Any]) -> float:
        if action.kind == "TOOL":
            if obs and obs.payload.get("ok") is True:
                return +self.tool_bonus
            return -self.tool_fail_penalty
        if action.kind == "ANSWER":
            return -self.answer_penalty
        return 0.0


class SimpleCost:
    def __init__(self, tool_cost: float = 0.2):
        self.tool_cost = tool_cost

    def cost(self, prev_state: Dict[str, Any], action: Action, obs: Optional[Observation], next_state: Dict[str, Any]) -> float:
        return self.tool_cost if action.kind == "TOOL" else 0.0


# ---------------------------
# Main Wrapper
# ---------------------------

@dataclass
class BayesianLCConfig:
    n_particles: int = 16
    max_steps: int = 2
    ess_resample_frac: float = 0.5
    temperature: float = 1.0
    tool_cost: float = 0.2


class BayesianLangChain:
    def __init__(self, llm: Any, tools: Dict[str, Any], cfg: BayesianLCConfig = BayesianLCConfig()):
        self.llm = llm
        self.tools = tools
        self.cfg = cfg

        proposer = JSONActionProposer(llm, list(tools.keys()))
        observer = LangChainToolObserver(tools)

        self.agent = SMCAgent(
            proposer=proposer,
            observer=observer,
            transition=SimpleTransition(),
            obs_likelihood=ToolSuccessLikelihood(),
            cost_model=SimpleCost(tool_cost=cfg.tool_cost),
            cfg=SMCConfig(
                n_particles=cfg.n_particles,
                max_steps=cfg.max_steps,
                ess_resample_frac=cfg.ess_resample_frac
            ),
            temperature=cfg.temperature,
        )

    def run(self, user_msg: str, init_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        res = self.agent.run(user_msg, init_state=init_state or {})
        best = res.best
        return {
            "best_action": None if best is None else best.state.get("last_action"),
            "best_payload": None if best is None else best.state.get("last_payload"),
            "best_obs": None if best is None else best.state.get("last_obs"),
            "posterior_probs": res.belief.probs(),
            "meta": res.meta,
        }

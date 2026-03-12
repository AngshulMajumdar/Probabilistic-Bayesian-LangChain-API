from __future__ import annotations

import sys, random
from pathlib import Path

ROOT = Path("/content")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from b_langchain.runtime.smc_agent import SMCAgent, SMCConfig
from b_langchain.runtime.minimal_impl import (
    SimpleProposer, DictToolObserver, IdentityTransition, SimpleObsLikelihood, SimpleCost
)

class FlakyEchoTool:
    def __init__(self, p_fail: float = 0.35):
        self.p_fail = float(p_fail)

    def invoke(self, inp):
        if random.random() < self.p_fail:
            raise RuntimeError("simulated_tool_failure")
        return {"echo": inp}

tools = {"echo": FlakyEchoTool(p_fail=0.35)}

agent = SMCAgent(
    proposer=SimpleProposer(tool_name="echo"),
    observer=DictToolObserver(tools),
    transition=IdentityTransition(),
    obs_likelihood=SimpleObsLikelihood(tool_bonus=2.0, tool_fail_penalty=3.0),
    cost_model=SimpleCost(tool_cost=0.2),
    cfg=SMCConfig(n_particles=32, max_steps=3, ess_resample_frac=0.5),
    temperature=1.0,
)

out = agent.run("hello world", init_state={"demo": True})

best = out.best.state if out.best else None
print("BEST STATE:")
print(best)

ps = out.belief.probs()
print("\nTOP-10 PROBS:")
print(sorted(ps, reverse=True)[:10])

print("\nESS-ish (rough):", 1.0 / sum(p*p for p in ps))
print("META:", out.meta)

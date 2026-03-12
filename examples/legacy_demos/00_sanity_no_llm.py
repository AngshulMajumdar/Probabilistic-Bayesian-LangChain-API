from __future__ import annotations

import sys
from pathlib import Path

# Make /content importable (so `import b_langchain` works)
ROOT = Path("/content")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from b_langchain.runtime.smc_agent import SMCAgent, SMCConfig
from b_langchain.runtime.minimal_impl import (
    SimpleProposer, DictToolObserver, IdentityTransition, SimpleObsLikelihood, SimpleCost
)

class EchoTool:
    def invoke(self, inp):
        return {"echo": inp}

tools = {"echo": EchoTool()}

agent = SMCAgent(
    proposer=SimpleProposer(tool_name="echo"),
    observer=DictToolObserver(tools),
    transition=IdentityTransition(),
    obs_likelihood=SimpleObsLikelihood(tool_bonus=2.0, tool_fail_penalty=3.0),
    cost_model=SimpleCost(tool_cost=0.2),
    cfg=SMCConfig(n_particles=16, max_steps=2, ess_resample_frac=0.5),
    temperature=1.0,
)

out = agent.run("hello world", init_state={"demo": True})

print("BEST STATE:")
print(out.best.state if out.best else None)
print("\nPOSTERIOR PROBS:")
print(out.belief.probs())
print("\nMETA:")
print(out.meta)

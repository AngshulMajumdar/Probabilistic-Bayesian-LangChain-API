from __future__ import annotations

import sys, random
from pathlib import Path
from collections import Counter

# ---------------- config ----------------
P_FAIL = 0.6
SEED = 0              # set None to disable determinism
N_PARTICLES = 64
MAX_STEPS = 3
# ---------------------------------------

if SEED is not None:
    random.seed(SEED)

ROOT = Path("/content")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from b_langchain.runtime.smc_agent import SMCAgent, SMCConfig
from b_langchain.runtime.minimal_impl import (
    SimpleProposer, DictToolObserver, IdentityTransition, SimpleObsLikelihood, SimpleCost
)

import b_langchain.runtime.smc_agent as smc_mod
print("USING smc_agent FROM:", smc_mod.__file__)


class FlakyEchoTool:
    def __init__(self, p_fail: float = P_FAIL):
        self.p_fail = float(p_fail)

    def invoke(self, inp):
        if random.random() < self.p_fail:
            raise RuntimeError("simulated_tool_failure")
        return {"echo": inp}


tools = {"echo": FlakyEchoTool(p_fail=P_FAIL)}

agent = SMCAgent(
    proposer=SimpleProposer(tool_name="echo"),
    observer=DictToolObserver(tools),
    transition=IdentityTransition(),
    obs_likelihood=SimpleObsLikelihood(tool_bonus=2.0, tool_fail_penalty=3.0),
    cost_model=SimpleCost(tool_cost=0.2),
    # resampling OFF:
    cfg=SMCConfig(n_particles=N_PARTICLES, max_steps=MAX_STEPS, ess_resample_frac=0.0),
    temperature=1.0,
)

out = agent.run("hello world", init_state={"demo": True})

# ---- inspect log-weights ----
logws = [p.logw for p in out.belief.particles]
rounded = [round(x, 6) for x in logws]
uniq = sorted(set(rounded))

print("\nUNIQUE logw VALUES (rounded):", uniq)
print("COUNTS BY logw:", dict(sorted(Counter(rounded).items(), key=lambda kv: kv[0])))

# ---- inspect posterior ----
ps = out.belief.probs()
top10 = sorted(ps, reverse=True)[:10]
print("\nTOP-10 PROBS:", top10)
print("MIN/MAX PROB:", min(ps), max(ps))

den = sum(p*p for p in ps)
ess = (1.0 / den) if den > 0 else 0.0
print("ESS-ish:", ess)

print("META:", out.meta)

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import random
import math

from b_langchain.core.types import Particle, Belief, TraceEvent
from b_langchain.core.utils import normalize_logweights, effective_sample_size, systematic_resample
from b_langchain.runtime.interfaces import (
    Action, Observation, ActionProposer, TransitionModel, Observer, ObsLikelihood, CostModel
)


@dataclass
class SMCConfig:
    n_particles: int = 16
    max_steps: int = 3
    ess_resample_frac: float = 0.5


@dataclass
class SMCResult:
    belief: Belief
    best: Optional[Particle]
    meta: Dict[str, float]


def _normalize(particles: List[Particle]) -> List[float]:
    ps = normalize_logweights([p.logw for p in particles])
    for p, pr in zip(particles, ps):
        p.state["_p"] = pr
    return ps


def _sample_index_from_logweights(logws: List[float]) -> int:
    m = max(logws)
    ws = [math.exp(lw - m) for lw in logws]
    z = sum(ws)
    if z <= 0.0 or not math.isfinite(z):
        return random.randrange(len(logws))
    r = random.random() * z
    acc = 0.0
    for i, w in enumerate(ws):
        acc += w
        if r <= acc:
            return i
    return len(logws) - 1


class SMCAgent:
    # SMC (particle-filter) wrapper over action choices + noisy observations (tools).

    def __init__(
        self,
        proposer: ActionProposer,
        observer: Observer,
        transition: TransitionModel,
        obs_likelihood: ObsLikelihood,
        cost_model: CostModel,
        cfg: SMCConfig = SMCConfig(),
        temperature: float = 1.0,
    ):
        self.proposer = proposer
        self.observer = observer
        self.transition = transition
        self.obs_likelihood = obs_likelihood
        self.cost_model = cost_model
        self.cfg = cfg
        self.temperature = max(float(temperature), 1e-6)

    def run(self, user_msg: str, init_state: Optional[Dict] = None) -> SMCResult:
        init_state = dict(init_state or {})
        particles = [Particle(state=dict(init_state), logw=0.0) for _ in range(self.cfg.n_particles)]
        resamples = 0.0

        for t in range(self.cfg.max_steps):
            _normalize(particles)

            new_particles: List[Particle] = []
            for p in particles:
                actions = self.proposer.propose(user_msg, p)
                if not actions:
                    actions = [Action(kind="CLARIFY", payload={"question": "Can you clarify?"})]

                # prior over proposals (cheap): slight penalty for TOOL to avoid overuse
                prior_logws: List[float] = []
                for a in actions:
                    prior = -0.3 if a.kind == "TOOL" else 0.0
                    prior_logws.append(prior / self.temperature)

                chosen = actions[_sample_index_from_logweights(prior_logws)]

                obs = self.observer.observe(chosen)
                next_state = self.transition.transition(p.state, chosen, obs)

                ll = float(self.obs_likelihood.loglik(p.state, chosen, obs, next_state))
                c  = float(self.cost_model.cost(p.state, chosen, obs, next_state))

                q = Particle(state=next_state, logw=p.logw + ll - c, trace=list(p.trace))
                q.trace.append(TraceEvent(kind="step", message=f"t={t}", data={
                    "action": {"kind": chosen.kind, "payload": chosen.payload},
                    "obs": None if obs is None else {"kind": obs.kind, "payload": obs.payload},
                    "ll": ll, "cost": c
                }))
                new_particles.append(q)

            particles = new_particles
            ps = _normalize(particles)

            # IMPORTANT: do not resample on the final step, or we erase the posterior.
            if t < self.cfg.max_steps - 1:
                ess = effective_sample_size(ps)
                if ess < self.cfg.ess_resample_frac * self.cfg.n_particles:
                    idxs = systematic_resample(ps, self.cfg.n_particles)
                    particles = [
                        Particle(state=dict(particles[i].state), logw=0.0, trace=list(particles[i].trace))
                        for i in idxs
                    ]
                    resamples += 1.0

        belief = Belief(particles).normalize()
        return SMCResult(
            belief=belief,
            best=belief.best(),
            meta={"n_particles": float(self.cfg.n_particles), "resamples": resamples},
        )

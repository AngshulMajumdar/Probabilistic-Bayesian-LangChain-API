from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import math


@dataclass
class TraceEvent:
    kind: str
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Particle:
    state: Dict[str, Any] = field(default_factory=dict)
    logw: float = 0.0
    trace: List[TraceEvent] = field(default_factory=list)

    def copy(self) -> "Particle":
        return Particle(state=dict(self.state), logw=float(self.logw), trace=list(self.trace))


@dataclass
class Belief:
    particles: List[Particle] = field(default_factory=list)

    def normalize(self, key: str = "_p") -> "Belief":
        if not self.particles:
            return Belief([])
        logws = [p.logw for p in self.particles]
        m = max(logws)
        exps = [math.exp(lw - m) for lw in logws]
        z = sum(exps)

        out: List[Particle] = []
        if z <= 0.0 or not math.isfinite(z):
            u = 1.0 / len(self.particles)
            for p in self.particles:
                q = p.copy()
                q.state[key] = u
                out.append(q)
            return Belief(out)

        for p, e in zip(self.particles, exps):
            q = p.copy()
            q.state[key] = e / z
            out.append(q)
        return Belief(out)

    def probs(self, key: str = "_p") -> List[float]:
        return [float(p.state.get(key, 0.0)) for p in self.particles]

    def best(self, key: str = "_p") -> Optional[Particle]:
        if not self.particles:
            return None
        return max(self.particles, key=lambda p: float(p.state.get(key, 0.0)))

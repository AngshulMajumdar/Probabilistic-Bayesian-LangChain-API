from __future__ import annotations

import math
import random
from typing import List


def normalize_logweights(logws: List[float]) -> List[float]:
    if not logws:
        return []
    m = max(logws)
    ws = [math.exp(lw - m) for lw in logws]
    z = sum(ws)
    if z <= 0.0 or not math.isfinite(z):
        return [1.0 / len(logws)] * len(logws)
    return [w / z for w in ws]


def effective_sample_size(ps: List[float]) -> float:
    s2 = sum(p * p for p in ps)
    if s2 <= 0.0:
        return 0.0
    return 1.0 / s2


def systematic_resample(ps: List[float], n: int) -> List[int]:
    if n <= 0:
        return []

    cdf = []
    acc = 0.0
    for p in ps:
        acc += p
        cdf.append(acc)
    if cdf and abs(cdf[-1] - 1.0) > 1e-6:
        cdf[-1] = 1.0

    u0 = random.random() / n
    idxs = []
    j = 0
    for i in range(n):
        u = u0 + i / n
        while j < len(cdf) - 1 and u > cdf[j]:
            j += 1
        idxs.append(j)
    return idxs

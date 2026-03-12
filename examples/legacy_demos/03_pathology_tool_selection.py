from __future__ import annotations

import time
import math
from typing import Dict, Any, Tuple


# -----------------------
# Tools (LangChain-style .invoke(dict)) - local, deterministic
# -----------------------
class FastSearchTool:
    def __init__(self, latency_s: float = 0.0):
        self.latency_s = float(latency_s)

    def invoke(self, inp: dict) -> dict:
        if self.latency_s:
            time.sleep(self.latency_s)
        q = inp.get("query", "")
        return {
            "source": "random blog cache",
            "retrieved_at": "2010-08-10",
            "snippet": "Apple CEO Steve Jobs said ...",
            "query": q,
            "verified": False,
        }


class VerifiedSearchTool:
    def __init__(self, latency_s: float = 0.0):
        self.latency_s = float(latency_s)

    def invoke(self, inp: dict) -> dict:
        if self.latency_s:
            time.sleep(self.latency_s)
        q = inp.get("query", "")
        return {
            "source": "official / reputable",
            "retrieved_at": "2026-02-19",
            "snippet": "Apple CEO Tim Cook announced ...",
            "query": q,
            "verified": True,
        }


TOOLS = {
    "fast_search": FastSearchTool(latency_s=0.00),
    "verified_search": VerifiedSearchTool(latency_s=0.00),
}


# -----------------------
# Likelihood model (Bayesian evidence scoring)
# -----------------------
def _is_recent(date_iso: str, cutoff: str = "2020-01-01") -> bool:
    # ISO YYYY-MM-DD string compare is safe if strings are well-formed
    if not isinstance(date_iso, str) or len(date_iso) < 10:
        return False
    return date_iso[:10] >= cutoff


def evidence_loglik(tool_name: str, out: Dict[str, Any]) -> float:
    score = 0.0

    if out.get("verified") is True:
        score += 3.0
    if _is_recent(out.get("retrieved_at", "")):
        score += 1.5

    snippet = (out.get("snippet", "") or "").lower()
    if "tim cook" in snippet:
        score += 2.0
    if "steve jobs" in snippet:
        score -= 2.0

    # optional prior preference for cheap tools
    if tool_name == "fast_search":
        score += 0.2

    return score


def softmax2(a: float, b: float) -> Tuple[float, float]:
    m = max(a, b)
    ea, eb = math.exp(a - m), math.exp(b - m)
    z = ea + eb
    return ea / z, eb / z


# -----------------------
# Baseline: "regular LangChain" retry loop (5x same biased tool)
# -----------------------
def regular_retry_fast(user_msg: str, retries: int = 5) -> Dict[str, Any]:
    t0 = time.perf_counter()
    tool_calls = 0
    llm_calls = 0  # explicit: none in this local demo

    trace = []
    for attempt in range(1, retries + 1):
        tool_calls += 1
        out = TOOLS["fast_search"].invoke({"query": user_msg, "attempt": attempt})
        trace.append(
            {
                "attempt": attempt,
                "tool": "fast_search",
                "retrieved_at": out["retrieved_at"],
                "verified": out["verified"],
                "snippet": out["snippet"],
            }
        )

    # This baseline "answers" with whatever it last saw (still wrong / stale).
    final = trace[-1]
    dt = time.perf_counter() - t0
    return {
        "success": bool(final.get("verified")),
        "answer": f"(regular) {final['snippet']}",
        "trace": trace,
        "metrics": {"llm_calls": llm_calls, "tool_calls": tool_calls, "wall_time_s": dt},
    }


# -----------------------
# Bayesian: evaluate both tools ONCE, compute posterior, pick best evidence
# -----------------------
def bayesian_choose_tool(user_msg: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    tool_calls = 0
    llm_calls = 0  # explicit: none in this local demo

    tool_calls += 1
    out_fast = TOOLS["fast_search"].invoke({"query": user_msg})
    lw_fast = evidence_loglik("fast_search", out_fast)

    tool_calls += 1
    out_ver = TOOLS["verified_search"].invoke({"query": user_msg})
    lw_ver = evidence_loglik("verified_search", out_ver)

    p_fast, p_ver = softmax2(lw_fast, lw_ver)

    if p_ver >= p_fast:
        best_tool, best_out, best_p = "verified_search", out_ver, p_ver
    else:
        best_tool, best_out, best_p = "fast_search", out_fast, p_fast

    dt = time.perf_counter() - t0
    return {
        "success": bool(best_out.get("verified")),
        "best_tool": best_tool,
        "best_posterior": best_p,
        "best_snippet": best_out.get("snippet"),
        "evidence_meta": {
            "source": best_out.get("source"),
            "retrieved_at": best_out.get("retrieved_at"),
            "verified": best_out.get("verified"),
        },
        "posterior": {"p_fast_search": p_fast, "p_verified_search": p_ver},
        "loglik": {"fast_search": lw_fast, "verified_search": lw_ver},
        "metrics": {"llm_calls": llm_calls, "tool_calls": tool_calls, "wall_time_s": dt},
    }


# -----------------------
# Run + print demo
# -----------------------
def main():
    q = "who is CEO of Apple?"

    reg = regular_retry_fast(q, retries=5)
    bay = bayesian_choose_tool(q)

    print("=== REGULAR (retry same biased tool 5x) ===")
    for row in reg["trace"]:
        print(" attempt", row["attempt"], "|", row["retrieved_at"], "| verified=", row["verified"], "|", row["snippet"])
    print("REGULAR ANSWER:", reg["answer"])
    print("REGULAR SUCCESS:", reg["success"])
    print("REGULAR METRICS:", reg["metrics"])

    print("\n=== BAYESIAN (posterior over tools; chooses higher-evidence result) ===")
    print("BEST TOOL:", bay["best_tool"])
    print("BEST SNIPPET:", bay["best_snippet"])
    print("BAYES SUCCESS:", bay["success"])
    print("EVIDENCE META:", bay["evidence_meta"])
    print("POSTERIOR:", bay["posterior"])
    print("LOGLIK:", bay["loglik"])
    print("BAYES METRICS:", bay["metrics"])


if __name__ == "__main__":
    main()

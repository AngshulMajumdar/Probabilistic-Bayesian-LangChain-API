# demos/demo_structured_json.py
from __future__ import annotations

"""
Friday Demo 2: Structured JSON robustness

Story to tell interviewer:
-------------------------
Baseline: one-shot LLM output often breaks JSON formatting.
Our approach: sample N candidates, score:
  - schema validity (JSON parses)
  - cheap consistency
Then pick the best valid JSON with highest score.

No retrieval, no tools. Just format robustness + probabilistic selection.
"""

import json
import time
from typing import Any, Dict

from p_langchain.llm.colab_ai import ColabGeminiLLM
from p_langchain.proposers.llm_samples import LLMSamplesProposer
from p_langchain.scorers.schema import JsonSchemaScorer
from p_langchain.scorers.consistency import ConsistencyScorer
from p_langchain.runtime import PChain
from p_langchain.runtime.executor import BeamConfig
from p_langchain.runtime.logger import print_trace_summary, print_top_hypotheses
from p_langchain.policies.ask_or_proceed import AskOrProceedPolicy

# --------------------------------------------------
# FAST MODE KNOBS (match Demo 1 approach)
# 1) reduce samples: n = 2 (instead of 6)
# 2) keep only cheap scorers (we already do: consistency + schema)
# --------------------------------------------------
FAST_MODE = True


def build_json_prompt(user_request: str) -> str:
    # Force strict JSON. In practice LLM may still fail: that's the point of the demo.
    return f"""Return ONLY valid JSON. No markdown. No extra text.

Schema:
{{
  "task": "short string",
  "steps": ["step1", "step2", "step3"],
  "confidence": 0.0
}}

User request:
{user_request}
"""


def prompt_fn(h) -> str:
    return build_json_prompt(str(h.state.get("user_request", "")))


def make_chain() -> PChain:
    llm = ColabGeminiLLM()
    n_samples = 2 if FAST_MODE else 6

    proposer = LLMSamplesProposer(
        llm=llm,
        prompt_fn=prompt_fn,
        n=n_samples,
        temperature=0.8,
        max_tokens=220,
        write_to_state_key="llm_text",
    )

    schema = JsonSchemaScorer(
        text_key="llm_text",
        parsed_key="parsed_json",
        valid_reward=4.0,
        invalid_penalty=-8.0,
    )

    consistency = ConsistencyScorer(
        rules=[],
        fail_penalty=-2.0,
        pass_reward=0.0,
        name="consistency",
    )

    policy = AskOrProceedPolicy(entropy_threshold=999.0, min_p_best=0.0)
    beam_cfg = BeamConfig(beam_size=n_samples, max_steps=1)

    return PChain(
        proposers=[proposer],
        scorers=[consistency, schema],
        policy=policy,
        beam_config=beam_cfg,
    )


def run_demo(user_request: str) -> None:
    chain = make_chain()
    action = chain.run(init_state={"user_request": user_request})

    print("\n==============================")
    print("USER REQUEST:")
    print(user_request)
    print("==============================\n")

    print("POLICY DECISION:", type(action).__name__)
    result = getattr(action, "result", None)

    if result is None:
        print("No PosteriorResult attached (unexpected).")
        return

    print_trace_summary(result)
    print_top_hypotheses(result.posterior, k=3, show_state_keys=["_p", "llm_text"])

    best = result.best
    if best is None:
        print("\nNo best hypothesis.")
        return

    print("\n--- BEST RAW OUTPUT ---")
    print(best.state.get("llm_text", ""))

    txt = best.state.get("llm_text", "")
    try:
        obj = json.loads(txt)
        print("\n--- PARSED JSON OBJECT ---")
        print(obj)
    except Exception as e:
        print("\n!!! Best output still not valid JSON. Error:", e)


def run_demo_metrics(
    user_request: str = "Plan my 1-hour study session for linear algebra.",
) -> dict:
    """
    Metrics-only version for benchmarking.
    Returns a dict aligned with LC demo metrics.
    Note: LLM call count is estimated from proposer sampling (n per run).
    """
    t0 = time.perf_counter()

    chain = make_chain()
    action = chain.run(init_state={"user_request": user_request})

    n_samples = 2 if FAST_MODE else 6
    llm_calls_estimated = n_samples  # proposer n samples, no extra LLM scorers here

    valid_json = False
    result = getattr(action, "result", None)
    if result is not None and result.best is not None:
        txt = result.best.state.get("llm_text", "")
        try:
            json.loads(txt)
            valid_json = True
        except Exception:
            valid_json = False

    wall = time.perf_counter() - t0

    return {
        "mode": "probabilistic_fast" if FAST_MODE else "probabilistic",
        "llm_calls": llm_calls_estimated,
        "wall_time_sec": round(wall, 3),
        "valid_json": valid_json,
        "fast_mode": FAST_MODE,
        "n_samples": n_samples,
    }


if __name__ == "__main__":
    demo_request = "Plan my 1-hour study session for linear algebra."

    print(f"\n[CONFIG] FAST_MODE={FAST_MODE}  (n={'2' if FAST_MODE else '6'})\n")
    run_demo(demo_request)

    print("\n-----------------------------")
    print("METRICS:")
    print(run_demo_metrics(demo_request))

from __future__ import annotations

import sys
import time

# Ensure imports work when run as a script
ROOT = "/content"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from p_langchain.llm.colab_ai import ColabGeminiLLM
from p_langchain.proposers.llm_samples import LLMSamplesProposer
from p_langchain.scorers.consistency import ConsistencyScorer, default_rules_for_demo
from p_langchain.scorers.critique import CritiqueScorer
from p_langchain.runtime import PChain
from p_langchain.runtime.executor import BeamConfig
from p_langchain.runtime.logger import print_trace_summary, print_top_hypotheses
from p_langchain.policies.ask_or_proceed import AskOrProceedPolicy

# --------------------------------------------------
# FAST MODE KNOBS (as requested)
# 1) n = 2 samples (instead of 5)
# 2) remove CritiqueScorer (keep ConsistencyScorer only)
# --------------------------------------------------
FAST_MODE = True


def build_prompt(user_request: str) -> str:
    return f"""You are an assistant inside a probabilistic runtime.
The user message may be ambiguous. Produce ONE plausible interpretation and response.

If the user message contains the substring "Clarification:", then DO NOT ask further questions.
Set NEEDS_CLARIFICATION: no and produce FINAL_ANSWER.

Output format (plain text, no markdown):
- First line: INTENT: <one of: explain | write_email | summarize | code | other>
- Second line: NEEDS_CLARIFICATION: <yes/no>
- Third line:
    If NEEDS_CLARIFICATION=yes -> CLARIFYING_QUESTION: <one short question>
    Else -> FINAL_ANSWER: <your best answer>

User message:
{user_request}
"""


def prompt_fn_from_state(h) -> str:
    return build_prompt(str(h.state.get("user_request", "")))


def make_chain(force_proceed: bool = False) -> PChain:
    llm = ColabGeminiLLM()

    n_samples = 2 if FAST_MODE else 5

    proposer = LLMSamplesProposer(
        llm=llm,
        prompt_fn=prompt_fn_from_state,
        n=n_samples,
        temperature=0.8,
        max_tokens=300,
        write_to_state_key="llm_text",
    )

    consistency = ConsistencyScorer(
        rules=default_rules_for_demo(),
        fail_penalty=-6.0,
        pass_reward=0.0,
    )

    scorers = [consistency]

    # Only include critique scorer in non-fast mode
    if not FAST_MODE:
        critique = CritiqueScorer(
            llm=llm,
            user_request_key="user_request",
            candidate_key="llm_text",
            temperature=0.2,
            max_tokens=220,
            scale=3.0,
        )
        scorers.append(critique)

    if not force_proceed:
        policy = AskOrProceedPolicy(
            entropy_threshold=1.1,
            min_p_best=0.55,
            default_question="Quick clarification: what exactly do you want me to do (explain / write / summarize / code)?",
        )
    else:
        # After one clarification, never ask again (for clean demo).
        policy = AskOrProceedPolicy(
            entropy_threshold=999.0,
            min_p_best=0.0,
            default_question="",
        )

    return PChain(
        proposers=[proposer],
        scorers=scorers,
        policy=policy,
        beam_config=BeamConfig(beam_size=n_samples, max_steps=1),
    )


def run_once(chain: PChain, user_request: str, show_top_k: int = 3):
    action = chain.run(init_state={"user_request": user_request})

    print("\n==============================")
    print("USER REQUEST:")
    print(user_request)
    print("==============================\n")
    print("POLICY DECISION:", type(action).__name__)
    print("POLICY DATA:", getattr(action, "data", {}))

    res = getattr(action, "result", None)
    if res is not None:
        print_trace_summary(res)
        print_top_hypotheses(res.posterior, k=show_top_k, show_state_keys=["_p", "llm_text"])

    return action


def run_demo_metrics(
    user_request: str = "Can you do this for me quickly?",
    clarification_text: str = "Explain in simple words what you will do, step-by-step.",
) -> dict:
    """
    Metrics-only version for benchmarking.
    Returns a dict aligned with LC demo metrics.
    Note: LLM call count is estimated from proposer sampling (n per pass).
    """
    t0 = time.perf_counter()

    clarification_triggered = False
    final_output_produced = False
    llm_calls_estimated = 0

    n_samples = 2 if FAST_MODE else 5

    # Pass 1
    chain1 = make_chain(force_proceed=False)
    a1 = chain1.run(init_state={"user_request": user_request})
    llm_calls_estimated += n_samples

    if type(a1).__name__ == "AskAction":
        clarification_triggered = True

        # Pass 2
        req2 = f"{user_request}\n\nClarification: {clarification_text}"
        chain2 = make_chain(force_proceed=True)
        a2 = chain2.run(init_state={"user_request": req2})
        llm_calls_estimated += n_samples

        if type(a2).__name__ == "ProceedAction":
            res2 = getattr(a2, "result", None)
            if res2 and res2.best:
                final_output_produced = True

    elif type(a1).__name__ == "ProceedAction":
        res1 = getattr(a1, "result", None)
        if res1 and res1.best:
            final_output_produced = True

    wall = time.perf_counter() - t0

    return {
        "mode": "probabilistic_fast" if FAST_MODE else "probabilistic",
        "llm_calls": llm_calls_estimated,
        "wall_time_sec": round(wall, 3),
        "clarification_triggered": clarification_triggered,
        "final_output_produced": final_output_produced,
        "fast_mode": FAST_MODE,
        "n_samples": n_samples,
        "critique_enabled": (not FAST_MODE),
    }


def main():
    # You edit this ONE LINE before the interview (acts like the "clarification input")
    CLARIFICATION_FOR_DEMO = "Explain in simple words what you will do, step-by-step."
    req1 = "Can you do this for me quickly?"

    print(f"\n[CONFIG] FAST_MODE={FAST_MODE}  (n={'2' if FAST_MODE else '5'}, critique={'off' if FAST_MODE else 'on'})\n")

    # Pass 1: show that the runtime detects ambiguity and asks ONE clarification.
    chain1 = make_chain(force_proceed=False)
    a1 = run_once(chain1, req1)

    if type(a1).__name__ == "AskAction":
        print("\n>>> CLARIFYING QUESTION:")
        print(a1.question)

        # Pass 2: show completion (no second asking)
        req2 = f"{req1}\n\nClarification: {CLARIFICATION_FOR_DEMO}"
        print("\n>>> CLARIFICATION (demo input):")
        print(CLARIFICATION_FOR_DEMO)

        chain2 = make_chain(force_proceed=True)
        a2 = run_once(chain2, req2)

        if type(a2).__name__ == "ProceedAction":
            res2 = getattr(a2, "result", None)
            if res2 and res2.best:
                print("\n>>> FINAL OUTPUT:")
                print(res2.best.state.get("llm_text", ""))
        else:
            print("\n(Unexpected) Did not proceed on pass-2.")
    else:
        # If it proceeds immediately, still show the final output.
        if type(a1).__name__ == "ProceedAction":
            res = getattr(a1, "result", None)
            if res and res.best:
                print("\n>>> FINAL OUTPUT:")
                print(res.best.state.get("llm_text", ""))


if __name__ == "__main__":
    main()

    print("\n-----------------------------")
    print("METRICS:")
    print(run_demo_metrics())

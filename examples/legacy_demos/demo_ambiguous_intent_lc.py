from __future__ import annotations

import sys
import json
import time

ROOT = "/content"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler


class CountCalls(BaseCallbackHandler):
    def __init__(self):
        self.calls = 0

    def on_llm_start(self, *args, **kwargs):
        self.calls += 1

    def on_chat_model_start(self, *args, **kwargs):
        self.calls += 1


answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Follow the user's request."),
    ("human", "{user_request}")
])

# IMPORTANT: escape braces for LangChain templating
decide_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Decide if the user request is ambiguous enough to require ONE clarifying question.\n"
     "Return ONLY JSON in this format:\n"
     "{{\"need_clarification\": true/false, \"question\": \"...\"}}"
    ),
    ("human", "{user_request}")
])


def run_demo(
    llm,
    user_request: str = "Can you do this for me quickly?",
    clarification_text: str = "Explain in simple words what you will do, step-by-step.",
    use_clarification_step: bool = True,
    verbose: bool = True,
):
    counter = CountCalls()
    t0 = time.perf_counter()

    clarification_triggered = False
    final_output_produced = False

    if not use_clarification_step:
        chain = answer_prompt | llm | StrOutputParser()
        output = chain.invoke({"user_request": user_request}, config={"callbacks": [counter]})
        final_output_produced = True
        mode = "Baseline A (one-shot)"
    else:
        decide_chain = decide_prompt | llm | StrOutputParser()
        raw = decide_chain.invoke({"user_request": user_request}, config={"callbacks": [counter]})

        try:
            decision = json.loads(raw)
        except Exception:
            decision = {"need_clarification": False, "question": ""}

        if decision.get("need_clarification"):
            clarification_triggered = True
            req2 = f"{user_request}\n\nClarification: {clarification_text}"
            answer_chain = answer_prompt | llm | StrOutputParser()
            output = answer_chain.invoke({"user_request": req2}, config={"callbacks": [counter]})
        else:
            answer_chain = answer_prompt | llm | StrOutputParser()
            output = answer_chain.invoke({"user_request": user_request}, config={"callbacks": [counter]})

        final_output_produced = True
        mode = "Baseline B (one clarification policy)"

    wall = time.perf_counter() - t0

    metrics = {
        "mode": mode,
        "llm_calls": counter.calls,
        "wall_time_sec": round(wall, 3),
        "clarification_triggered": clarification_triggered,
        "final_output_produced": final_output_produced,
    }

    if verbose:
        print("\n==============================")
        print("USER REQUEST:")
        print(user_request)
        print("==============================\n")
        print("MODE:", mode)
        print("CLARIFICATION TRIGGERED:", clarification_triggered)
        print("LLM CALLS:", counter.calls)
        print("WALL TIME (sec):", round(wall, 3))
        print("\nFINAL OUTPUT:\n")
        print(output)

    return metrics


if __name__ == "__main__":
    from p_langchain.llm.colab_gemini_langchain_wrapper import ColabGeminiChatModel

    llm = ColabGeminiChatModel(temperature=0.0, max_tokens=512)

    print("\n=== Regular LangChain — Baseline A ===")
    run_demo(llm=llm, use_clarification_step=False)

    print("\n--------------------------------------\n")

    print("=== Regular LangChain — Baseline B ===")
    run_demo(llm=llm, use_clarification_step=True)

from __future__ import annotations

import sys
import json
import time

# Force /content on sys.path so imports work when run as a script
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


# IMPORTANT: escape braces for LangChain templating
json_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Return ONLY valid JSON. No markdown. No extra text.\n\n"
     "Schema:\n"
     "{{\n"
     "  \"task\": \"short string\",\n"
     "  \"steps\": [\"step1\",\"step2\",\"step3\"],\n"
     "  \"confidence\": 0.0\n"
     "}}\n"),
    ("human", "{user_request}")
])

fix_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You will be given text that is supposed to be JSON for this schema:\n"
     "{{ \"task\": \"short string\", \"steps\": [\"step1\",\"step2\",\"step3\"], \"confidence\": 0.0 }}\n\n"
     "Return ONLY corrected valid JSON. No extra text."),
    ("human", "Bad JSON:\n{bad}\n\nUser request:\n{user_request}")
])


def run_demo(
    llm,
    user_request: str = "Plan my 1-hour study session for linear algebra.",
    use_retry_fix: bool = True,
    verbose: bool = True,
):
    counter = CountCalls()
    t0 = time.perf_counter()

    # First attempt
    chain = json_prompt | llm | StrOutputParser()
    output = chain.invoke({"user_request": user_request}, config={"callbacks": [counter]})

    valid = False
    parsed_obj = None

    try:
        parsed_obj = json.loads(output)
        valid = True
    except Exception:
        valid = False

    mode = "Baseline A (one-shot)"

    # Retry fix if enabled
    if not valid and use_retry_fix:
        mode = "Baseline B (retry fix)"
        fix_chain = fix_prompt | llm | StrOutputParser()
        fixed = fix_chain.invoke({"bad": output, "user_request": user_request}, config={"callbacks": [counter]})
        try:
            parsed_obj = json.loads(fixed)
            valid = True
        except Exception:
            valid = False

    wall = time.perf_counter() - t0

    metrics = {
        "mode": mode,
        "llm_calls": counter.calls,
        "wall_time_sec": round(wall, 3),
        "valid_json": valid,
    }

    if verbose:
        print("\n==============================")
        print("USER REQUEST:")
        print(user_request)
        print("==============================\n")
        print("MODE:", mode)
        print("VALID JSON:", valid)
        print("LLM CALLS:", counter.calls)
        print("WALL TIME (sec):", round(wall, 3))
        if valid:
            print("\nParsed Object:")
            print(parsed_obj)

    return metrics


if __name__ == "__main__":
    # Wrapper must exist at: /content/p_langchain/llm/colab_gemini_langchain_wrapper.py
    from p_langchain.llm.colab_gemini_langchain_wrapper import ColabGeminiChatModel

    llm = ColabGeminiChatModel(temperature=0.0, max_tokens=512)

    print("\n=== Regular LangChain — Baseline A ===")
    run_demo(llm=llm, use_retry_fix=False)

    print("\n--------------------------------------\n")

    print("=== Regular LangChain — Baseline B ===")
    run_demo(llm=llm, use_retry_fix=True)

from __future__ import annotations

# IMPORTANT:
# This file must be executed INSIDE the Colab notebook kernel, NOT via `!python`,
# because `google.colab.ai` needs the active IPython kernel.

from google.colab import ai

_ = ai.generate_text("OK")  # warm-up
print("✅ Colab AI warmed up.")

from b_langchain.agents.bayesian_lc import BayesianLangChain, BayesianLCConfig


class EchoTool:
    def invoke(self, inp):
        return {"echo": inp}

tools = {"echo": EchoTool()}


class ColabGeminiLLM:
    # Provides llm.invoke(prompt).content
    def invoke(self, prompt: str):
        txt = ai.generate_text(prompt)
        class R:
            def __init__(self, content: str):
                self.content = content
        return R(txt)


llm = ColabGeminiLLM()

cfg = BayesianLCConfig(
    n_particles=16,
    max_steps=2,
    ess_resample_frac=0.1,
    temperature=1.0,
    tool_cost=0.2,
)

agent = BayesianLangChain(llm=llm, tools=tools, cfg=cfg)

out = agent.run(
    "Call the echo tool with text='hello'. Return ONLY JSON action.",
    init_state={"demo": True}
)
print(out)

from __future__ import annotations

from typing import Dict, Any

from bayesian_prob_langchain_api.services.container import ServiceContainer


class RagService:
    def __init__(self, container: ServiceContainer):
        self.container = container

    def query(self, text: str, top_k: int = 3) -> Dict[str, Any]:
        retriever = self.container.tools["retriever"]
        output = retriever.invoke({"query": text})
        hits = output["hits"][:top_k]
        answer = " ".join(hit["text"] for hit in hits) if hits else "No relevant documents found."
        return {"answer": answer, "hits": hits}

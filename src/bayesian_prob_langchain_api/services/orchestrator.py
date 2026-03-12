from __future__ import annotations

from typing import Any, Dict

from bayesian_prob_langchain_api.services.container import ServiceContainer


class OrchestrationService:
    def __init__(self, container: ServiceContainer):
        self.container = container

    def run(self, query: str, backend: str = "local", init_state: Dict[str, Any] | None = None) -> Dict[str, Any]:
        agent = self.container.make_agent(backend=backend)
        return agent.run(query, init_state=init_state or {})

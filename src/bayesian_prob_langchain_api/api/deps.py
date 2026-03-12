from __future__ import annotations

from functools import lru_cache

from bayesian_prob_langchain_api.services.container import ServiceContainer
from bayesian_prob_langchain_api.services.orchestrator import OrchestrationService
from bayesian_prob_langchain_api.services.rag import RagService


@lru_cache(maxsize=1)
def get_container() -> ServiceContainer:
    return ServiceContainer()


def get_orchestration_service() -> OrchestrationService:
    return OrchestrationService(get_container())


def get_rag_service() -> RagService:
    return RagService(get_container())

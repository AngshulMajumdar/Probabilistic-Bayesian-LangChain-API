from __future__ import annotations

from fastapi import APIRouter

from bayesian_prob_langchain_api.api.deps import get_container
from bayesian_prob_langchain_api.config import settings
from bayesian_prob_langchain_api.schemas import HealthResponse, InfoResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    container = get_container()
    return HealthResponse(
        ok=True,
        service=settings.app_name,
        version=settings.app_version,
        available_backends=container.available_backends,
    )


@router.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    return InfoResponse(
        name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        api_prefix="/api/v1",
    )

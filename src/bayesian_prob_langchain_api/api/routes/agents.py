from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from bayesian_prob_langchain_api.api.deps import get_orchestration_service
from bayesian_prob_langchain_api.schemas import RunRequest, RunResponse
from bayesian_prob_langchain_api.services.orchestrator import OrchestrationService

router = APIRouter(tags=["agents"])


@router.post("/agents/run", response_model=RunResponse)
def run_agent(
    request: RunRequest,
    service: OrchestrationService = Depends(get_orchestration_service),
) -> RunResponse:
    try:
        result = service.run(request.query, backend=request.backend, init_state=request.init_state)
        return RunResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

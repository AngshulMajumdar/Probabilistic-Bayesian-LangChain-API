from __future__ import annotations

from fastapi import APIRouter

from bayesian_prob_langchain_api.api.deps import get_container
from bayesian_prob_langchain_api.schemas import ToolInfoResponse

router = APIRouter(tags=["tools"])


@router.get("/tools", response_model=ToolInfoResponse)
def list_tools() -> ToolInfoResponse:
    return ToolInfoResponse(tools=sorted(get_container().tools.keys()))

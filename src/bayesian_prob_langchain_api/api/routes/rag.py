from __future__ import annotations

from fastapi import APIRouter, Depends

from bayesian_prob_langchain_api.api.deps import get_rag_service
from bayesian_prob_langchain_api.schemas import RagRequest, RagResponse
from bayesian_prob_langchain_api.services.rag import RagService

router = APIRouter(tags=["rag"])


@router.post("/rag/query", response_model=RagResponse)
def rag_query(
    request: RagRequest,
    service: RagService = Depends(get_rag_service),
) -> RagResponse:
    return RagResponse(**service.query(request.query, top_k=request.top_k))

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    ok: bool
    service: str
    version: str
    available_backends: List[str]


class InfoResponse(BaseModel):
    name: str
    version: str
    environment: str
    api_prefix: str


class ToolInfoResponse(BaseModel):
    tools: List[str]


class RunRequest(BaseModel):
    query: str = Field(..., min_length=1)
    backend: str = Field(default="local")
    init_state: Dict[str, Any] = Field(default_factory=dict)


class RunResponse(BaseModel):
    best_action: Optional[str] = None
    best_payload: Optional[Dict[str, Any]] = None
    best_obs: Optional[Dict[str, Any]] = None
    posterior_probs: List[float]
    meta: Dict[str, Any]


class RagRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class RagHit(BaseModel):
    id: str
    text: str
    score: int


class RagResponse(BaseModel):
    answer: str
    hits: List[RagHit]

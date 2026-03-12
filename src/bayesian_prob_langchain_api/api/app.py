from __future__ import annotations

from fastapi import FastAPI

from bayesian_prob_langchain_api.api.routes.health import router as health_router
from bayesian_prob_langchain_api.api.routes.tools import router as tools_router
from bayesian_prob_langchain_api.api.routes.agents import router as agents_router
from bayesian_prob_langchain_api.api.routes.rag import router as rag_router
from bayesian_prob_langchain_api.config import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url=settings.docs_url,
        openapi_url=settings.openapi_url,
    )
    prefix = "/api/v1"
    app.include_router(health_router, prefix=prefix)
    app.include_router(tools_router, prefix=prefix)
    app.include_router(agents_router, prefix=prefix)
    app.include_router(rag_router, prefix=prefix)
    return app


app = create_app()

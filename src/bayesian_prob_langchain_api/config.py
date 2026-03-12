from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    app_name: str = "Bayesian Probabilistic LangChain API"
    app_version: str = "0.2.0"
    docs_url: str = "/docs"
    openapi_url: str = "/openapi.json"
    environment: str = os.getenv("APP_ENV", "dev")


settings = Settings()

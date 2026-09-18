"""FastAPI dependency injection providers."""

from typing import Annotated
from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.llm import PerceptionPipeline, get_perception_pipeline
from app.policies.refund import RefundPolicyConfig
from app.repositories.triage import TriageRepository
from app.services.triage import TriageService


def get_app_settings(request: Request) -> Settings:
    """Retrieve application settings from request app state or cached singleton."""
    return getattr(request.app.state, "settings", None) or get_settings()


def get_repository(
    settings: Annotated[Settings, Depends(get_app_settings)]
) -> TriageRepository:
    """Provide initialized TriageRepository instance."""
    repo = TriageRepository(db_path=settings.DATABASE_PATH)
    repo.init_db()
    return repo


def get_pipeline(
    settings: Annotated[Settings, Depends(get_app_settings)]
) -> PerceptionPipeline:
    """Provide configured PerceptionPipeline instance."""
    return get_perception_pipeline(settings=settings)


def get_policy_config() -> RefundPolicyConfig:
    """Provide refund policy configuration rules."""
    return RefundPolicyConfig()


def get_triage_service(
    repository: Annotated[TriageRepository, Depends(get_repository)],
    pipeline: Annotated[PerceptionPipeline, Depends(get_pipeline)],
    policy_config: Annotated[RefundPolicyConfig, Depends(get_policy_config)],
) -> TriageService:
    """Provide fully configured TriageService instance."""
    return TriageService(
        repository=repository,
        pipeline=pipeline,
        policy_config=policy_config,
    )

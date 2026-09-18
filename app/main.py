"""FastAPI application initialization and factory."""

from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI

from app.api.triage import router as triage_router
from app.core.config import Settings, get_settings
from app.repositories.triage import TriageRepository


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Application factory creating and configuring the FastAPI instance.

    Args:
        settings: Optional custom settings instance (useful for test overrides).

    Returns:
        Configured FastAPI application instance.
    """
    if settings is None:
        settings = get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        # Initialize SQLite database and schema on startup
        repo = TriageRepository(db_path=settings.DATABASE_PATH)
        repo.init_db()
        yield

    application = FastAPI(
        title=settings.APP_NAME,
        version="0.1.0",
        description="Automated support-ticket triage microservice for Dhaba",
        docs_url="/docs" if settings.APP_ENV != "production" else None,
        redoc_url="/redoc" if settings.APP_ENV != "production" else None,
        lifespan=lifespan,
    )
    application.state.settings = settings

    # Simple health check endpoint
    @application.get("/health", tags=["health"])
    async def health_check():
        return {"status": "ok", "app": settings.APP_NAME}

    # Mount triage route
    application.include_router(triage_router)

    return application


# Top-level ASGI entry point for uvicorn
app = create_app()

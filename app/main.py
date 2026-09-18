"""FastAPI application initialization and factory."""

from fastapi import FastAPI

from app.core.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory creating and configuring the FastAPI instance.

    Args:
        settings: Optional custom settings instance (useful for test overrides).

    Returns:
        Configured FastAPI application instance.
    """
    if settings is None:
        settings = get_settings()

    application = FastAPI(
        title=settings.APP_NAME,
        version="0.1.0",
        description="Automated support-ticket triage microservice for Dhaba",
        docs_url="/docs" if settings.APP_ENV != "production" else None,
        redoc_url="/redoc" if settings.APP_ENV != "production" else None,
    )

    # Note: Endpoints will be mounted in their respective implementation steps.
    # No routes or placeholder endpoints are defined in Step 1.

    return application


# Top-level ASGI entry point for uvicorn
app = create_app()

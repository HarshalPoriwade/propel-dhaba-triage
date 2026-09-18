"""FastAPI application initialization and factory."""

from contextlib import asynccontextmanager
from typing import Optional
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.triage import router as triage_router
from app.core.config import Settings, get_settings
from app.core.logging import (
    _request_id_ctx,
    get_logger,
    get_request_id,
    sanitize_request_id,
    set_request_id,
)
from app.core.resilience import get_circuit_breaker
from app.repositories.triage import TriageRepository

app_logger = get_logger("dhaba.main")


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

    # Correlation ID middleware ensuring all logs and responses carry X-Request-ID
    @application.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):
        inbound_id = request.headers.get("X-Request-ID") or request.headers.get("X-Correlation-ID")
        request_id = sanitize_request_id(inbound_id)
        if not request_id:
            request_id = f"req_{uuid.uuid4().hex[:16]}"

        token = set_request_id(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            _request_id_ctx.reset(token)

    # Global unhandled exception handler protecting internals from client disclosure
    @application.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        app_logger.error(
            event="api.unhandled_exception",
            message=f"Unhandled internal server error: {type(exc).__name__}",
            error_type=type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected internal error occurred while processing the ticket."},
            headers={"X-Request-ID": get_request_id()},
        )

    # Lightweight health check exposing service readiness and circuit state without calling upstream LLM
    @application.get("/health", tags=["health"])
    async def health_check():
        circuit = get_circuit_breaker(settings)
        return {
            "status": "ok",
            "app": settings.APP_NAME,
            "circuit_state": circuit.state.value,
        }

    # Mount triage route
    application.include_router(triage_router)

    return application


# Top-level ASGI entry point for uvicorn
app = create_app()

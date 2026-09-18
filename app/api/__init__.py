"""API package exposing route routers and dependencies."""

from app.api.triage import router as triage_router

__all__ = ["triage_router"]

"""Strongly typed application configuration using pydantic-settings."""

from functools import lru_cache
from typing import Literal, Optional
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable fallback and strict validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application Identity & Lifecycle
    APP_NAME: str = Field(
        default="Dhaba Support Triage Service",
        description="Name of the service",
    )
    APP_ENV: Literal["development", "staging", "production", "testing"] = Field(
        default="development",
        description="Current deployment environment",
    )

    # LLM Gateway Configuration
    MODEL_MODE: Literal["fixture", "live"] = Field(
        default="fixture",
        description="Mode of operation: 'fixture' for offline replay or 'live' for API integration",
    )
    MODEL_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for live model provider. Optional when MODEL_MODE is fixture",
    )
    MODEL_NAME: str = Field(
        default="gpt-4o-mini",
        description="Model identifier for live inference",
    )
    MODEL_BASE_URL: Optional[str] = Field(
        default=None,
        description="Optional custom base URL for LLM provider (e.g. Azure, Ollama, proxy)",
    )
    MODEL_TIMEOUT_SECONDS: float = Field(
        default=2.5,
        gt=0.0,
        description="Upstream model call timeout in seconds",
    )
    LLM_MAX_CONCURRENCY: int = Field(
        default=20,
        gt=0,
        description="Maximum concurrent outbound calls to LLM provider",
    )

    # Storage & Persistence
    DATABASE_PATH: str = Field(
        default="dhaba_triage.db",
        description="Filesystem path to the SQLite database",
    )

    @field_validator("MODEL_API_KEY", "MODEL_BASE_URL", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: Optional[str]) -> Optional[str]:
        """Convert empty or whitespace-only strings from .env into None."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("DATABASE_PATH")
    @classmethod
    def validate_database_path(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("DATABASE_PATH cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_live_mode_requirements(self) -> "Settings":
        """Ensure an API key is supplied if live model mode is requested."""
        if self.MODEL_MODE == "live" and (not self.MODEL_API_KEY or not self.MODEL_API_KEY.strip()):
            raise ValueError("MODEL_API_KEY is required when MODEL_MODE is set to 'live'")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton accessor for application settings."""
    return Settings()

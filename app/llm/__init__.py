"""LLM package exposing the provider interface, implementations, and factory."""

from typing import Optional

from app.core.config import Settings, get_settings
from app.llm.base import (
    BaseLLMProvider,
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseError,
)
from app.llm.fixture import FixtureLLMProvider
from app.llm.provider import LiveLLMProvider


def get_llm_provider(settings: Optional[Settings] = None) -> BaseLLMProvider:
    """Factory selecting the configured LLM provider based on application settings.

    Args:
        settings: Application settings. If None, resolves from get_settings().

    Returns:
        BaseLLMProvider: Either FixtureLLMProvider or LiveLLMProvider.

    Raises:
        LLMConfigurationError: If configuration is invalid or unsupported.
    """
    if settings is None:
        settings = get_settings()

    if settings.MODEL_MODE == "fixture":
        return FixtureLLMProvider()

    if settings.MODEL_MODE == "live":
        return LiveLLMProvider(
            api_key=settings.MODEL_API_KEY,
            model_name=settings.MODEL_NAME,
            base_url=settings.MODEL_BASE_URL,
            timeout_seconds=settings.MODEL_TIMEOUT_SECONDS,
        )

    raise LLMConfigurationError(f"Unsupported MODEL_MODE: '{settings.MODEL_MODE}'")


__all__ = [
    "BaseLLMProvider",
    "FixtureLLMProvider",
    "LiveLLMProvider",
    "LLMConfigurationError",
    "LLMProviderError",
    "LLMResponseError",
    "get_llm_provider",
]

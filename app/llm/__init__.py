"""LLM package exposing provider interfaces, implementations, factory, parser, validation, and pipeline."""

from typing import Optional

from app.core.config import Settings, get_settings
from app.llm.base import (
    BaseLLMProvider,
    LLMConfigurationError,
    LLMParseError,
    LLMProviderError,
    LLMResponseError,
    LLMSchemaValidationError,
    LLMSemanticValidationError,
    LLMTimeoutError,
)
from app.llm.fallback import create_fallback_perception
from app.llm.fixture import FixtureLLMProvider
from app.llm.parser import parse_llm_perception
from app.llm.pipeline import PerceptionPipeline, PerceptionPipelineResult
from app.llm.prompts import (
    SYSTEM_PROMPT,
    build_chat_messages,
    build_retry_messages,
    build_ticket_user_prompt,
)
from app.llm.provider import LiveLLMProvider
from app.llm.validation import validate_llm_perception


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


def get_perception_pipeline(
    settings: Optional[Settings] = None,
    provider: Optional[BaseLLMProvider] = None,
) -> PerceptionPipeline:
    """Factory constructing the resilient perception pipeline wrapping a provider.

    Args:
        settings: Application settings. If None, resolves from get_settings().
        provider: Optional explicit provider instance.

    Returns:
        PerceptionPipeline with bounded retries and deterministic fallback.
    """
    if settings is None:
        settings = get_settings()

    if provider is None:
        provider = get_llm_provider(settings)

    return PerceptionPipeline(provider=provider, max_retries=settings.LLM_MAX_RETRIES)


__all__ = [
    "BaseLLMProvider",
    "FixtureLLMProvider",
    "LiveLLMProvider",
    "LLMConfigurationError",
    "LLMParseError",
    "LLMProviderError",
    "LLMResponseError",
    "LLMSchemaValidationError",
    "LLMSemanticValidationError",
    "LLMTimeoutError",
    "PerceptionPipeline",
    "PerceptionPipelineResult",
    "SYSTEM_PROMPT",
    "build_chat_messages",
    "build_retry_messages",
    "build_ticket_user_prompt",
    "create_fallback_perception",
    "get_llm_provider",
    "get_perception_pipeline",
    "parse_llm_perception",
    "validate_llm_perception",
]

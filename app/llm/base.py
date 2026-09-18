"""Base abstractions and typed exceptions for LLM perception providers."""

from abc import ABC, abstractmethod

from app.domain.models import Ticket
from app.schemas.llm import LLMPerceptionOutput


class LLMProviderError(Exception):
    """Base exception for LLM provider failures."""

    error_code: str = "LLM_PROVIDER_ERROR"

    def __init__(self, message: str, error_code: str | None = None):
        super().__init__(message)
        if error_code is not None:
            self.error_code = error_code


class LLMConfigurationError(LLMProviderError):
    """Raised when an LLM provider is misconfigured or lacks necessary credentials."""

    error_code: str = "CONFIGURATION_ERROR"


class LLMResponseError(LLMProviderError):
    """Raised when an upstream LLM provider returns an unrecoverable or invalid response."""

    error_code: str = "PROVIDER_ERROR"


class LLMParseError(LLMResponseError):
    """Raised when raw model output cannot be decoded as valid JSON."""

    error_code: str = "MALFORMED_JSON"


class LLMSchemaValidationError(LLMResponseError):
    """Raised when parsed JSON fails structural Pydantic validation."""

    error_code: str = "SCHEMA_VALIDATION_ERROR"


class LLMSemanticValidationError(LLMResponseError):
    """Raised when structured output fails semantic reasonability checks."""

    error_code: str = "SEMANTIC_VALIDATION_ERROR"


class LLMTimeoutError(LLMResponseError):
    """Raised when an upstream LLM provider call exceeds the configured timeout."""

    error_code: str = "TIMEOUT_ERROR"


class BaseLLMProvider(ABC):
    """Abstract interface for ticket perception and classification."""

    @abstractmethod
    async def generate_perception(self, ticket: Ticket) -> LLMPerceptionOutput:
        """Analyze an untrusted ticket and return structured perception data.

        Args:
            ticket: Domain ticket containing subject, body, and customer context.

        Returns:
            LLMPerceptionOutput with category, severity, intent, and draft reply.

        Raises:
            LLMProviderError: If the provider fails or encounters an unrecoverable issue.
        """

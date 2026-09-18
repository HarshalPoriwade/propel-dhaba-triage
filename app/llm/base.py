"""Base abstractions and exceptions for LLM perception providers."""

from abc import ABC, abstractmethod

from app.domain.models import Ticket
from app.schemas.llm import LLMPerceptionOutput


class LLMProviderError(Exception):
    """Base exception for LLM provider failures."""


class LLMConfigurationError(LLMProviderError):
    """Raised when an LLM provider is misconfigured or lacks necessary credentials."""


class LLMResponseError(LLMProviderError):
    """Raised when an upstream LLM provider returns an unrecoverable error."""


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

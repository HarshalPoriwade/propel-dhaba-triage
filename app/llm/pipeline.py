"""Resilient LLM perception pipeline managing bounded retries, validation, and safe degradation."""

from dataclasses import dataclass
from typing import Optional

from app.domain.models import Ticket
from app.llm.base import (
    BaseLLMProvider,
    LLMConfigurationError,
    LLMResponseError,
)
from app.llm.fallback import create_fallback_perception
from app.schemas.llm import LLMPerceptionOutput


@dataclass(frozen=True)
class PerceptionPipelineResult:
    """Outcome of running the perception pipeline on a ticket."""

    perception: LLMPerceptionOutput
    degraded: bool
    attempts: int
    error_code: Optional[str] = None


class PerceptionPipeline(BaseLLMProvider):
    """Pipeline governing parse, validation, bounded retry, and deterministic degradation."""

    def __init__(self, provider: BaseLLMProvider, max_retries: int = 1):
        """Initialize pipeline with target provider and bounded retry limit.

        Args:
            provider: Concrete BaseLLMProvider instance (Live or Fixture).
            max_retries: Maximum number of retry attempts for recoverable failures (default: 1).
        """
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self.provider = provider
        self.max_retries = max_retries

    async def execute(self, ticket: Ticket) -> PerceptionPipelineResult:
        """Execute perception generation with bounded retry and deterministic fallback.

        Args:
            ticket: Domain Ticket entity.

        Returns:
            PerceptionPipelineResult containing validated perception, degraded flag, and attempt count.
        """
        max_attempts = 1 + self.max_retries
        last_error_code: Optional[str] = None
        last_error_reason: str = "Invalid structured output"

        for attempt in range(1, max_attempts + 1):
            try:
                if attempt == 1:
                    perception = await self.provider.generate_perception(ticket)
                else:
                    if hasattr(self.provider, "generate_perception_retry"):
                        perception = await self.provider.generate_perception_retry(
                            ticket, error_reason=last_error_reason
                        )
                    else:
                        perception = await self.provider.generate_perception(ticket)

                return PerceptionPipelineResult(
                    perception=perception,
                    degraded=False,
                    attempts=attempt,
                    error_code=None,
                )

            except LLMConfigurationError:
                # Deterministic configuration failure (e.g. missing API key) is unrecoverable; do NOT loop
                return PerceptionPipelineResult(
                    perception=create_fallback_perception(ticket, reason="CONFIGURATION_ERROR"),
                    degraded=True,
                    attempts=attempt,
                    error_code="CONFIGURATION_ERROR",
                )

            except LLMResponseError as err:
                last_error_code = getattr(err, "error_code", "PROVIDER_ERROR")
                last_error_reason = last_error_code

                if attempt >= max_attempts:
                    # Retry budget exhausted; degrade to deterministic safe fallback
                    return PerceptionPipelineResult(
                        perception=create_fallback_perception(ticket, reason=last_error_code),
                        degraded=True,
                        attempts=attempt,
                        error_code=last_error_code,
                    )

        # Fallback safeguard in case loop exits unexpectedly
        return PerceptionPipelineResult(
            perception=create_fallback_perception(ticket, reason="EXHAUSTED"),
            degraded=True,
            attempts=max_attempts,
            error_code=last_error_code or "PROVIDER_ERROR",
        )

    async def generate_perception(self, ticket: Ticket) -> LLMPerceptionOutput:
        """Generate perception conforming to BaseLLMProvider interface."""
        result = await self.execute(ticket)
        return result.perception

"""Live LLM provider communicating with an external OpenAI-compatible chat API."""

from typing import Any, Dict, List, Optional
import httpx

from app.domain.models import Ticket
from app.llm.base import (
    BaseLLMProvider,
    LLMConfigurationError,
    LLMResponseError,
    LLMTimeoutError,
)
from app.llm.parser import parse_llm_perception
from app.llm.prompts import build_chat_messages, build_retry_messages
from app.schemas.llm import LLMPerceptionOutput


class LiveLLMProvider(BaseLLMProvider):
    """Concrete provider communicating with an external OpenAI-compatible chat API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        timeout_seconds: float = 2.5,
    ):
        """Initialize the live LLM provider.

        Args:
            api_key: Secret API key for authentication.
            model_name: Target model identifier.
            base_url: Custom API endpoint URL.
            timeout_seconds: Network request timeout in seconds.

        Raises:
            LLMConfigurationError: If api_key is missing or empty.
        """
        if not api_key or not api_key.strip():
            raise LLMConfigurationError("MODEL_API_KEY is required for LiveLLMProvider.")

        self.api_key = api_key.strip()
        self.model_name = model_name
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def call_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Execute chat completion HTTP request and return raw content string.

        Args:
            messages: List of OpenAI-compatible message dictionaries.

        Returns:
            Raw content string from model completion.

        Raises:
            LLMTimeoutError: If the request exceeds timeout_seconds.
            LLMResponseError: If upstream API returns an HTTP or network error.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.TimeoutException as err:
            raise LLMTimeoutError(
                f"Upstream provider timed out after {self.timeout_seconds}s"
            ) from err
        except httpx.HTTPStatusError as err:
            raise LLMResponseError(
                f"Upstream provider returned HTTP {err.response.status_code}"
            ) from err
        except httpx.RequestError as err:
            raise LLMResponseError(f"Network error calling upstream provider: {err}") from err
        except Exception as err:
            raise LLMResponseError(f"Unexpected provider error: {err}") from err

    async def generate_perception(self, ticket: Ticket) -> LLMPerceptionOutput:
        """Generate perception using initial chat messages.

        Args:
            ticket: Domain ticket containing customer input.

        Returns:
            Validated and semantically verified LLMPerceptionOutput.
        """
        messages = build_chat_messages(ticket)
        raw_content = await self.call_messages(messages)
        return parse_llm_perception(raw_content)

    async def generate_perception_retry(
        self, ticket: Ticket, error_reason: str
    ) -> LLMPerceptionOutput:
        """Generate perception using targeted retry messages reinforcing schema constraints.

        Args:
            ticket: Domain ticket containing customer input.
            error_reason: Brief description of previous attempt failure.

        Returns:
            Validated and semantically verified LLMPerceptionOutput.
        """
        messages = build_retry_messages(ticket, error_reason=error_reason)
        raw_content = await self.call_messages(messages)
        return parse_llm_perception(raw_content)

"""Live LLM provider communicating with an external model API."""

import json
from typing import Optional
import httpx

from app.domain.models import Ticket
from app.llm.base import BaseLLMProvider, LLMConfigurationError, LLMResponseError
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
            timeout_seconds: Network request timeout.

        Raises:
            LLMConfigurationError: If api_key is missing or empty.
        """
        if not api_key or not api_key.strip():
            raise LLMConfigurationError("MODEL_API_KEY is required for LiveLLMProvider.")

        self.api_key = api_key.strip()
        self.model_name = model_name
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def generate_perception(self, ticket: Ticket) -> LLMPerceptionOutput:
        """Execute chat completion against the external provider and return perception data.

        Args:
            ticket: Domain ticket containing customer input.

        Returns:
            Validated LLMPerceptionOutput.

        Raises:
            LLMResponseError: If the upstream call fails, times out, or returns invalid data.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Minimal prompt payload for this step; prompt optimization and retry belong to later steps
        prompt_content = (
            f"Analyze this customer ticket and output JSON with category, severity, "
            f"user_requested_refund (bool), claimed_issue, detected_language, "
            f"suggested_escalation (bool), confidence (0.0-1.0), and draft_reply.\n\n"
            f"Subject: {ticket.subject}\n"
            f"Body: {ticket.body}"
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a customer support ticket analyzer. Output valid JSON only.",
                },
                {"role": "user", "content": prompt_content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

            raw_content = data["choices"][0]["message"]["content"]
            parsed_json = json.loads(raw_content)
            return LLMPerceptionOutput.model_validate(parsed_json)
        except httpx.HTTPStatusError as err:
            raise LLMResponseError(
                f"Upstream provider returned HTTP {err.response.status_code}: {err.response.text}"
            ) from err
        except httpx.RequestError as err:
            raise LLMResponseError(f"Network error calling upstream provider: {err}") from err
        except Exception as err:
            raise LLMResponseError(f"Failed to parse provider response into LLMPerceptionOutput: {err}") from err

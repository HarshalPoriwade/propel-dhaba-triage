"""Deterministic fixture LLM provider running completely offline without API tokens."""

import json
from pathlib import Path
from typing import Dict, Optional, Union

from app.domain.enums import Category, Severity
from app.domain.models import Ticket
from app.llm.base import BaseLLMProvider, LLMResponseError
from app.schemas.llm import LLMPerceptionOutput


class FixtureLLMProvider(BaseLLMProvider):
    """Deterministic, offline LLM provider reading from curated perception fixtures."""

    def __init__(self, fixture_path: Optional[Union[str, Path]] = None):
        """Initialize provider by loading perception fixtures from disk.

        Args:
            fixture_path: Optional custom path to fixtures JSON file.
        """
        if fixture_path is None:
            fixture_path = Path(__file__).parents[2] / "fixtures" / "llm_responses.json"
        self.fixture_path = Path(fixture_path)
        self._fixtures: Dict[str, dict] = self._load_fixtures()

    def _load_fixtures(self) -> Dict[str, dict]:
        """Load and parse fixture JSON dictionary."""
        if not self.fixture_path.is_file():
            raise LLMResponseError(f"Fixture file not found at: {self.fixture_path}")
        try:
            with open(self.fixture_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as err:
            raise LLMResponseError(f"Failed to read fixture file: {err}") from err

    async def generate_perception(self, ticket: Ticket) -> LLMPerceptionOutput:
        """Return deterministic perception response for the given ticket.

        Args:
            ticket: Domain ticket containing customer input.

        Returns:
            Validated LLMPerceptionOutput.
        """
        fixture_data = self._fixtures.get(ticket.id)
        if fixture_data is not None:
            return LLMPerceptionOutput.model_validate(fixture_data)

        # Fallback for unknown/arbitrary test tickets not in the 12 primary fixtures
        return LLMPerceptionOutput(
            category=Category.GENERAL,
            severity=Severity.LOW,
            user_requested_refund=False,
            claimed_issue=f"General support query for ticket {ticket.id}",
            detected_language="en",
            suggested_escalation=False,
            confidence=0.85,
            draft_reply="Thank you for reaching out to Dhaba. We have received your inquiry and our support team will get back to you shortly.",
        )

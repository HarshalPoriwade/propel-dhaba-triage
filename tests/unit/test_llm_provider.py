"""Unit tests for the LLM provider abstraction, fixture mode, and provider factory."""

import json
from pathlib import Path
import pytest

from app.core.config import Settings
from app.domain.models import Ticket
from app.llm import (
    BaseLLMProvider,
    FixtureLLMProvider,
    LiveLLMProvider,
    LLMConfigurationError,
    get_llm_provider,
)
from app.schemas.llm import LLMPerceptionOutput


def load_all_dhaba_tickets():
    """Helper to load all 12 tickets from the official dhaba_tickets.json source."""
    tickets_path = Path(__file__).parents[2] / "dhaba_tickets.json"
    with open(tickets_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["tickets"]


@pytest.mark.asyncio
async def test_fixture_provider_deterministic_results():
    """Verify fixture provider returns identical, deterministic results across multiple calls."""
    provider = FixtureLLMProvider()
    ticket_dict = load_all_dhaba_tickets()[0]  # T-1001

    ticket = Ticket(
        id=ticket_dict["id"],
        received_at=ticket_dict["received_at"],
        subject=ticket_dict["subject"],
        body=ticket_dict["body"],
        purchases=(),
        app_opens_since_renewal=0,
    )

    res1 = await provider.generate_perception(ticket)
    res2 = await provider.generate_perception(ticket)

    assert isinstance(res1, LLMPerceptionOutput)
    assert res1.category == res2.category
    assert res1.severity == res2.severity
    assert res1.user_requested_refund == res2.user_requested_refund
    assert res1.claimed_issue == res2.claimed_issue
    assert res1.confidence == res2.confidence
    assert res1.draft_reply == res2.draft_reply


@pytest.mark.asyncio
async def test_all_12_tickets_have_fixture_coverage():
    """Verify that all 12 tickets in dhaba_tickets.json have valid fixture perception entries."""
    provider = FixtureLLMProvider()
    raw_tickets = load_all_dhaba_tickets()
    assert len(raw_tickets) == 12

    for raw in raw_tickets:
        ticket = Ticket(
            id=raw["id"],
            received_at=raw["received_at"],
            subject=raw["subject"],
            body=raw["body"],
            purchases=(),
            app_opens_since_renewal=raw["app_opens_since_renewal"],
        )
        perception = await provider.generate_perception(ticket)
        assert isinstance(perception, LLMPerceptionOutput)
        assert perception.confidence >= 0.0
        assert perception.confidence <= 1.0
        assert len(perception.draft_reply) > 0
        assert len(perception.claimed_issue) > 0


@pytest.mark.asyncio
async def test_fixture_provider_unknown_ticket_safe_fallback():
    """Verify that an arbitrary ticket not in the 12 fixtures falls back to safe default perception."""
    provider = FixtureLLMProvider()
    ticket = Ticket(
        id="T-CUSTOM-999",
        received_at="2026-09-10T12:00:00+05:30",
        subject="Custom subject",
        body="Custom message",
        purchases=(),
        app_opens_since_renewal=0,
    )
    perception = await provider.generate_perception(ticket)
    assert isinstance(perception, LLMPerceptionOutput)
    assert perception.category.value == "general"
    assert perception.confidence == 0.85


def test_provider_selection_chooses_fixture_by_default():
    """Verify that get_llm_provider defaults to FixtureLLMProvider when MODEL_MODE=fixture."""
    settings = Settings(MODEL_MODE="fixture", MODEL_API_KEY=None)
    provider = get_llm_provider(settings)
    assert isinstance(provider, BaseLLMProvider)
    assert isinstance(provider, FixtureLLMProvider)


def test_provider_selection_chooses_live_when_configured():
    """Verify that get_llm_provider instantiates LiveLLMProvider when MODEL_MODE=live and key provided."""
    settings = Settings(
        MODEL_MODE="live",
        MODEL_API_KEY="sk-test-mock-key-12345",
        MODEL_NAME="gpt-4o-mini",
        MODEL_TIMEOUT_SECONDS=3.0,
    )
    provider = get_llm_provider(settings)
    assert isinstance(provider, BaseLLMProvider)
    assert isinstance(provider, LiveLLMProvider)
    assert provider.model_name == "gpt-4o-mini"
    assert provider.timeout_seconds == 3.0


def test_live_provider_fails_safely_without_api_key():
    """Verify that LiveLLMProvider explicitly refuses to initialize without an API key."""
    with pytest.raises(LLMConfigurationError) as exc_info:
        LiveLLMProvider(api_key=None)
    assert "MODEL_API_KEY is required" in str(exc_info.value)

    with pytest.raises(LLMConfigurationError) as exc_info_empty:
        LiveLLMProvider(api_key="   ")
    assert "MODEL_API_KEY is required" in str(exc_info_empty.value)

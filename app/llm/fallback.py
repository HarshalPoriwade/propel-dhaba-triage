"""Deterministic safe fallback generator for degraded triage perception."""

from typing import Optional

from app.domain.enums import Category, Severity
from app.domain.models import Ticket
from app.schemas.llm import LLMPerceptionOutput


def create_fallback_perception(
    ticket: Ticket,
    reason: Optional[str] = None,
) -> LLMPerceptionOutput:
    """Construct a deterministic, conservative fallback perception when LLM generation fails.

    Key Invariants:
    - Never authorizes or calculates refunds (user_requested_refund=False, no financial fields).
    - Never claims refunds or cancellations were executed.
    - Sets confidence strictly to 0.0 to signal automated uncertainty.
    - Sets suggested_escalation=True so downstream support tools route to a human.
    - Emits a polite, non-committal holding reply in English.
    - Never exposes internal stack traces or technical errors to the customer.

    Args:
        ticket: Domain Ticket entity.
        reason: Optional internal reason string for telemetry/logging (not exposed in reply).

    Returns:
        Deterministic LLMPerceptionOutput instance.
    """
    return LLMPerceptionOutput(
        category=Category.GENERAL,
        severity=Severity.MEDIUM,
        user_requested_refund=False,
        claimed_issue="Automated triage degraded: ticket flagged for manual human review",
        detected_language="en",
        suggested_escalation=True,
        confidence=0.0,
        draft_reply=(
            "Thank you for contacting Dhaba support. We have received your message "
            "and our customer care team is reviewing it. We will get back to you shortly."
        ),
    )

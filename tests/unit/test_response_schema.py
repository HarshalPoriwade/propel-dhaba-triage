"""Unit tests for the final HTTP response schema contracts."""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from app.domain.enums import Category, Severity
from app.domain.models import RefundDecision, TriageResult
from app.schemas.response import RefundResponse, TicketTriageResponse


def test_valid_response_schema_construction():
    """Verify clean construction and serialization of TicketTriageResponse."""
    now = datetime.now(timezone.utc)
    response = TicketTriageResponse(
        ticket_id="T-1001",
        category=Category.BILLING,
        severity=Severity.MEDIUM,
        refund=RefundResponse(
            should_refund=True,
            amount_inr=249,
            reason="First renewal charged within 24h with zero app opens.",
        ),
        reply_draft="We have approved your refund of ₹249.",
        needs_human=False,
        confidence=0.95,
        degraded=False,
        triaged_at=now,
    )
    assert response.ticket_id == "T-1001"
    assert response.refund.should_refund is True
    assert response.refund.amount_inr == 249

    # Verify JSON serialization succeeds
    json_str = response.model_dump_json()
    assert "T-1001" in json_str
    assert "249" in json_str


def test_response_from_domain_mapping():
    """Verify TicketTriageResponse.from_domain correctly maps domain models."""
    now = datetime.now(timezone.utc)
    domain_result = TriageResult(
        ticket_id="T-1006",
        category=Category.BILLING,
        severity=Severity.HIGH,
        refund=RefundDecision(
            should_refund=True,
            amount_inr=1499,
            reason="Annual renewal within 48h with zero opens",
            eligible_purchase_ids=("pay_F2",),
        ),
        reply_draft="We have approved your annual plan refund.",
        needs_human=False,
        confidence=0.98,
        degraded=False,
        triaged_at=now,
    )

    api_response = TicketTriageResponse.from_domain(domain_result)
    assert api_response.ticket_id == "T-1006"
    assert api_response.category == Category.BILLING
    assert api_response.refund.amount_inr == 1499
    assert api_response.refund.should_refund is True
    assert api_response.confidence == 0.98


def test_response_rejects_out_of_bounds_confidence():
    """Verify confidence must strictly be between 0.0 and 1.0."""
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        TicketTriageResponse(
            ticket_id="T-1001",
            category=Category.BILLING,
            severity=Severity.MEDIUM,
            refund=RefundResponse(should_refund=False, amount_inr=0, reason="None"),
            reply_draft="Reply",
            needs_human=False,
            confidence=1.5,  # Invalid: > 1.0
            degraded=False,
            triaged_at=now,
        )

    with pytest.raises(ValidationError):
        TicketTriageResponse(
            ticket_id="T-1001",
            category=Category.BILLING,
            severity=Severity.MEDIUM,
            refund=RefundResponse(should_refund=False, amount_inr=0, reason="None"),
            reply_draft="Reply",
            needs_human=False,
            confidence=-0.1,  # Invalid: < 0.0
            degraded=False,
            triaged_at=now,
        )


def test_response_rejects_negative_refund_amount():
    """Verify refund amount in response must be non-negative."""
    with pytest.raises(ValidationError):
        RefundResponse(
            should_refund=False,
            amount_inr=-249,
            reason="Invalid negative amount",
        )


def test_response_rejects_extra_fields():
    """Verify extra unexpected fields are rejected on response schema."""
    now = datetime.now(timezone.utc)
    raw = {
        "ticket_id": "T-1001",
        "category": "billing",
        "severity": "medium",
        "refund": {
            "should_refund": False,
            "amount_inr": 0,
            "reason": "None",
        },
        "reply_draft": "Reply",
        "needs_human": False,
        "confidence": 0.9,
        "degraded": False,
        "triaged_at": now.isoformat(),
        "unauthorized_field": "fake",
    }
    with pytest.raises(ValidationError) as exc_info:
        TicketTriageResponse.model_validate(raw)
    assert "extra_forbidden" in str(exc_info.value)

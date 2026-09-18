"""Unit tests for domain models and enumerations."""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from app.domain.enums import Category, PurchaseStatus, PurchaseType, Severity
from app.domain.models import Purchase, RefundDecision, Ticket, TriageResult


def test_enums_string_values():
    """Verify enums are properly defined and string-compatible."""
    assert Category.BILLING == "billing"
    assert Category.CANCELLATION == "cancellation"
    assert Category.TECHNICAL == "technical"
    assert Category.ACCOUNT == "account"
    assert Category.FEATURE_REQUEST == "feature_request"
    assert Category.COMPLAINT == "complaint"
    assert Category.GENERAL == "general"

    assert Severity.LOW == "low"
    assert Severity.MEDIUM == "medium"
    assert Severity.HIGH == "high"
    assert Severity.CRITICAL == "critical"

    assert PurchaseType.TRIAL == "trial"
    assert PurchaseType.RENEWAL == "renewal"

    assert PurchaseStatus.INITIATED == "initiated"
    assert PurchaseStatus.FAILED == "failed"
    assert PurchaseStatus.SUCCESSFUL == "successful"


def test_purchase_model_immutability():
    """Verify Purchase model is frozen and immutable."""
    now = datetime.now(timezone.utc)
    purchase = Purchase(
        id="pay_01",
        type=PurchaseType.TRIAL,
        amount_inr=1,
        status=PurchaseStatus.SUCCESSFUL,
        at=now,
    )
    assert purchase.amount_inr == 1

    with pytest.raises(ValidationError):
        # Attempting to mutate a frozen model must raise ValidationError
        purchase.amount_inr = 249  # type: ignore


def test_purchase_monetary_exactness():
    """Verify purchase requires non-negative integer amount and rejects negative values."""
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        Purchase(
            id="pay_01",
            type=PurchaseType.RENEWAL,
            amount_inr=-50,
            status=PurchaseStatus.SUCCESSFUL,
            at=now,
        )


def test_ticket_model_immutability():
    """Verify Ticket model is frozen and immutable."""
    now = datetime.now(timezone.utc)
    purchase = Purchase(
        id="pay_A1",
        type=PurchaseType.TRIAL,
        amount_inr=1,
        status=PurchaseStatus.SUCCESSFUL,
        at=now,
    )
    ticket = Ticket(
        id="T-1001",
        received_at=now,
        subject="Issue",
        body="Message body",
        purchases=(purchase,),
        app_opens_since_renewal=0,
    )
    assert ticket.id == "T-1001"
    assert len(ticket.purchases) == 1

    with pytest.raises(ValidationError):
        ticket.id = "T-9999"  # type: ignore


def test_refund_decision_model():
    """Verify RefundDecision properties and immutability."""
    decision = RefundDecision(
        should_refund=True,
        amount_inr=249,
        reason="First renewal within 24h with zero opens",
        eligible_purchase_ids=("pay_A2",),
    )
    assert decision.should_refund is True
    assert decision.amount_inr == 249
    assert decision.eligible_purchase_ids == ("pay_A2",)

    with pytest.raises(ValidationError):
        decision.should_refund = False  # type: ignore


def test_triage_result_model():
    """Verify TriageResult aggregates domain entities cleanly."""
    now = datetime.now(timezone.utc)
    decision = RefundDecision(
        should_refund=False,
        amount_inr=0,
        reason="App crash bug report; no refund requested",
    )
    result = TriageResult(
        ticket_id="T-1002",
        category=Category.TECHNICAL,
        severity=Severity.HIGH,
        refund=decision,
        reply_draft="We apologize for the crash...",
        needs_human=True,
        confidence=0.92,
        degraded=False,
        triaged_at=now,
    )
    assert result.ticket_id == "T-1002"
    assert result.category == Category.TECHNICAL
    assert result.severity == Severity.HIGH
    assert result.refund.amount_inr == 0
    assert result.needs_human is True

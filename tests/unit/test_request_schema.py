"""Unit tests for the HTTP request schema using real fixture data."""

import json
from datetime import datetime, timezone
from pathlib import Path
import pytest
from pydantic import ValidationError

from app.domain.enums import PurchaseStatus, PurchaseType
from app.domain.models import Ticket
from app.schemas.request import PurchaseInput, TicketTriageRequest


def load_all_dhaba_tickets():
    """Helper to load all 12 tickets from the official dhaba_tickets.json source."""
    tickets_path = Path(__file__).parents[2] / "dhaba_tickets.json"
    with open(tickets_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["tickets"]


def test_all_12_dhaba_tickets_validate_successfully():
    """Verify that all 12 real tickets from dhaba_tickets.json validate without error."""
    tickets = load_all_dhaba_tickets()
    assert len(tickets) == 12

    for raw_ticket in tickets:
        request = TicketTriageRequest.model_validate(raw_ticket)
        assert request.id == raw_ticket["id"]
        assert len(request.purchases) == len(raw_ticket["purchases"])
        assert request.app_opens_since_renewal == raw_ticket["app_opens_since_renewal"]

        # Verify domain conversion works seamlessly
        domain_ticket = request.to_domain()
        assert isinstance(domain_ticket, Ticket)
        assert domain_ticket.id == raw_ticket["id"]
        assert len(domain_ticket.purchases) == len(raw_ticket["purchases"])


def test_request_missing_required_field():
    """Verify that omitting required fields raises ValidationError."""
    raw = {
        "id": "T-1001",
        "received_at": "2026-09-02T09:14:00+05:30",
        # missing subject
        "body": "refund it",
        "purchases": [],
        "app_opens_since_renewal": 0,
    }
    with pytest.raises(ValidationError) as exc_info:
        TicketTriageRequest.model_validate(raw)
    assert "subject" in str(exc_info.value)


def test_request_rejects_floating_point_monetary_amount():
    """Verify that strict integer validation prevents float amounts like 249.50."""
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        PurchaseInput(
            id="pay_01",
            type=PurchaseType.RENEWAL,
            amount_inr=249.50,  # type: ignore - float strictly disallowed
            status=PurchaseStatus.SUCCESSFUL,
            at=now,
        )


def test_request_rejects_negative_monetary_amount():
    """Verify negative transaction amounts are rejected."""
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        PurchaseInput(
            id="pay_01",
            type=PurchaseType.RENEWAL,
            amount_inr=-100,
            status=PurchaseStatus.SUCCESSFUL,
            at=now,
        )


def test_request_rejects_negative_app_opens():
    """Verify negative app opens count is rejected."""
    raw = {
        "id": "T-1001",
        "received_at": "2026-09-02T09:14:00+05:30",
        "subject": "subject",
        "body": "body",
        "purchases": [],
        "app_opens_since_renewal": -1,
    }
    with pytest.raises(ValidationError) as exc_info:
        TicketTriageRequest.model_validate(raw)
    assert "app_opens_since_renewal" in str(exc_info.value)


def test_request_rejects_invalid_purchase_status_enum():
    """Verify unsupported purchase statuses (e.g. 'pending') fail validation."""
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError) as exc_info:
        PurchaseInput(
            id="pay_01",
            type=PurchaseType.TRIAL,
            amount_inr=1,
            status="pending",  # type: ignore
            at=now,
        )
    assert "status" in str(exc_info.value)


def test_request_rejects_excessively_large_body():
    """Verify DOS prevention: bodies exceeding 10,000 characters fail validation."""
    raw = {
        "id": "T-1001",
        "received_at": "2026-09-02T09:14:00+05:30",
        "subject": "subject",
        "body": "A" * 10001,
        "purchases": [],
        "app_opens_since_renewal": 0,
    }
    with pytest.raises(ValidationError) as exc_info:
        TicketTriageRequest.model_validate(raw)
    assert "body" in str(exc_info.value)


def test_request_rejects_unexpected_extra_fields():
    """Verify extra fields are strictly forbidden."""
    raw = {
        "id": "T-1001",
        "received_at": "2026-09-02T09:14:00+05:30",
        "subject": "subject",
        "body": "body",
        "purchases": [],
        "app_opens_since_renewal": 0,
        "unauthorized_field": "injected_data",
    }
    with pytest.raises(ValidationError) as exc_info:
        TicketTriageRequest.model_validate(raw)
    assert "extra_forbidden" in str(exc_info.value)


# =========================================================================
# Step 10 Tests: Comprehensive Request Validation Matrix
# =========================================================================


def test_request_rejects_missing_and_empty_ticket_id():
    """Verify missing or empty ticket ID is rejected."""
    # Missing id
    with pytest.raises(ValidationError, match="Field required"):
        TicketTriageRequest.model_validate({
            "received_at": "2026-09-02T09:14:00+05:30",
            "subject": "Sub",
            "body": "Body",
            "purchases": [],
            "app_opens_since_renewal": 0,
        })

    # Empty id
    with pytest.raises(ValidationError, match="String should have at least 1 character"):
        TicketTriageRequest.model_validate({
            "id": "",
            "received_at": "2026-09-02T09:14:00+05:30",
            "subject": "Sub",
            "body": "Body",
            "purchases": [],
            "app_opens_since_renewal": 0,
        })


def test_request_rejects_invalid_body_type():
    """Verify non-string body types fail validation."""
    with pytest.raises(ValidationError):
        TicketTriageRequest.model_validate({
            "id": "T-1001",
            "received_at": "2026-09-02T09:14:00+05:30",
            "subject": "Sub",
            "body": 12345,  # int instead of str
            "purchases": [],
            "app_opens_since_renewal": 0,
        })


def test_request_rejects_invalid_purchases_structure():
    """Verify non-list purchases structure fails validation."""
    with pytest.raises(ValidationError):
        TicketTriageRequest.model_validate({
            "id": "T-1001",
            "received_at": "2026-09-02T09:14:00+05:30",
            "subject": "Sub",
            "body": "Body",
            "purchases": "not-a-list",
            "app_opens_since_renewal": 0,
        })


def test_request_rejects_invalid_purchase_type_enum():
    """Verify unsupported purchase type string fails enum validation."""
    with pytest.raises(ValidationError, match="Input should be 'trial' or 'renewal'"):
        PurchaseInput(
            id="pay_01",
            type="annual_subscription",  # type: ignore
            amount_inr=1499,
            status=PurchaseStatus.SUCCESSFUL,
            at=datetime.now(timezone.utc),
        )


def test_request_rejects_invalid_datetime_format():
    """Verify malformed datetime string fails validation."""
    with pytest.raises(ValidationError):
        TicketTriageRequest.model_validate({
            "id": "T-1001",
            "received_at": "not-a-valid-iso-date",
            "subject": "Sub",
            "body": "Body",
            "purchases": [],
            "app_opens_since_renewal": 0,
        })


def test_request_rejects_excessively_long_subject():
    """Verify subject exceeding 500 characters fails validation."""
    with pytest.raises(ValidationError, match="String should have at most 500 characters"):
        TicketTriageRequest.model_validate({
            "id": "T-1001",
            "received_at": "2026-09-02T09:14:00+05:30",
            "subject": "S" * 501,
            "body": "Body",
            "purchases": [],
            "app_opens_since_renewal": 0,
        })


def test_request_rejects_excessive_purchases_count():
    """Verify purchases list exceeding 100 items fails validation."""
    now = datetime.now(timezone.utc)
    purchases = [
        {
            "id": f"pay_{i}",
            "type": "renewal",
            "amount_inr": 249,
            "status": "successful",
            "at": now.isoformat(),
        }
        for i in range(101)
    ]
    with pytest.raises(ValidationError, match="List should have at most 100 items"):
        TicketTriageRequest.model_validate({
            "id": "T-1001",
            "received_at": now.isoformat(),
            "subject": "Sub",
            "body": "Body",
            "purchases": purchases,
            "app_opens_since_renewal": 0,
        })


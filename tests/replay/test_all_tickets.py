"""End-to-end replay test suite executing all 12 real tickets from dhaba_tickets.json.

Uses deterministic fixture mode without requiring external network or paid API keys.
Produces tests/replay/results.json artifact for subsequent reporting and README documentation.
"""

import json
import os
from pathlib import Path
import tempfile
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.domain.enums import Category, Severity
from app.main import create_app
from app.schemas.response import TicketTriageResponse


@pytest.fixture
def replay_client():
    """Create TestClient with an isolated database for 12-ticket replay."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    settings = Settings(
        APP_NAME="Dhaba Replay Runner",
        APP_ENV="testing",
        MODEL_MODE="fixture",
        DATABASE_PATH=db_path,
    )
    application = create_app(settings)
    with TestClient(application) as client:
        yield client

    if os.path.exists(db_path):
        os.remove(db_path)


def load_all_dhaba_tickets():
    """Load the authoritative 12 tickets from dhaba_tickets.json."""
    tickets_path = Path(__file__).parents[2] / "dhaba_tickets.json"
    with open(tickets_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["tickets"]


def test_replay_all_12_tickets_end_to_end(replay_client):
    """Execute all 12 real tickets through the full application path and generate results.json artifact."""
    tickets = load_all_dhaba_tickets()
    assert len(tickets) == 12

    replay_records = []

    for ticket_data in tickets:
        ticket_id = ticket_data["id"]

        # Post ticket payload to POST /triage
        response = replay_client.post("/triage", json=ticket_data)
        assert response.status_code == 200, f"Ticket {ticket_id} failed with status {response.status_code}"

        result = response.json()
        validated = TicketTriageResponse.model_validate(result)

        # Ticket-specific invariant assertions
        assert validated.ticket_id == ticket_id
        assert isinstance(validated.category, Category)
        assert isinstance(validated.severity, Severity)
        assert validated.refund.amount_inr >= 0
        assert len(validated.reply_draft) > 0

        # Record concise, sanitized replay summary (no raw ticket bodies or personal data)
        replay_records.append({
            "ticket_id": validated.ticket_id,
            "category": validated.category.value,
            "severity": validated.severity.value,
            "refund_authorized": validated.refund.should_refund,
            "refund_amount_inr": validated.refund.amount_inr,
            "refund_reason": validated.refund.reason,
            "needs_human": validated.needs_human,
            "confidence": validated.confidence,
            "degraded": validated.degraded,
            "reply_preview": validated.reply_draft[:120] + ("..." if len(validated.reply_draft) > 120 else ""),
        })

    # Assert specific archetype classifications and safety determinations
    results_by_id = {r["ticket_id"]: r for r in replay_records}

    # T-1001: Renewal complaint -> billing, medium, refund held under undefined baseline
    assert results_by_id["T-1001"]["category"] == "billing"
    assert results_by_id["T-1001"]["severity"] == "medium"
    assert results_by_id["T-1001"]["refund_authorized"] is False
    assert "POLICY_UNDEFINED_MANUAL_REVIEW" in results_by_id["T-1001"]["refund_reason"]
    assert results_by_id["T-1001"]["needs_human"] is True

    # T-1002: Crash -> technical, high, needs_human=True
    assert results_by_id["T-1002"]["category"] == "technical"
    assert results_by_id["T-1002"]["severity"] == "high"
    assert results_by_id["T-1002"]["needs_human"] is True

    # T-1003: Prompt injection VIP -> account, refund held, zero rupees authorized
    assert results_by_id["T-1003"]["category"] == "account"
    assert results_by_id["T-1003"]["refund_authorized"] is False
    assert results_by_id["T-1003"]["refund_amount_inr"] == 0

    # T-1004: Trial cancel -> cancellation, low, no refund
    assert results_by_id["T-1004"]["category"] == "cancellation"
    assert results_by_id["T-1004"]["severity"] == "low"
    assert results_by_id["T-1004"]["refund_authorized"] is False

    # T-1005: Initiated payment -> billing, high severity, no refund requested in body (activation requested), needs_human=True
    assert results_by_id["T-1005"]["category"] == "billing"
    assert results_by_id["T-1005"]["severity"] == "high"
    assert results_by_id["T-1005"]["refund_authorized"] is False
    assert results_by_id["T-1005"]["refund_amount_inr"] == 0
    assert "NO_REFUND_REQUEST" in results_by_id["T-1005"]["refund_reason"]
    assert results_by_id["T-1005"]["needs_human"] is True

    # T-1006: Annual renewal -> billing, high, refund held under undefined baseline
    assert results_by_id["T-1006"]["category"] == "billing"
    assert results_by_id["T-1006"]["severity"] == "high"
    assert results_by_id["T-1006"]["refund_authorized"] is False
    assert results_by_id["T-1006"]["needs_human"] is True

    # T-1007: Feature request -> feature_request, low, needs_human=False
    assert results_by_id["T-1007"]["category"] == "feature_request"
    assert results_by_id["T-1007"]["severity"] == "low"
    assert results_by_id["T-1007"]["needs_human"] is False

    # T-1008: Fraud & cyber cell -> complaint, critical, HIGH_RISK_MANUAL_REVIEW
    assert results_by_id["T-1008"]["category"] == "complaint"
    assert results_by_id["T-1008"]["severity"] == "critical"
    assert results_by_id["T-1008"]["refund_authorized"] is False
    assert "HIGH_RISK_MANUAL_REVIEW" in results_by_id["T-1008"]["refund_reason"]
    assert results_by_id["T-1008"]["needs_human"] is True

    # T-1009: Double charge claim (1 failed, 1 successful) -> CONFLICTING_PAYMENT_EVIDENCE
    assert results_by_id["T-1009"]["category"] == "billing"
    assert results_by_id["T-1009"]["refund_authorized"] is False
    assert "CONFLICTING_PAYMENT_EVIDENCE" in results_by_id["T-1009"]["refund_reason"]
    assert results_by_id["T-1009"]["needs_human"] is True

    # T-1010: GST invoice -> billing, low, needs_human=False
    assert results_by_id["T-1010"]["category"] == "billing"
    assert results_by_id["T-1010"]["severity"] == "low"
    assert results_by_id["T-1010"]["needs_human"] is False

    # T-1011: Pre-sales prompt injection -> general, low, no system prompt leakage
    assert results_by_id["T-1011"]["category"] == "general"
    assert results_by_id["T-1011"]["severity"] == "low"
    assert "system prompt" not in results_by_id["T-1011"]["reply_preview"].lower()
    assert results_by_id["T-1011"]["needs_human"] is False

    # T-1012: 1-star review threat -> complaint, high, refund held under review
    assert results_by_id["T-1012"]["category"] == "complaint"
    assert results_by_id["T-1012"]["severity"] == "high"
    assert results_by_id["T-1012"]["refund_authorized"] is False
    assert results_by_id["T-1012"]["needs_human"] is True

    # Save output artifact to tests/replay/results.json
    output_dir = Path(__file__).parent
    output_path = output_dir / "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_tickets": len(replay_records),
                "mode": "fixture",
                "results": replay_records,
            },
            f,
            indent=2,
        )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_fixture_provider_unknown_ticket_returns_safe_neutral_response(replay_client):
    """Verify that an unknown ticket ID in fixture mode safely returns intended neutral behavior."""
    unknown_ticket = {
        "id": "T-UNKNOWN-999",
        "received_at": "2026-09-02T10:00:00+05:30",
        "subject": "Unknown ticket",
        "body": "This ticket has no fixture response",
        "purchases": [],
        "app_opens_since_renewal": 0,
    }

    response = replay_client.post("/triage", json=unknown_ticket)
    assert response.status_code == 200

    data = response.json()
    validated = TicketTriageResponse.model_validate(data)

    assert validated.ticket_id == "T-UNKNOWN-999"
    assert validated.category == Category.GENERAL
    assert validated.severity == Severity.LOW
    assert validated.refund.should_refund is False
    assert validated.confidence == 0.85

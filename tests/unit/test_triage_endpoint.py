"""Lightweight verification tests for POST /triage endpoint and orchestration."""

import os
import tempfile
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.domain.enums import Category, Severity
from app.main import create_app
from app.repositories.triage import ProcessingStatus, TriageRepository
from app.schemas.response import TicketTriageResponse


@pytest.fixture
def temp_db():
    """Provide an isolated temporary SQLite database path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def client(temp_db):
    """FastAPI TestClient configured with isolated SQLite DB and fixture LLM mode."""
    test_settings = Settings(
        APP_NAME="Dhaba Triage Test",
        APP_ENV="testing",
        MODEL_MODE="fixture",
        DATABASE_PATH=temp_db,
    )
    application = create_app(test_settings)
    with TestClient(application) as test_client:
        yield test_client


class TestTriageEndpoint:
    def test_health_check_endpoint(self, client):
        """Verify GET /health returns 200 OK and application metadata."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["app"] == "Dhaba Triage Test"

    def test_post_triage_success(self, client):
        """Verify POST /triage accepts valid ticket and returns complete triage response."""
        payload = {
            "id": "T-1001",
            "received_at": "2026-09-02T09:14:00+05:30",
            "subject": "charged 249 without telling me",
            "body": "i only paid 1 rupee to try the app. today 249 is gone from my account. i did not agree to this. refund it.",
            "purchases": [
                {
                    "id": "pay_A1",
                    "type": "trial",
                    "amount_inr": 1,
                    "status": "successful",
                    "at": "2026-08-31T20:02:00+05:30",
                },
                {
                    "id": "pay_A2",
                    "type": "renewal",
                    "amount_inr": 249,
                    "status": "successful",
                    "at": "2026-09-01T20:04:00+05:30",
                },
            ],
            "app_opens_since_renewal": 0,
        }

        response = client.post("/triage", json=payload)
        assert response.status_code == 200

        data = response.json()
        validated = TicketTriageResponse.model_validate(data)

        assert validated.ticket_id == "T-1001"
        assert validated.category == Category.BILLING
        assert validated.severity == Severity.MEDIUM
        assert validated.refund.should_refund is False  # Baseline: Dhaba policy not defined
        assert validated.refund.amount_inr == 0
        assert "POLICY_UNDEFINED_MANUAL_REVIEW" in validated.refund.reason
        assert validated.needs_human is True
        assert validated.confidence == 0.95
        assert validated.degraded is False
        assert len(validated.reply_draft) > 0

    def test_post_triage_idempotency_returns_stored_result(self, client, temp_db):
        """Submitting the same ticket ID twice returns stored result without re-processing."""
        payload = {
            "id": "T-1002",
            "received_at": "2026-09-02T11:40:00+05:30",
            "subject": "app crashes on opening",
            "body": "since yesterday update the app closes immediately when i open it.",
            "purchases": [],
            "app_opens_since_renewal": 41,
        }

        # First request
        resp1 = client.post("/triage", json=payload)
        assert resp1.status_code == 200
        data1 = resp1.json()

        # Second request with same ticket ID
        resp2 = client.post("/triage", json=payload)
        assert resp2.status_code == 200
        data2 = resp2.json()

        # Identical responses returned
        assert data1 == data2

        # Verify only 1 record exists in repository
        repo = TriageRepository(db_path=temp_db)
        assert repo.count_records() == 1

        record = repo.get_record("T-1002")
        assert record is not None
        assert record.status == ProcessingStatus.COMPLETED

    def test_post_triage_handles_in_progress_conflict(self, client, temp_db):
        """If a ticket is currently in_progress, endpoint returns 409 Conflict."""
        repo = TriageRepository(db_path=temp_db)
        repo.claim_ticket("T-1004")  # Claimed but not completed

        payload = {
            "id": "T-1004",
            "received_at": "2026-09-03T18:05:00+05:30",
            "subject": "cancel",
            "body": "cancel",
            "purchases": [],
            "app_opens_since_renewal": 1,
        }

        response = client.post("/triage", json=payload)
        assert response.status_code == 409
        assert "currently being triaged" in response.json()["detail"]

    def test_post_triage_invalid_schema_returns_422(self, client):
        """Malformed payloads fail Pydantic validation and return 422 Unprocessable Entity."""
        # Missing required fields
        response = client.post("/triage", json={"id": "T-INVALID"})
        assert response.status_code == 422

    def test_post_triage_releases_claim_on_unexpected_failure(self, client, temp_db, monkeypatch):
        """If pipeline execution raises an unexpected error, the in_progress claim is released."""
        from app.services.triage import TriageService

        async def failing_pipeline(*args, **kwargs):
            raise RuntimeError("Simulated catastrophic crash")

        # Mock the pipeline execute inside TriageService
        monkeypatch.setattr(
            "app.llm.pipeline.PerceptionPipeline.execute", failing_pipeline
        )

        payload = {
            "id": "T-CRASH",
            "received_at": "2026-09-02T09:14:00+05:30",
            "subject": "Crash test",
            "body": "This ticket will cause a crash",
            "purchases": [],
            "app_opens_since_renewal": 0,
        }

        response = client.post("/triage", json=payload)
        assert response.status_code == 500
        assert "internal error occurred" in response.json()["detail"]

        # Verify in_progress claim was released from repository
        repo = TriageRepository(db_path=temp_db)
        record = repo.get_record("T-CRASH")
        assert record is None  # Released!

    def test_refund_boundary_independent_of_llm(self, client):
        """T-1008 fraud case routes to human review regardless of LLM confidence or category."""
        payload = {
            "id": "T-1008",
            "received_at": "2026-09-05T16:47:00+05:30",
            "subject": "fraud",
            "body": "my father is 71 and does not know how to use apps. he says he never installed this. there are three charges on his card. we are reporting to cyber cell if not resolved today.",
            "purchases": [
                {"id": "pay_H1", "type": "trial", "amount_inr": 1, "status": "successful", "at": "2026-07-09T21:12:00+05:30"},
                {"id": "pay_H2", "type": "renewal", "amount_inr": 249, "status": "successful", "at": "2026-07-10T21:14:00+05:30"},
                {"id": "pay_H3", "type": "renewal", "amount_inr": 249, "status": "successful", "at": "2026-08-10T21:13:00+05:30"},
            ],
            "app_opens_since_renewal": 0,
        }

        response = client.post("/triage", json=payload)
        assert response.status_code == 200
        data = response.json()

        # Deterministic policy strictly governs refund
        assert data["refund"]["should_refund"] is False
        assert data["refund"]["amount_inr"] == 0
        assert "HIGH_RISK_MANUAL_REVIEW" in data["refund"]["reason"]
        assert data["needs_human"] is True


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

    def test_post_triage_retry_succeeds_after_released_claim(self, client, temp_db, monkeypatch):
        """Verify that after a failure releases the claim, a subsequent attempt successfully completes."""
        payload = {
            "id": "T-RETRY-SUCCESS",
            "received_at": "2026-09-02T09:14:00+05:30",
            "subject": "Intermittent failure",
            "body": "First attempt fails, second succeeds",
            "purchases": [],
            "app_opens_since_renewal": 0,
        }

        # Attempt 1: Simulate crash
        def failing_exec(*args, **kwargs):
            raise RuntimeError("Temporary glitch")

        monkeypatch.setattr("app.llm.pipeline.PerceptionPipeline.execute", failing_exec)
        resp1 = client.post("/triage", json=payload)
        assert resp1.status_code == 500

        # Undo mock so Attempt 2 runs real pipeline in fixture mode
        monkeypatch.undo()

        resp2 = client.post("/triage", json=payload)
        assert resp2.status_code == 200
        assert resp2.json()["ticket_id"] == "T-RETRY-SUCCESS"

        repo = TriageRepository(db_path=temp_db)
        record = repo.get_record("T-RETRY-SUCCESS")
        assert record is not None
        assert record.status == ProcessingStatus.COMPLETED

    def test_release_claim_never_deletes_completed_record(self, temp_db):
        """Verify release_claim only targets in_progress and never deletes completed rows."""
        repo = TriageRepository(db_path=temp_db)
        repo.init_db()
        claim = repo.claim_ticket("T-SAFE")
        assert claim.status.value == "claimed"

        # Mark completed
        from datetime import datetime, timezone
        from app.domain.enums import Category, Severity
        from app.schemas.response import RefundResponse, TicketTriageResponse

        completed_resp = TicketTriageResponse(
            ticket_id="T-SAFE",
            category=Category.GENERAL,
            severity=Severity.LOW,
            refund=RefundResponse(should_refund=False, amount_inr=0, reason="None"),
            reply_draft="Hello",
            needs_human=False,
            confidence=0.9,
            degraded=False,
            triaged_at=datetime.now(timezone.utc),
        )
        repo.save_completed_result("T-SAFE", completed_resp)

        # Attempt to release claim on completed ticket
        repo.release_claim("T-SAFE")

        # Record must still exist in completed status!
        record = repo.get_record("T-SAFE")
        assert record is not None
        assert record.status == ProcessingStatus.COMPLETED

    def test_api_error_does_not_leak_internals(self, client, monkeypatch):
        """Verify HTTP 500 error body does not contain tracebacks, file paths, or SQL."""
        def catastrophic_error(*args, **kwargs):
            raise Exception("sqlite3.OperationalError: table triage_records locked at /var/db/dhaba.db")

        monkeypatch.setattr("app.llm.pipeline.PerceptionPipeline.execute", catastrophic_error)

        payload = {
            "id": "T-LEAK-TEST",
            "received_at": "2026-09-02T09:14:00+05:30",
            "subject": "Subject",
            "body": "Body",
            "purchases": [],
            "app_opens_since_renewal": 0,
        }
        response = client.post("/triage", json=payload)
        assert response.status_code == 500
        detail = response.json()["detail"]

        # Sensitive internals MUST NOT be leaked to the client
        assert "sqlite3" not in detail
        assert "OperationalError" not in detail
        assert "dhaba.db" not in detail
        assert "Traceback" not in detail
        assert detail == "An unexpected internal error occurred while processing the ticket."

    def test_post_triage_circuit_breaker_outage_degrades_and_persists(self, temp_db, monkeypatch):
        """Verify endpoint degrades safely during outage, trips circuit, fails fast, and persists."""
        from app.core.resilience import reset_shared_circuit_breaker, get_circuit_breaker
        from app.llm.base import LLMTimeoutError

        reset_shared_circuit_breaker()

        test_settings = Settings(
            APP_NAME="Dhaba Resilience Test",
            APP_ENV="testing",
            MODEL_MODE="fixture",
            DATABASE_PATH=temp_db,
            LLM_FAILURE_THRESHOLD=2,
            LLM_RECOVERY_SECONDS=30.0,
            LLM_MAX_RETRIES=0,
        )
        application = create_app(test_settings)
        test_client = TestClient(application)

        call_counter = [0]

        async def failing_generate(*args, **kwargs):
            call_counter[0] += 1
            raise LLMTimeoutError("Upstream provider timed out after 2.5s")

        monkeypatch.setattr(
            "app.llm.fixture.FixtureLLMProvider.generate_perception",
            failing_generate,
        )

        ticket_1 = {
            "id": "T-FAIL-1",
            "received_at": "2026-09-02T09:14:00+05:30",
            "subject": "Outage ticket 1",
            "body": "Body 1",
            "purchases": [],
            "app_opens_since_renewal": 0,
        }
        ticket_2 = {
            "id": "T-FAIL-2",
            "received_at": "2026-09-02T09:14:00+05:30",
            "subject": "Outage ticket 2",
            "body": "Body 2",
            "purchases": [],
            "app_opens_since_renewal": 0,
        }
        ticket_3 = {
            "id": "T-FAST-FAIL-3",
            "received_at": "2026-09-02T09:14:00+05:30",
            "subject": "Outage ticket 3",
            "body": "Body 3",
            "purchases": [],
            "app_opens_since_renewal": 0,
        }

        # Request 1: Provider fails -> returns degraded 200 OK
        r1 = test_client.post("/triage", json=ticket_1)
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["degraded"] is True
        assert d1["needs_human"] is True
        assert d1["confidence"] == 0.0
        assert d1["refund"]["should_refund"] is False
        assert d1["refund"]["amount_inr"] == 0
        assert "timed out" not in d1["reply_draft"]
        assert call_counter[0] == 1

        # Request 2: Provider fails again -> trips circuit to OPEN
        r2 = test_client.post("/triage", json=ticket_2)
        assert r2.status_code == 200
        assert r2.json()["degraded"] is True
        assert call_counter[0] == 2

        # Request 3: Circuit is OPEN -> fails fast WITHOUT calling provider!
        r3 = test_client.post("/triage", json=ticket_3)
        assert r3.status_code == 200
        d3 = r3.json()
        assert d3["degraded"] is True
        assert d3["needs_human"] is True
        assert d3["confidence"] == 0.0
        assert call_counter[0] == 2  # Provider call count DID NOT INCREASE

        # Verify degraded result is stored in SQLite repository
        repo = TriageRepository(db_path=temp_db)
        record = repo.get_record("T-FAST-FAIL-3")
        assert record is not None
        assert record.status == ProcessingStatus.COMPLETED

        # Verify idempotency on degraded record (duplicate request returns stored result)
        r3_dup = test_client.post("/triage", json=ticket_3)
        assert r3_dup.status_code == 200
        assert r3_dup.json() == d3
        assert call_counter[0] == 2

        reset_shared_circuit_breaker()

    def test_post_triage_degraded_never_authorizes_refund_even_with_financial_evidence(
        self, temp_db, monkeypatch
    ):
        """Verify degraded mode never authorizes funds even if ticket has verified purchases."""
        from app.core.resilience import reset_shared_circuit_breaker
        from app.llm.base import LLMResponseError

        reset_shared_circuit_breaker()

        test_settings = Settings(
            APP_NAME="Dhaba Resilience Test",
            APP_ENV="testing",
            MODEL_MODE="fixture",
            DATABASE_PATH=temp_db,
            LLM_FAILURE_THRESHOLD=1,
        )
        application = create_app(test_settings)
        test_client = TestClient(application)

        async def failing_generate(*args, **kwargs):
            raise LLMResponseError("HTTP 503 Provider Down")

        monkeypatch.setattr(
            "app.llm.fixture.FixtureLLMProvider.generate_perception",
            failing_generate,
        )

        ticket_payload = {
            "id": "T-FINANCIAL-OUTAGE",
            "received_at": "2026-09-02T09:14:00+05:30",
            "subject": "annual plan renewal refund please",
            "body": "charged 1499 for annual plan. please refund my money immediately.",
            "purchases": [
                {
                    "id": "pay_BIG",
                    "type": "renewal",
                    "amount_inr": 1499,
                    "status": "successful",
                    "at": "2026-09-02T09:00:00+05:30",
                }
            ],
            "app_opens_since_renewal": 0,
        }

        resp = test_client.post("/triage", json=ticket_payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["degraded"] is True
        assert data["needs_human"] is True
        # Must not authorize any refund during degraded processing
        assert data["refund"]["should_refund"] is False
        assert data["refund"]["amount_inr"] == 0
        # Must not claim refund execution in reply
        reply = data["reply_draft"].lower()
        assert "refund executed" not in reply
        assert "money returned" not in reply
        assert "reversed" not in reply

        reset_shared_circuit_breaker()




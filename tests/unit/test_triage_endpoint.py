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
        from app.core.resilience import reset_shared_circuit_breaker
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


class TestObservabilityAndLogging:
    """Step 12: Tests verifying safe structured logging, correlation IDs, and 3am diagnostics."""

    def test_request_id_generated_and_returned_in_header(self, client):
        """Verify that every response carries an X-Request-ID header generated if absent."""
        payload = {
            "id": "T-REQ-GEN",
            "received_at": "2026-09-02T09:14:00+05:30",
            "subject": "Subject",
            "body": "Body",
            "purchases": [],
            "app_opens_since_renewal": 0,
        }
        resp = client.post("/triage", json=payload)
        assert resp.status_code == 200
        req_id = resp.headers.get("X-Request-ID")
        assert req_id is not None
        assert req_id.startswith("req_")
        assert len(req_id) >= 10

    def test_inbound_safe_request_id_preserved_in_header(self, client):
        """Verify an inbound safe correlation ID is respected and echoed in X-Request-ID."""
        payload = {
            "id": "T-REQ-PRESERVE",
            "received_at": "2026-09-02T09:14:00+05:30",
            "subject": "Subject",
            "body": "Body",
            "purchases": [],
            "app_opens_since_renewal": 0,
        }
        custom_id = "trace-correlation-id-abc-123"
        resp = client.post("/triage", json=payload, headers={"X-Request-ID": custom_id})
        assert resp.status_code == 200
        assert resp.headers.get("X-Request-ID") == custom_id

    def test_duplicate_requests_receive_distinct_request_ids(self, client):
        """Verify duplicate requests for the same ticket receive distinct correlation IDs."""
        payload = {
            "id": "T-DUP-REQ",
            "received_at": "2026-09-02T09:14:00+05:30",
            "subject": "Subject",
            "body": "Body",
            "purchases": [],
            "app_opens_since_renewal": 0,
        }
        r1 = client.post("/triage", json=payload)
        r2 = client.post("/triage", json=payload)
        assert r1.status_code == 200
        assert r2.status_code == 200
        id1 = r1.headers.get("X-Request-ID")
        id2 = r2.headers.get("X-Request-ID")
        assert id1 is not None and id2 is not None
        assert id1 != id2

    def test_structured_logs_capture_lifecycle_and_operational_metadata(self, client, caplog):
        """Verify structured logs emit lifecycle events with operational fields without data leakage."""
        import logging
        caplog.set_level(logging.INFO)

        payload = {
            "id": "T-1001",
            "received_at": "2026-09-02T09:14:00+05:30",
            "subject": "charged 249 without telling me",
            "body": "i only paid 1 rupee to try the app. today 249 is gone. refund it.",
            "purchases": [
                {"id": "pay_A1", "type": "trial", "amount_inr": 1, "status": "successful", "at": "2026-09-01T09:00:00+05:30"},
                {"id": "pay_A2", "type": "renewal", "amount_inr": 249, "status": "successful", "at": "2026-09-02T09:00:00+05:30"},
            ],
            "app_opens_since_renewal": 0,
        }
        resp = client.post("/triage", json=payload)
        assert resp.status_code == 200
        req_id = resp.headers.get("X-Request-ID")

        events = [r.__dict__.get("event") for r in caplog.records if "event" in r.__dict__]
        assert "triage.request_received" in events
        assert "llm.request_started" in events
        assert "triage.completed" in events

        # Inspect completed record fields
        completed_record = next(r for r in caplog.records if r.__dict__.get("event") == "triage.completed")
        d = completed_record.__dict__
        assert d["ticket_id"] == "T-1001"
        assert d["category"] == "billing"
        assert d["severity"] == "medium"
        assert d["refund_authorized"] is False
        assert d["refund_amount_inr"] == 0
        assert d["needs_human"] is True
        assert d["confidence"] == 0.95
        assert d["degraded"] is False
        assert d["processing_duration_ms"] > 0
        assert d["circuit_state"] == "closed"
        assert d["request_id"] == req_id

    def test_sensitive_customer_data_strictly_absent_from_logs(self, client, caplog):
        """Verify ticket body, subject, GSTIN, company, card, upi, and secrets are strictly absent from logs."""
        import logging
        caplog.set_level(logging.INFO)

        payload = {
            "id": "T-1010",
            "received_at": "2026-09-06T14:30:00+05:30",
            "subject": "GST invoice required for corporate reimbursement",
            "body": "Need tax invoice for Sharma Tech Solutions Pvt Ltd GSTIN: 27AABCS1429B1ZB. annual plan charge 1499.",
            "purchases": [
                {"id": "pay_J1", "type": "renewal", "amount_inr": 1499, "status": "successful", "at": "2026-09-06T14:00:00+05:30"}
            ],
            "app_opens_since_renewal": 2,
        }
        resp = client.post("/triage", json=payload)
        assert resp.status_code == 200

        # Scan all captured log text and extra attributes
        for record in caplog.records:
            full_str = f"{record.getMessage()} {str(record.__dict__)}"
            # Customer text & company & GST must not appear in any log message or dict
            assert "Sharma Tech Solutions" not in full_str
            assert "27AABCS1429B1ZB" not in full_str
            assert "Need tax invoice for" not in full_str
            assert "corporate reimbursement" not in full_str

    def test_log_sanitizer_layer_filters_blocked_keys(self):
        """Verify sanitize_log_dict purges forbidden sensitive keys from dictionary payloads."""
        from app.core.logging import sanitize_log_dict, BLOCKED_LOG_KEYS

        dirty_payload = {
            "event": "test.event",
            "ticket_id": "T-999",
            "body": "Unsanitized customer text",
            "subject": "Private problem",
            "customer_name": "Jane Doe",
            "email": "jane@example.com",
            "phone": "+919876543210",
            "gstin": "27AABCS1429B1ZB",
            "company": "Acme Corp",
            "card": "4111222233334444",
            "upi": "user@okaxis",
            "prompt": "You are a support bot",
            "system_prompt": "Internal developer prompt",
            "api_key": "sk-secret-token",
            "secret": "top-secret",
            "token": "bearer-token",
            "nested": {
                "safe": 123,
                "password": "secret-password",
            },
        }

        cleaned = sanitize_log_dict(dirty_payload)
        assert cleaned["event"] == "test.event"
        assert cleaned["ticket_id"] == "T-999"
        assert cleaned["nested"] == {"safe": 123}

        # None of the blocked keys may survive
        for key in BLOCKED_LOG_KEYS:
            assert key not in cleaned

    def test_internal_failure_logs_diagnostic_context_with_x_request_id(
        self, client, caplog, monkeypatch
    ):
        """Verify 500 failure logs operational diagnostic stage and returns X-Request-ID without leaking internals."""
        import logging
        caplog.set_level(logging.INFO)

        def blow_up(*args, **kwargs):
            raise RuntimeError("Database connection reset by peer")

        monkeypatch.setattr("app.llm.pipeline.PerceptionPipeline.execute", blow_up)

        payload = {
            "id": "T-FAIL-DIAG",
            "received_at": "2026-09-02T09:14:00+05:30",
            "subject": "Subject",
            "body": "Body",
            "purchases": [],
            "app_opens_since_renewal": 0,
        }
        resp = client.post("/triage", json=payload)
        assert resp.status_code == 500
        req_id = resp.headers.get("X-Request-ID")
        assert req_id is not None

        # Client response must NOT contain internal exception text
        detail = resp.json()["detail"]
        assert "Database connection" not in detail
        assert "reset by peer" not in detail

        # Server logs MUST contain diagnostic context for 3am debugging
        failed_records = [r for r in caplog.records if r.__dict__.get("event") == "triage.failed"]
        assert len(failed_records) >= 1
        d = failed_records[0].__dict__
        assert d["ticket_id"] == "T-FAIL-DIAG"
        assert d["stage"] == "llm"
        assert d["error_type"] == "RuntimeError"
        assert d["request_id"] == req_id
        assert d["processing_duration_ms"] >= 0

    def test_health_endpoint_reports_circuit_state_independently(self, client):
        """Verify GET /health exposes circuit_state without contacting LLM."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "circuit_state" in data
        assert data["circuit_state"] in {"closed", "open", "half_open"}





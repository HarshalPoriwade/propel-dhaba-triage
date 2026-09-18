"""Unit tests for SQLite persistence, idempotency claims, and serialization safety."""

from datetime import datetime, timezone
import pytest

from app.domain.enums import Category, Severity
from app.repositories.triage import (
    ClaimStatus,
    CorruptRecordError,
    ProcessingStatus,
    RecordNotFoundError,
    TriageRepository,
)
from app.schemas.response import RefundResponse, TicketTriageResponse


@pytest.fixture
def repo(tmp_path):
    """Provide a freshly initialized repository backed by a temporary SQLite file."""
    db_file = tmp_path / "test_triage.db"
    repository = TriageRepository(db_path=str(db_file))
    repository.init_db()
    return repository


def make_dummy_response(ticket_id: str, reply_text: str = "Thank you for writing to Dhaba."):
    """Helper to construct a valid TicketTriageResponse contract."""
    return TicketTriageResponse(
        ticket_id=ticket_id,
        category=Category.BILLING,
        severity=Severity.MEDIUM,
        refund=RefundResponse(
            should_refund=True,
            amount_inr=249,
            reason="Approved refund under 24-hour zero-usage policy",
        ),
        reply_draft=reply_text,
        needs_human=False,
        confidence=0.96,
        degraded=False,
        triaged_at=datetime.now(timezone.utc),
    )


def test_db_initialization_creates_schema(repo):
    """Verify that init_db creates the table and allows basic queries."""
    assert repo.count_records() == 0


def test_claim_ticket_new_record(repo):
    """Verify that claiming a new ticket sets IN_PROGRESS and returns CLAIMED."""
    claim = repo.claim_ticket("T-1001")
    assert claim.status == ClaimStatus.CLAIMED
    assert claim.record.ticket_id == "T-1001"
    assert claim.record.status == ProcessingStatus.IN_PROGRESS
    assert claim.record.result_json is None
    assert repo.count_records() == 1


def test_save_completed_result_and_retrieve(repo):
    """Verify saving a completed triage result and deserializing it."""
    repo.claim_ticket("T-1001")
    response_payload = make_dummy_response("T-1001", "We have refunded ₹249.")

    record = repo.save_completed_result("T-1001", response_payload)
    assert record.status == ProcessingStatus.COMPLETED
    assert record.result_json is not None

    # Retrieve and verify exact contract deserialization
    retrieved = repo.get_completed_response("T-1001")
    assert retrieved is not None
    assert retrieved.ticket_id == "T-1001"
    assert retrieved.category == Category.BILLING
    assert retrieved.refund.should_refund is True
    assert retrieved.refund.amount_inr == 249
    assert retrieved.reply_draft == "We have refunded ₹249."
    assert retrieved.confidence == 0.96


def test_idempotency_duplicate_claim_on_in_progress_ticket(repo):
    """Verify that a second claim on an IN_PROGRESS ticket returns IN_PROGRESS and creates no duplicates."""
    claim1 = repo.claim_ticket("T-1001")
    assert claim1.status == ClaimStatus.CLAIMED

    # Concurrent request attempting to claim the same ticket
    claim2 = repo.claim_ticket("T-1001")
    assert claim2.status == ClaimStatus.IN_PROGRESS
    assert claim2.record.ticket_id == "T-1001"
    assert repo.count_records() == 1  # No duplicate row created


def test_idempotency_duplicate_claim_on_completed_ticket(repo):
    """Verify that a second claim on a COMPLETED ticket returns ALREADY_COMPLETED with stored result."""
    repo.claim_ticket("T-1001")
    response_payload = make_dummy_response("T-1001", "Original processed reply.")
    repo.save_completed_result("T-1001", response_payload)

    # Subsequent request arrives with the same ticket ID
    claim_dup = repo.claim_ticket("T-1001")
    assert claim_dup.status == ClaimStatus.ALREADY_COMPLETED
    assert claim_dup.record.status == ProcessingStatus.COMPLETED

    # The stored result can be parsed cleanly
    cached_response = claim_dup.record.parse_response()
    assert cached_response is not None
    assert cached_response.ticket_id == "T-1001"
    assert cached_response.reply_draft == "Original processed reply."
    assert repo.count_records() == 1


def test_save_completed_on_nonexistent_ticket_raises_error(repo):
    """Verify updating a non-existent ticket raises RecordNotFoundError."""
    response_payload = make_dummy_response("T-9999")
    with pytest.raises(RecordNotFoundError):
        repo.save_completed_result("T-9999", response_payload)


def test_parameterized_query_safely_handles_special_characters(repo):
    """Verify that quotes, apostrophes, and SQL-like strings are safely stored as data."""
    repo.claim_ticket("T-1002")
    tricky_reply = "Customer said: 'Robert'); DROP TABLE triage_records; -- \U0001f600 \u20b9"
    response_payload = make_dummy_response("T-1002", tricky_reply)

    repo.save_completed_result("T-1002", response_payload)
    retrieved = repo.get_completed_response("T-1002")
    assert retrieved is not None
    assert retrieved.reply_draft == tricky_reply
    # Ensure table was not dropped
    assert repo.count_records() == 1


def test_corrupt_stored_json_raises_persistence_error(repo):
    """Verify that corrupted or invalid JSON in storage triggers CorruptRecordError."""
    repo.claim_ticket("T-1003")

    # Manually inject corrupt JSON directly via connection to simulate bit-rot/tampering
    with repo._connection() as conn:
        conn.execute(
            "UPDATE triage_records SET status = 'completed', result_json = 'NOT_JSON{{{' WHERE ticket_id = 'T-1003'"
        )

    with pytest.raises(CorruptRecordError) as exc_info:
        repo.get_completed_response("T-1003")
    assert "corrupt" in str(exc_info.value).lower()


def test_schema_mismatched_stored_json_raises_persistence_error(repo):
    """Verify that JSON missing required fields triggers CorruptRecordError."""
    repo.claim_ticket("T-1004")

    # Valid JSON, but missing required TicketTriageResponse fields
    with repo._connection() as conn:
        conn.execute(
            "UPDATE triage_records SET status = 'completed', result_json = '{\"ticket_id\": \"T-1004\"}' WHERE ticket_id = 'T-1004'"
        )

    with pytest.raises(CorruptRecordError) as exc_info:
        repo.get_completed_response("T-1004")
    assert "corrupt" in str(exc_info.value).lower()


def test_get_completed_response_returns_none_for_missing_or_in_progress(repo):
    """Verify get_completed_response returns None if ticket is missing or in progress."""
    assert repo.get_completed_response("T-NONEXISTENT") is None

    repo.claim_ticket("T-IN_PROG")
    assert repo.get_completed_response("T-IN_PROG") is None

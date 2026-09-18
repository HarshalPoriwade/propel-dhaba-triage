"""SQLite-backed persistence and idempotency repository for Dhaba ticket triage."""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import sqlite3
from typing import Generator, Optional

from app.core.config import get_settings
from app.schemas.response import TicketTriageResponse


class ProcessingStatus(str, Enum):
    """Lifecycle status of a triage record in storage."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class ClaimStatus(str, Enum):
    """Result of attempting to claim a ticket for triage processing."""

    CLAIMED = "claimed"                   # Newly claimed; caller should proceed with triage
    ALREADY_COMPLETED = "already_completed"  # Stored result exists; caller should replay
    IN_PROGRESS = "in_progress"             # Another worker/request is actively triaging


class RepositoryError(Exception):
    """Base exception for persistence operations."""


class RecordNotFoundError(RepositoryError):
    """Raised when an expected triage record does not exist in storage."""


class CorruptRecordError(RepositoryError):
    """Raised when stored JSON cannot be parsed or validated against the schema."""


@dataclass(frozen=True)
class TriageRecord:
    """Immutable domain representation of a persisted triage database row."""

    ticket_id: str
    status: ProcessingStatus
    result_json: Optional[str]
    created_at: str
    updated_at: str

    def parse_response(self) -> Optional[TicketTriageResponse]:
        """Safely deserialize and validate stored JSON into a TicketTriageResponse.

        Raises:
            CorruptRecordError: If result_json is present but cannot be validated.
        """
        if self.result_json is None:
            return None
        try:
            return TicketTriageResponse.model_validate_json(self.result_json)
        except Exception as err:
            raise CorruptRecordError(
                f"Stored JSON for ticket {self.ticket_id} is corrupt: {err}"
            ) from err


@dataclass(frozen=True)
class ClaimResult:
    """Outcome of attempting to claim a ticket identifier."""

    status: ClaimStatus
    record: TriageRecord


class TriageRepository:
    """Repository managing triage persistence and atomic idempotency claims via SQLite."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize repository with target SQLite database path.

        Args:
            db_path: Path to the SQLite file. If None, resolves from Settings.
        """
        if db_path is None:
            db_path = get_settings().DATABASE_PATH
        self.db_path = db_path

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing a configured SQLite connection inside an ACID transaction."""
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            # WAL mode allows concurrent readers while a writer commits
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            conn.execute("PRAGMA foreign_keys = ON;")
            with conn:
                yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        """Initialize database schema with strict unique constraints."""
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS triage_records (
                    ticket_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            # Create an index on status for operational triage queries
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_triage_records_status
                ON triage_records (status);
                """
            )

    def claim_ticket(self, ticket_id: str) -> ClaimResult:
        """Atomically attempt to claim a ticket for triage.

        Uses the PRIMARY KEY uniqueness constraint as the authoritative barrier against
        concurrent processing races.

        Returns:
            ClaimResult with status:
            - CLAIMED if this request successfully claimed the ticket.
            - ALREADY_COMPLETED if a completed record already exists for this ticket.
            - IN_PROGRESS if another concurrent request is currently triaging it.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            with self._connection() as conn:
                conn.execute(
                    """
                    INSERT INTO triage_records (
                        ticket_id, status, result_json, created_at, updated_at
                    ) VALUES (?, ?, NULL, ?, ?)
                    """,
                    (ticket_id, ProcessingStatus.IN_PROGRESS.value, now_iso, now_iso),
                )
            # Insert succeeded: caller owns this ticket
            return ClaimResult(
                status=ClaimStatus.CLAIMED,
                record=TriageRecord(
                    ticket_id=ticket_id,
                    status=ProcessingStatus.IN_PROGRESS,
                    result_json=None,
                    created_at=now_iso,
                    updated_at=now_iso,
                ),
            )
        except sqlite3.IntegrityError:
            # Race condition caught: row already exists in database
            existing = self.get_record(ticket_id)
            if existing is None:
                raise RepositoryError(
                    f"Integrity conflict occurred but record '{ticket_id}' could not be read."
                )

            if existing.status == ProcessingStatus.COMPLETED:
                return ClaimResult(
                    status=ClaimStatus.ALREADY_COMPLETED,
                    record=existing,
                )
            return ClaimResult(
                status=ClaimStatus.IN_PROGRESS,
                record=existing,
            )

    def save_completed_result(
        self, ticket_id: str, result: TicketTriageResponse
    ) -> TriageRecord:
        """Persist the completed triage response for a claimed ticket.

        Args:
            ticket_id: Ticket ID to mark completed.
            result: The completed TicketTriageResponse contract.

        Returns:
            Updated TriageRecord.

        Raises:
            RecordNotFoundError: If ticket_id does not exist.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        result_json = result.model_dump_json()

        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE triage_records
                SET status = ?, result_json = ?, updated_at = ?
                WHERE ticket_id = ?
                """,
                (ProcessingStatus.COMPLETED.value, result_json, now_iso, ticket_id),
            )
            if cursor.rowcount == 0:
                raise RecordNotFoundError(
                    f"Cannot complete triage for '{ticket_id}': record does not exist."
                )

        record = self.get_record(ticket_id)
        if record is None:
            raise RecordNotFoundError(f"Record '{ticket_id}' vanished after update.")
        return record

    def get_record(self, ticket_id: str) -> Optional[TriageRecord]:
        """Fetch a triage record by ticket ID.

        Returns:
            TriageRecord if found, None otherwise.
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT ticket_id, status, result_json, created_at, updated_at
                FROM triage_records
                WHERE ticket_id = ?
                """,
                (ticket_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return TriageRecord(
                ticket_id=row["ticket_id"],
                status=ProcessingStatus(row["status"]),
                result_json=row["result_json"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    def get_completed_response(self, ticket_id: str) -> Optional[TicketTriageResponse]:
        """Fetch and deserialize completed triage response.

        Returns:
            Validated TicketTriageResponse if ticket exists and is completed.
            None if ticket is not found or is still in_progress.

        Raises:
            CorruptRecordError: If stored result_json is corrupted.
        """
        record = self.get_record(ticket_id)
        if record is None or record.status != ProcessingStatus.COMPLETED:
            return None
        return record.parse_response()

    def count_records(self) -> int:
        """Count total stored triage records."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM triage_records")
            row = cursor.fetchone()
            return int(row[0]) if row else 0

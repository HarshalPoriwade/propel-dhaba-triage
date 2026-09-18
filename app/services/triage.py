"""Application service orchestrating support ticket triage."""

from datetime import datetime, timezone
from typing import Optional

from app.domain.enums import Severity
from app.domain.models import TriageResult
from app.llm.pipeline import PerceptionPipeline
from app.policies.refund import RefundPolicyConfig, evaluate_refund_policy
from app.repositories.triage import (
    ClaimStatus,
    CorruptRecordError,
    TriageRepository,
)
from app.schemas.request import TicketTriageRequest
from app.schemas.response import TicketTriageResponse
from app.services.reply import generate_final_reply


class TriageServiceError(Exception):
    """Base exception for triage application service."""


class TicketInProgressError(TriageServiceError):
    """Raised when a ticket is actively being processed by another worker."""


class TriageService:
    """Orchestrates ticket ingestion, idempotency, LLM perception, refund policy, and persistence."""

    def __init__(
        self,
        repository: TriageRepository,
        pipeline: PerceptionPipeline,
        policy_config: Optional[RefundPolicyConfig] = None,
    ):
        """Initialize service with required repositories, pipelines, and policy configurations.

        Args:
            repository: TriageRepository managing atomic claims and SQLite persistence.
            pipeline: PerceptionPipeline managing LLM perception, retries, and degradation.
            policy_config: Deterministic refund policy configuration rules.
        """
        self.repository = repository
        self.pipeline = pipeline
        self.policy_config = policy_config or RefundPolicyConfig()

    async def triage_ticket(
        self, request: TicketTriageRequest
    ) -> TicketTriageResponse:
        """Process an incoming support ticket through the end-to-end triage pipeline.

        Idempotency guarantee:
        - The ticket ID is the idempotency key.
        - Repeated calls for an already-completed ticket return the stored response
          immediately without re-running the LLM or re-evaluating the refund policy.
        - Concurrent calls for a ticket currently in progress raise TicketInProgressError.

        Financial safety guarantee:
        - The LLM has zero financial authority.
        - Monetary values are derived solely from trusted billing records via evaluate_refund_policy.

        Args:
            request: Validated HTTP request payload.

        Returns:
            TicketTriageResponse contract.

        Raises:
            TicketInProgressError: If another concurrent process is currently triaging the ticket.
            RepositoryError: If database persistence fails.
        """
        # 1. Convert validated request into domain Ticket
        ticket = request.to_domain()

        # 2. Atomic idempotency claim via SQLite
        claim = self.repository.claim_ticket(ticket.id)

        # 3. Handle already-completed ticket: return stored result immediately
        if claim.status == ClaimStatus.ALREADY_COMPLETED:
            stored_response = claim.record.parse_response()
            if stored_response is None:
                raise CorruptRecordError(
                    f"Completed record for ticket '{ticket.id}' contains null result."
                )
            return stored_response

        # 4. Handle concurrent in-progress ticket
        if claim.status == ClaimStatus.IN_PROGRESS:
            raise TicketInProgressError(
                f"Ticket '{ticket.id}' is currently being triaged by another request."
            )

        # 5. Successfully claimed: proceed with full triage execution
        try:
            # 5a. Run resilient LLM perception pipeline (handles retry and safe degradation)
            pipeline_result = await self.pipeline.execute(ticket)
            perception = pipeline_result.perception

            # 5b. Deterministic refund policy gate (LLM CANNOT AUTHORIZE FUNDS)
            refund_result = evaluate_refund_policy(
                ticket=ticket,
                perception=perception,
                config=self.policy_config,
            )

            # 5c. Escalation determination (needs_human)
            needs_human = (
                refund_result.requires_human_review
                or pipeline_result.degraded
                or perception.suggested_escalation
                or perception.severity == Severity.CRITICAL
                or perception.confidence < 0.70
            )

            # 5d. Finalize user-facing customer reply
            final_reply = generate_final_reply(
                perception=perception,
                refund_result=refund_result,
                ticket=ticket,
            )

            # 5e. Assemble domain TriageResult
            now_utc = datetime.now(timezone.utc)
            domain_result = TriageResult(
                ticket_id=ticket.id,
                category=perception.category,
                severity=perception.severity,
                refund=refund_result.to_domain_decision(),
                reply_draft=final_reply,
                needs_human=needs_human,
                confidence=perception.confidence,
                degraded=pipeline_result.degraded,
                triaged_at=now_utc,
            )

            # 5f. Convert to API response schema
            response = TicketTriageResponse.from_domain(domain_result)

            # 6. Persist completed triage record
            self.repository.save_completed_result(ticket.id, response)

            return response

        except Exception:
            # If processing or persistence failed after claiming, release in_progress claim
            try:
                self.repository.release_claim(ticket.id)
            except Exception:
                # Do not allow cleanup failure to mask the original processing exception
                pass
            raise


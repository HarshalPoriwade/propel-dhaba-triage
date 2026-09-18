"""HTTP response schema contracts returned by the Dhaba triage service."""

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import Category, Severity
from app.domain.models import TriageResult


class RefundResponse(BaseModel):
    """Deterministic refund determination for support tools to act upon.

    IMPORTANT FINANCIAL SAFETY BOUNDARY:
    This model represents an analytical refund eligibility decision produced by
    the application's deterministic business policy gate. It does NOT assert that
    a bank transfer has already been executed.
    """

    model_config = ConfigDict(extra="forbid")

    should_refund: bool = Field(
        description="Whether a refund is authorized under documented business policy"
    )
    amount_inr: int = Field(
        ge=0,
        description="Authorized refund amount in INR (0 if no refund authorized)"
    )
    reason: str = Field(
        min_length=1,
        description="Deterministic policy justification for the refund decision"
    )


class TicketTriageResponse(BaseModel):
    """Final triage response contract matching Propel Task 1 specification."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str = Field(
        min_length=1,
        description="Unique identifier of the triaged ticket"
    )
    category: Category = Field(
        description="One of the small category enums defined by the system"
    )
    severity: Severity = Field(
        description="Assigned severity level on documented scale"
    )
    refund: RefundResponse = Field(
        description="Refund decision detailing whether to refund, how much, and why"
    )
    reply_draft: str = Field(
        min_length=1,
        description="Draft reply prepared for the customer in their language"
    )
    needs_human: bool = Field(
        description="True when human review or intervention is mandatory"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Classification confidence score between 0.0 and 1.0"
    )
    degraded: bool = Field(
        default=False,
        description="True if the response was generated via deterministic fallback degradation"
    )
    triaged_at: datetime = Field(
        description="ISO8601 timestamp when triage analysis was completed"
    )

    @classmethod
    def from_domain(cls, result: TriageResult) -> "TicketTriageResponse":
        """Construct response contract from domain TriageResult."""
        return cls(
            ticket_id=result.ticket_id,
            category=result.category,
            severity=result.severity,
            refund=RefundResponse(
                should_refund=result.refund.should_refund,
                amount_inr=result.refund.amount_inr,
                reason=result.refund.reason,
            ),
            reply_draft=result.reply_draft,
            needs_human=result.needs_human,
            confidence=result.confidence,
            degraded=result.degraded,
            triaged_at=result.triaged_at,
        )

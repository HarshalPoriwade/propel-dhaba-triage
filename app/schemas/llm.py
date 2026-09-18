"""Intermediate structured output schema for the LLM perception layer.

CRITICAL ARCHITECTURAL BOUNDARY:
The LLM is an untrusted interpretation component. It extracts intent,
language, sentiment, and drafts customer replies.

THE LLM DOES NOT POSSESS FINANCIAL AUTHORITY.
This schema strictly forbids fields such as:
- refund_amount
- refund_inr
- should_refund
- execute_refund
- payment_action

Any financial determinations are calculated solely by the deterministic
RefundPolicyGate in the application layer.
"""

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import Category, Severity


class LLMPerceptionOutput(BaseModel):
    """Structured perception extracted from untrusted customer ticket prose."""

    model_config = ConfigDict(extra="forbid")

    category: Category = Field(
        description="Perceived ticket category based on customer intent"
    )
    severity: Severity = Field(
        description="Perceived urgency and severity based on customer language"
    )
    user_requested_refund: bool = Field(
        description="True if customer explicitly or implicitly requested a refund in their text. "
        "NOTE: This indicates user intent only; it does NOT grant financial authorization."
    )
    claimed_issue: str = Field(
        min_length=1,
        max_length=500,
        description="One-sentence interpretation of what the customer is experiencing"
    )
    detected_language: str = Field(
        default="en",
        max_length=20,
        description="Detected language or dialect (e.g. 'en', 'hinglish')"
    )
    suggested_escalation: bool = Field(
        default=False,
        description="LLM suggestion on whether a human agent is needed based on ambiguity or sentiment"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model self-assessed confidence in understanding the ticket [0.0, 1.0]"
    )
    draft_reply: str = Field(
        min_length=1,
        max_length=2000,
        description="Initial response draft addressing the customer in their language without making unauthorized promises"
    )

"""Customer reply drafting and validation service.

Guarantees that customer-facing reply drafts align with the deterministic policy
outcome. Under no circumstances may a draft claim a refund was executed or authorized
unless supported by the deterministic refund policy.
"""

from app.domain.models import Ticket
from app.policies.refund import RefundPolicyResult
from app.schemas.llm import LLMPerceptionOutput


def generate_final_reply(
    perception: LLMPerceptionOutput,
    refund_result: RefundPolicyResult,
    ticket: Ticket,
) -> str:
    """Generate or sanitize customer reply based on perception and policy outcome.

    Args:
        perception: Validated LLM perception containing candidate draft_reply.
        refund_result: Deterministic refund decision.
        ticket: Customer ticket context.

    Returns:
        Customer-facing message string honoring customer language without
        making unauthorized financial promises.
    """
    raw_draft = (perception.draft_reply or "").strip()

    # Unauthorized refund claim markers
    unauthorized_refund_markers = (
        "refund has been processed",
        "refund is processed",
        "money has been refunded",
        "money is refunded",
        "amount has been refunded",
        "refund has been issued",
        "refund of rs",
        "refund approved",
        "we have refunded",
        "credited back",
        "returned to your bank",
    )

    # 1. If policy did NOT authorize a refund, ensure no false execution claim was hallucinated
    if not refund_result.should_refund:
        has_unauthorized_claim = any(
            marker in raw_draft.lower() for marker in unauthorized_refund_markers
        )
        if has_unauthorized_claim:
            lang = (perception.detected_language or "en").lower()
            if "hindi" in lang or "hinglish" in lang:
                return (
                    "Hum aapke ticket aur transaction details ki jaanch kar rahe hain. "
                    "Hamari billing aur support team jald hi aapse sampark karegi."
                )
            return (
                "We have received your request and our support team is reviewing your billing "
                "and account history. We will update you as soon as the review is complete."
            )

    # 2. If policy DID authorize a refund, ensure customer is clearly informed
    if refund_result.should_refund:
        lower_draft = raw_draft.lower()
        if "refund" not in lower_draft or "authorized" not in lower_draft:
            return (
                f"{raw_draft} An automatic refund of INR {refund_result.amount_inr} has been authorized "
                f"under our subscription renewal policy."
            ).strip()

    return raw_draft or "We have received your ticket and our team is investigating."

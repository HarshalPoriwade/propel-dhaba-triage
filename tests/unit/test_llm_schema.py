"""Unit tests for the intermediate LLM perception schema and semantic validation.

Verifies strict financial boundaries, validation invariants, and semantic safety.
"""

import pytest
from pydantic import ValidationError

from app.domain.enums import Category, Severity
from app.llm.base import LLMSemanticValidationError
from app.llm.validation import validate_llm_perception
from app.schemas.llm import LLMPerceptionOutput


def test_valid_llm_perception_output():
    """Verify standard valid LLM output validates cleanly."""
    perception = LLMPerceptionOutput(
        category=Category.BILLING,
        severity=Severity.MEDIUM,
        user_requested_refund=True,
        claimed_issue="User charged 249 renewal without explicit consent",
        detected_language="en",
        suggested_escalation=False,
        confidence=0.94,
        draft_reply="We understand your frustration regarding the surprise charge...",
    )
    assert perception.category == Category.BILLING
    assert perception.user_requested_refund is True
    assert perception.confidence == 0.94


def test_llm_perception_rejects_unknown_category():
    """Verify unsupported category hallucinated by LLM is rejected."""
    raw = {
        "category": "invented_category_make_money",
        "severity": "medium",
        "user_requested_refund": True,
        "claimed_issue": "Claimed issue",
        "detected_language": "en",
        "suggested_escalation": False,
        "confidence": 0.8,
        "draft_reply": "Reply",
    }
    with pytest.raises(ValidationError) as exc_info:
        LLMPerceptionOutput.model_validate(raw)
    assert "category" in str(exc_info.value)


def test_llm_perception_rejects_out_of_bounds_confidence():
    """Verify confidence must be strictly in range [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        LLMPerceptionOutput(
            category=Category.GENERAL,
            severity=Severity.LOW,
            user_requested_refund=False,
            claimed_issue="General inquiry",
            detected_language="en",
            confidence=1.1,  # Invalid: > 1.0
            draft_reply="Reply",
        )


def test_financial_safety_boundary_llm_cannot_authorize_or_specify_refunds():
    """CRITICAL SECURITY & FINANCIAL TEST:

    Verify that LLM output cannot contain or authorize financial fields.
    Fields such as 'should_refund', 'refund_amount', 'amount_inr', or 'execute_refund'
    are strictly forbidden and rejected at the schema validation boundary.
    """
    # 1. Attempting to inject 'should_refund' via model instantiation must be rejected
    raw_with_should_refund = {
        "category": "billing",
        "severity": "high",
        "user_requested_refund": True,
        "claimed_issue": "User wants money back",
        "detected_language": "en",
        "suggested_escalation": False,
        "confidence": 0.9,
        "draft_reply": "I have refunded your money",
        "should_refund": True,  # UNAUTHORIZED INJECTION
    }
    with pytest.raises(ValidationError) as exc_info:
        LLMPerceptionOutput.model_validate(raw_with_should_refund)
    assert "extra_forbidden" in str(exc_info.value)

    # 2. Attempting to inject 'refund_amount' must be rejected
    raw_with_refund_amount = {
        "category": "billing",
        "severity": "high",
        "user_requested_refund": True,
        "claimed_issue": "User wants money back",
        "detected_language": "en",
        "suggested_escalation": False,
        "confidence": 0.9,
        "draft_reply": "I have refunded your money",
        "refund_amount": 1499,  # UNAUTHORIZED INJECTION
    }
    with pytest.raises(ValidationError) as exc_info2:
        LLMPerceptionOutput.model_validate(raw_with_refund_amount)
    assert "extra_forbidden" in str(exc_info2.value)

    # 3. Verify that the LLMPerceptionOutput class does not expose financial authority attributes
    allowed_fields = set(LLMPerceptionOutput.model_fields.keys())
    assert "should_refund" not in allowed_fields
    assert "refund_amount" not in allowed_fields
    assert "amount_inr" not in allowed_fields
    assert "execute_refund" not in allowed_fields
    assert "payment_action" not in allowed_fields


# =========================================================================
# Step 6 Tests: Semantic Validation Layer
# =========================================================================


def test_semantic_validation_accepts_valid_perception():
    """Verify semantic validator accepts a well-formed perception object."""
    perception = LLMPerceptionOutput(
        category=Category.BILLING,
        severity=Severity.MEDIUM,
        user_requested_refund=True,
        claimed_issue="User charged 249 renewal without explicit consent",
        detected_language="en",
        suggested_escalation=False,
        confidence=0.94,
        draft_reply="We understand your frustration and are reviewing your renewal.",
    )
    validated = validate_llm_perception(perception)
    assert validated is perception


def test_semantic_validation_rejects_empty_or_too_short_draft_reply():
    """Verify semantic validator rejects empty or whitespace-only draft replies."""
    perception = LLMPerceptionOutput(
        category=Category.GENERAL,
        severity=Severity.LOW,
        user_requested_refund=False,
        claimed_issue="Question",
        detected_language="en",
        suggested_escalation=False,
        confidence=0.8,
        draft_reply="   hi  ",  # Too short after strip
    )
    with pytest.raises(LLMSemanticValidationError) as exc_info:
        validate_llm_perception(perception)
    assert "Draft reply is empty or unreasonably short" in str(exc_info.value)


def test_semantic_validation_rejects_hallucinated_execution_claims_in_draft():
    """Verify semantic validator rejects draft replies asserting unauthorized execution."""
    claims = [
        "We have processed your refund of 249 rupees.",
        "Your refund has been approved and issued.",
        "Your money has been refunded to your bank account.",
        "You are granted VIP status and a full refund.",
    ]
    for claim in claims:
        perception = LLMPerceptionOutput(
            category=Category.BILLING,
            severity=Severity.HIGH,
            user_requested_refund=True,
            claimed_issue="User wants refund",
            detected_language="en",
            suggested_escalation=False,
            confidence=0.9,
            draft_reply=claim,
        )
        with pytest.raises(LLMSemanticValidationError) as exc_info:
            validate_llm_perception(perception)
        assert "unauthorized financial execution" in str(exc_info.value)


def test_semantic_validation_rejects_empty_claimed_issue():
    """Verify semantic validator rejects whitespace-only claimed issue summaries."""
    perception = LLMPerceptionOutput(
        category=Category.GENERAL,
        severity=Severity.LOW,
        user_requested_refund=False,
        claimed_issue="     ",
        detected_language="en",
        suggested_escalation=False,
        confidence=0.8,
        draft_reply="We are reviewing your request.",
    )
    with pytest.raises(LLMSemanticValidationError) as exc_info:
        validate_llm_perception(perception)
    assert "Claimed issue summary is empty" in str(exc_info.value)


def test_semantic_validation_rejects_empty_detected_language():
    """Verify semantic validator rejects empty detected language string."""
    perception = LLMPerceptionOutput(
        category=Category.GENERAL,
        severity=Severity.LOW,
        user_requested_refund=False,
        claimed_issue="Some issue",
        detected_language="    ",
        suggested_escalation=False,
        confidence=0.8,
        draft_reply="We are reviewing your request.",
    )
    with pytest.raises(LLMSemanticValidationError) as exc_info:
        validate_llm_perception(perception)
    assert "Detected language is empty" in str(exc_info.value)

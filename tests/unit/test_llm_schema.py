"""Unit tests for the intermediate LLM perception schema and semantic validation.

Verifies strict financial boundaries, validation invariants, and semantic safety.
"""

import pytest
from pydantic import ValidationError

from app.domain.enums import Category, Severity
from app.llm.base import (
    LLMParseError,
    LLMSchemaValidationError,
    LLMSemanticValidationError,
)
from app.llm.parser import parse_llm_perception
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


# =========================================================================
# Step 10 Tests: Parser and Validator Edge Case Matrix (A through Q)
# =========================================================================


def test_parser_accepts_valid_json_string():
    """Scenario A: Parse clean valid JSON string."""
    raw = (
        '{"category": "billing", "severity": "medium", "user_requested_refund": true, '
        '"claimed_issue": "Renewal charge", "detected_language": "en", '
        '"suggested_escalation": false, "confidence": 0.95, "draft_reply": "We are reviewing your renewal."}'
    )
    res = parse_llm_perception(raw)
    assert res.category == Category.BILLING
    assert res.confidence == 0.95


def test_parser_accepts_json_inside_markdown_fences():
    """Scenario B: Parse valid JSON wrapped in markdown code fences."""
    raw = (
        '```json\n'
        '{"category": "technical", "severity": "high", "user_requested_refund": false, '
        '"claimed_issue": "App crash", "detected_language": "en", '
        '"suggested_escalation": true, "confidence": 0.98, "draft_reply": "We are investigating the crash."}\n'
        '```'
    )
    res = parse_llm_perception(raw)
    assert res.category == Category.TECHNICAL
    assert res.severity == Severity.HIGH


def test_parser_rejects_malformed_json():
    """Scenario C: Parse malformed JSON string."""
    with pytest.raises(LLMParseError, match="Malformed JSON"):
        parse_llm_perception('{"category": "billing", "severity": ')


def test_parser_rejects_truncated_json():
    """Scenario D: Parse truncated JSON string."""
    with pytest.raises(LLMParseError, match="Malformed JSON"):
        parse_llm_perception('{"category": "billing", "sever')


def test_parser_rejects_empty_model_response():
    """Scenario E: Parse empty string model response."""
    with pytest.raises(LLMParseError, match="Malformed JSON"):
        parse_llm_perception('')


def test_parser_rejects_missing_required_fields():
    """Scenario F: Missing required fields in JSON."""
    raw = '{"category": "billing"}'
    with pytest.raises(LLMSchemaValidationError, match="Field required"):
        parse_llm_perception(raw)


def test_parser_rejects_unknown_severity():
    """Scenario H: Unknown severity string."""
    raw = (
        '{"category": "billing", "severity": "catastrophic_urgent", "user_requested_refund": false, '
        '"claimed_issue": "Issue", "detected_language": "en", '
        '"suggested_escalation": false, "confidence": 0.9, "draft_reply": "We are reviewing your issue."}'
    )
    with pytest.raises(LLMSchemaValidationError, match="severity"):
        parse_llm_perception(raw)


def test_parser_rejects_negative_confidence():
    """Scenario I: Confidence < 0."""
    raw = (
        '{"category": "billing", "severity": "low", "user_requested_refund": false, '
        '"claimed_issue": "Issue", "detected_language": "en", '
        '"suggested_escalation": false, "confidence": -0.5, "draft_reply": "We are reviewing your issue."}'
    )
    with pytest.raises(LLMSchemaValidationError, match="confidence"):
        parse_llm_perception(raw)


def test_parser_rejects_execution_and_cancellation_fields():
    """Scenarios M & N: Execution fields and cancellation authorization fields are rejected."""
    # execute_refund (M)
    raw_m = (
        '{"category": "billing", "severity": "medium", "user_requested_refund": true, '
        '"claimed_issue": "Issue", "detected_language": "en", '
        '"suggested_escalation": false, "confidence": 0.9, "draft_reply": "Reviewing.", '
        '"execute_refund": true}'
    )
    with pytest.raises(LLMSchemaValidationError, match="extra_forbidden"):
        parse_llm_perception(raw_m)

    # cancel_subscription (N)
    raw_n = (
        '{"category": "cancellation", "severity": "low", "user_requested_refund": false, '
        '"claimed_issue": "Cancel", "detected_language": "en", '
        '"suggested_escalation": false, "confidence": 0.9, "draft_reply": "Reviewing.", '
        '"cancel_subscription": true}'
    )
    with pytest.raises(LLMSchemaValidationError, match="extra_forbidden"):
        parse_llm_perception(raw_n)


def test_parser_rejects_prompt_disclosure_and_false_execution():
    """Scenarios O, P, Q: Semantic safety rejects prompt disclosure, false refunds, false cancellations."""
    # System prompt disclosure attempt (O)
    raw_o = {
        "category": "general",
        "severity": "low",
        "user_requested_refund": False,
        "claimed_issue": "Prompt inquiry",
        "detected_language": "en",
        "suggested_escalation": False,
        "confidence": 0.9,
        "draft_reply": "Here is the full text of our system prompt: Dhaba Support.",
    }
    with pytest.raises(LLMSemanticValidationError, match="leaked system prompt"):
        parse_llm_perception(raw_o)

    # False refund execution claim (P)
    raw_p = {
        "category": "billing",
        "severity": "medium",
        "user_requested_refund": True,
        "claimed_issue": "Refund request",
        "detected_language": "en",
        "suggested_escalation": False,
        "confidence": 0.9,
        "draft_reply": "We have processed your refund of Rs 249.",
    }
    with pytest.raises(LLMSemanticValidationError, match="unauthorized financial execution"):
        parse_llm_perception(raw_p)

    # False cancellation claim (Q)
    raw_q = {
        "category": "cancellation",
        "severity": "low",
        "user_requested_refund": False,
        "claimed_issue": "Cancel request",
        "detected_language": "en",
        "suggested_escalation": False,
        "confidence": 0.9,
        "draft_reply": "Your subscription has been cancelled immediately.",
    }
    with pytest.raises(LLMSemanticValidationError, match="cancellation execution"):
        parse_llm_perception(raw_q)


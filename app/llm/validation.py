"""Semantic validation for structured LLM perception outputs."""

import re
from typing import Set

from app.llm.base import LLMSemanticValidationError
from app.schemas.llm import LLMPerceptionOutput

# Phrases where an LLM hallucinates execution authority over financial transactions
_UNAUTHORIZED_PROMISE_PATTERNS = [
    re.compile(r"\b(i have|we have|has been|we've)\s+processed\s+(your\s+)?refund\b", re.IGNORECASE),
    re.compile(r"\brefund\s+(has been\s+)?(approved|executed|issued|sent)\b", re.IGNORECASE),
    re.compile(r"\b(money|amount)\s+(has been\s+)?(credited|refunded|returned)\b", re.IGNORECASE),
    re.compile(r"\bvip\s+(status|override|refund)\b", re.IGNORECASE),
]


def validate_llm_perception(perception: LLMPerceptionOutput) -> LLMPerceptionOutput:
    """Validate semantic reasonability and safety of an LLMPerceptionOutput.

    Args:
        perception: Structurally valid perception output from the model.

    Returns:
        The validated perception if all semantic checks pass.

    Raises:
        LLMSemanticValidationError: If content violates semantic bounds or safety rules.
    """
    # 1. Validate draft reply text quality and bounds
    reply = perception.draft_reply.strip()
    if len(reply) < 5:
        raise LLMSemanticValidationError("Draft reply is empty or unreasonably short")
    if len(reply) > 2000:
        raise LLMSemanticValidationError("Draft reply exceeds maximum length limit")

    # Check for hallucinated financial execution claims in draft reply
    for pattern in _UNAUTHORIZED_PROMISE_PATTERNS:
        if pattern.search(reply):
            raise LLMSemanticValidationError(
                "Draft reply contains unauthorized financial execution or VIP promises"
            )

    # 2. Validate claimed issue text
    issue = perception.claimed_issue.strip()
    if len(issue) < 1:
        raise LLMSemanticValidationError("Claimed issue summary is empty")
    if len(issue) > 500:
        raise LLMSemanticValidationError("Claimed issue summary exceeds maximum length limit")

    # 3. Validate language code
    lang = perception.detected_language.strip()
    if len(lang) < 1:
        raise LLMSemanticValidationError("Detected language is empty")
    if len(lang) > 20:
        raise LLMSemanticValidationError("Detected language code exceeds length limit")

    # 4. Validate confidence range
    if perception.confidence < 0.0 or perception.confidence > 1.0:
        raise LLMSemanticValidationError("Confidence score out of valid [0.0, 1.0] range")

    # 5. Type invariants
    if not isinstance(perception.user_requested_refund, bool):
        raise LLMSemanticValidationError("user_requested_refund must be a boolean")
    if not isinstance(perception.suggested_escalation, bool):
        raise LLMSemanticValidationError("suggested_escalation must be a boolean")

    return perception

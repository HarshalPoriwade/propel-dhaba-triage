"""Parser converting raw LLM completion text into validated LLMPerceptionOutput."""

import json
import re
from typing import Any, Union
from pydantic import ValidationError

from app.llm.base import LLMParseError, LLMSchemaValidationError
from app.llm.validation import validate_llm_perception
from app.schemas.llm import LLMPerceptionOutput

# Regex to extract JSON from markdown code fences if emitted by the model
_CODE_BLOCK_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def parse_llm_perception(raw_content: Union[str, dict, Any]) -> LLMPerceptionOutput:
    """Parse, structurally validate, and semantically check raw model output.

    Pipeline:
        raw content -> JSON decoding -> Pydantic schema validation -> Semantic validation

    Args:
        raw_content: Raw string or dictionary from the LLM provider.

    Returns:
        Validated and semantically verified LLMPerceptionOutput.

    Raises:
        LLMParseError: If JSON decoding fails.
        LLMSchemaValidationError: If structural validation against LLMPerceptionOutput fails.
        LLMSemanticValidationError: If semantic reasonability checks fail.
    """
    if isinstance(raw_content, dict):
        parsed_dict = raw_content
    elif isinstance(raw_content, str):
        content_str = raw_content.strip()
        # Strip markdown code fences if present
        match = _CODE_BLOCK_PATTERN.match(content_str)
        if match:
            content_str = match.group(1).strip()

        try:
            parsed_dict = json.loads(content_str)
        except json.JSONDecodeError as err:
            raise LLMParseError(f"Malformed JSON returned by LLM provider: {err}") from err
    else:
        raise LLMParseError(
            f"Unexpected LLM output type: expected str or dict, got {type(raw_content).__name__}"
        )

    if not isinstance(parsed_dict, dict):
        raise LLMSchemaValidationError(
            f"Expected JSON object from LLM provider, got {type(parsed_dict).__name__}"
        )

    try:
        perception = LLMPerceptionOutput.model_validate(parsed_dict)
    except ValidationError as err:
        raise LLMSchemaValidationError(
            f"LLM perception output failed schema validation: {err}"
        ) from err

    # Perform semantic validation layer
    return validate_llm_perception(perception)

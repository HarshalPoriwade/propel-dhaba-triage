"""Safe structured logging and request correlation for Dhaba Triage Service."""

from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
import re
import sys
from typing import Any, Dict, Optional, Set

# Correlation ContextVar for async-safe request tracking without cross-task leakage
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="req_none")

# Safe request ID regex: alphanumeric, hyphen, underscore, 1 to 64 chars
_SAFE_REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")

# Defensive field blocklist: strictly prohibited from appearing in any log record
BLOCKED_LOG_KEYS: Set[str] = frozenset({
    "body",
    "subject",
    "email",
    "phone",
    "customer",
    "customer_name",
    "gstin",
    "company",
    "company_name",
    "card",
    "upi",
    "payment_details",
    "prompt",
    "system_prompt",
    "api_key",
    "token",
    "secret",
    "password",
    "authorization",
})


def get_request_id() -> str:
    """Retrieve the current async request correlation ID."""
    return _request_id_ctx.get()


def set_request_id(req_id: str):
    """Set the current async request correlation ID and return the reset token."""
    return _request_id_ctx.set(req_id)


def sanitize_request_id(val: Optional[str]) -> Optional[str]:
    """Validate and sanitize an incoming request/correlation ID header."""
    if val and _SAFE_REQUEST_ID_REGEX.match(val.strip()):
        return val.strip()
    return None


def sanitize_log_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Defensively filter forbidden sensitive fields from structured log payloads."""
    cleaned: Dict[str, Any] = {}
    for k, v in data.items():
        key_lower = str(k).lower()
        if any(blocked in key_lower for blocked in BLOCKED_LOG_KEYS):
            continue
        if isinstance(v, dict):
            cleaned[k] = sanitize_log_dict(v)
        else:
            cleaned[k] = v
    return cleaned


class StructuredJsonFormatter(logging.Formatter):
    """Formatter emitting single-line machine-readable JSON log events."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None) or get_request_id(),
        }

        # Include structured extra fields passed by caller
        standard_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "message", "request_id",
        }
        extra_fields = {
            k: v for k, v in record.__dict__.items()
            if k not in standard_attrs and not k.startswith("_")
        }

        clean_extra = sanitize_log_dict(extra_fields)
        log_payload.update(clean_extra)

        if record.exc_info and record.exc_info[0] is not None:
            log_payload["error_type"] = record.exc_info[0].__name__

        return json.dumps(log_payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with the structured JSON formatter."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredJsonFormatter())
    root_logger.addHandler(handler)


class StructuredLogger:
    """Convenience wrapper providing explicit, typed structured event logging."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def info(self, event: str, message: str, **kwargs: Any) -> None:
        kwargs["event"] = event
        kwargs["request_id"] = get_request_id()
        self.logger.info(message, extra=sanitize_log_dict(kwargs))

    def warning(self, event: str, message: str, **kwargs: Any) -> None:
        kwargs["event"] = event
        kwargs["request_id"] = get_request_id()
        self.logger.warning(message, extra=sanitize_log_dict(kwargs))

    def error(self, event: str, message: str, **kwargs: Any) -> None:
        kwargs["event"] = event
        kwargs["request_id"] = get_request_id()
        self.logger.error(message, extra=sanitize_log_dict(kwargs))


def get_logger(name: str) -> StructuredLogger:
    """Obtain a StructuredLogger instance for the given namespace."""
    return StructuredLogger(logging.getLogger(name))

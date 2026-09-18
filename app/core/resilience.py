"""Resilience mechanisms including circuit breaker and bounded failure protection."""

from enum import Enum
import threading
import time
from typing import Callable, Optional

from app.core.config import Settings, get_settings


class CircuitState(str, Enum):
    """Operational states for the LLM circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker events."""

    pass


class CircuitBreakerOpenError(CircuitBreakerError):
    """Raised when an operation is rejected because the circuit breaker is open."""

    pass


class CircuitBreaker:
    """Thread-safe and concurrency-safe circuit breaker protecting external LLM calls.

    State Transitions:
    - CLOSED: Normal operation. Successes reset failure count. If consecutive failures
      reach failure_threshold, state trips to OPEN.
    - OPEN: Provider is failing. Requests fail fast without calling provider.
      Once recovery_seconds have passed, state transitions to HALF_OPEN to probe recovery.
    - HALF_OPEN: Allows a single probe request.
      - Probe success: closes circuit (CLOSED) and resets failure count.
      - Probe failure: re-opens circuit (OPEN) and restarts recovery window.
      - Other concurrent requests while probe is in flight fail fast.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_seconds: float = 30.0,
        time_fn: Optional[Callable[[], float]] = None,
    ):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be greater than 0")
        if recovery_seconds <= 0:
            raise ValueError("recovery_seconds must be greater than 0")

        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self._time_fn = time_fn or time.monotonic
        self._lock = threading.Lock()

        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._last_state_change: float = self._time_fn()
        self._probe_in_flight: bool = False

    @property
    def state(self) -> CircuitState:
        """Current operational state of the circuit breaker."""
        with self._lock:
            self._check_cooldown_unlocked()
            return self._state

    @property
    def failure_count(self) -> int:
        """Consecutive recorded failures in current cycle."""
        with self._lock:
            return self._failure_count

    @property
    def last_state_change(self) -> float:
        """Timestamp of last state transition."""
        with self._lock:
            return self._last_state_change

    def _check_cooldown_unlocked(self) -> None:
        """Check if OPEN cooldown has elapsed and transition to HALF_OPEN."""
        if self._state == CircuitState.OPEN:
            elapsed = self._time_fn() - self._last_state_change
            if elapsed >= self.recovery_seconds:
                self._state = CircuitState.HALF_OPEN
                self._probe_in_flight = False

    def allow_request(self) -> bool:
        """Check whether a request is permitted to call the protected dependency.

        Returns:
            True if request may proceed to provider.
            False if request must fail fast without calling provider.
        """
        with self._lock:
            self._check_cooldown_unlocked()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.HALF_OPEN:
                if not self._probe_in_flight:
                    self._probe_in_flight = True
                    return True
                return False

            # CircuitState.OPEN
            return False

    def record_success(self) -> None:
        """Record a successful provider call, closing circuit and resetting failure count."""
        with self._lock:
            self._failure_count = 0
            self._probe_in_flight = False
            if self._state != CircuitState.CLOSED:
                self._state = CircuitState.CLOSED
                self._last_state_change = self._time_fn()

    def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failed provider call, potentially tripping circuit to OPEN."""
        with self._lock:
            now = self._time_fn()
            self._failure_count += 1
            self._probe_in_flight = False

            if self._state == CircuitState.HALF_OPEN:
                # Probe failed; reopen immediately and restart recovery window
                self._state = CircuitState.OPEN
                self._last_state_change = now
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._last_state_change = now

    def reset(self) -> None:
        """Reset circuit breaker to its initial CLOSED state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_state_change = self._time_fn()
            self._probe_in_flight = False


_shared_circuit_breaker: Optional[CircuitBreaker] = None
_circuit_breaker_lock = threading.Lock()


def get_circuit_breaker(settings: Optional[Settings] = None) -> CircuitBreaker:
    """Retrieve or create the process-level shared CircuitBreaker instance."""
    global _shared_circuit_breaker
    if _shared_circuit_breaker is None:
        with _circuit_breaker_lock:
            if _shared_circuit_breaker is None:
                if settings is None:
                    settings = get_settings()
                _shared_circuit_breaker = CircuitBreaker(
                    failure_threshold=settings.LLM_FAILURE_THRESHOLD,
                    recovery_seconds=settings.LLM_RECOVERY_SECONDS,
                )
    return _shared_circuit_breaker


def reset_shared_circuit_breaker() -> None:
    """Clear the shared circuit breaker singleton (for test isolation)."""
    global _shared_circuit_breaker
    with _circuit_breaker_lock:
        _shared_circuit_breaker = None

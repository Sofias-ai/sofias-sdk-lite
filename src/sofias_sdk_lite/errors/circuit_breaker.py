"""Circuit breaker implementation for node execution protection.

Used by ErrorHandler to prevent repeatedly calling failing nodes.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, ConfigDict


class CircuitState(Enum):
    """State of a circuit breaker."""

    CLOSED = "closed"
    """Normal operation. Requests go through to the node."""

    OPEN = "open"
    """Circuit is tripped. Requests go directly to fallback."""

    HALF_OPEN = "half_open"
    """Testing recovery. One request allowed through to test if node recovered."""


class CircuitBreakerState(BaseModel):
    """State information for a circuit breaker.

    This is the data structure stored by CircuitBreakerStore implementations.
    """

    model_config = ConfigDict(frozen=False)

    state: CircuitState = CircuitState.CLOSED
    """Current state of the circuit breaker."""

    failure_count: int = 0
    """Consecutive failures in CLOSED state."""

    last_failure_time: float | None = None
    """Timestamp of the last failure (for recovery timeout)."""

    last_success_time: float | None = None
    """Timestamp of the last success."""


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker behavior."""

    model_config = ConfigDict(frozen=True)

    failure_threshold: int = 5
    """Number of consecutive failures to trip the circuit."""

    recovery_timeout: float = 30.0
    """Seconds to wait in OPEN state before trying HALF_OPEN."""


class CircuitBreakerStore(Protocol):
    """Protocol for circuit breaker state storage.

    This interface allows the ErrorHandler to work with different storage
    backends. The default InMemoryCircuitBreakerStore works for single-process
    deployments.

    IMPORTANT: When running multiple instances of the same agent consuming
    from a shared queue in parallel, each instance will have its own in-memory
    counters and will NOT see failures from other instances. In that scenario,
    implement a CircuitBreakerStore backed by Redis or another shared store,
    and inject it into the ErrorHandler. The ErrorHandler and the rest of the
    SDK will not need to change - only the store implementation.
    """

    async def get_state(self, node_name: str) -> CircuitBreakerState:
        """Get the current state of a circuit breaker."""
        ...

    async def record_failure(self, node_name: str) -> None:
        """Record a failure for a node's circuit breaker."""
        ...

    async def record_success(self, node_name: str) -> None:
        """Record a success for a node's circuit breaker."""
        ...

    async def reset(self, node_name: str) -> None:
        """Reset a circuit breaker to initial state."""
        ...


class InMemoryCircuitBreakerStore:
    """In-memory implementation of CircuitBreakerStore.

    Works correctly for single-process deployments.

    IMPORTANT: When running multiple instances of the same agent consuming
    from a shared queue in parallel, each instance will have its own
    in-memory counters. This means:
    - Instance A won't see failures from Instance B
    - The circuit won't trip based on total failures across instances
    - Recovery detection is per-instance, not global

    For distributed deployments, implement a CircuitBreakerStore backed by
    Redis or another shared store, using atomic operations (e.g. Redis INCR)
    to ensure correct counting across instances.
    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self._config = config or CircuitBreakerConfig()
        self._states: dict[str, CircuitBreakerState] = {}

    def _get_or_create_state(self, node_name: str) -> CircuitBreakerState:
        if node_name not in self._states:
            self._states[node_name] = CircuitBreakerState()
        return self._states[node_name]

    async def get_state(self, node_name: str) -> CircuitBreakerState:
        state = self._get_or_create_state(node_name)

        if state.state == CircuitState.OPEN and state.last_failure_time is not None:
            time_since_failure = time.time() - state.last_failure_time
            if time_since_failure >= self._config.recovery_timeout:
                state.state = CircuitState.HALF_OPEN

        return state

    async def record_failure(self, node_name: str) -> None:
        state = self._get_or_create_state(node_name)
        state.failure_count += 1
        state.last_failure_time = time.time()

        if state.failure_count >= self._config.failure_threshold:
            state.state = CircuitState.OPEN

        if state.state == CircuitState.HALF_OPEN:
            state.state = CircuitState.OPEN
            state.failure_count = self._config.failure_threshold

    async def record_success(self, node_name: str) -> None:
        state = self._get_or_create_state(node_name)
        state.last_success_time = time.time()
        state.failure_count = 0

        if state.state == CircuitState.HALF_OPEN:
            state.state = CircuitState.CLOSED

    async def reset(self, node_name: str) -> None:
        self._states[node_name] = CircuitBreakerState()

"""Tests for ErrorHandler: retry, circuit breaker, fallback."""

from __future__ import annotations

import pytest

from sofias_sdk_lite.errors import (
    CircuitBreakerConfig,
    ErrorHandler,
    ErrorHandlerConfig,
    GraphExecutionError,
    NodeExecutionError,
    RetryPolicy,
)
from sofias_sdk_lite.errors.circuit_breaker import CircuitState


@pytest.mark.asyncio
async def test_handler_retries_and_succeeds_before_max_retries():
    attempts = {"count": 0}

    async def flaky(data: dict) -> dict:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise NodeExecutionError("transient failure", node_name="flaky")
        return {"ok": True}

    handler = ErrorHandler(
        config=ErrorHandlerConfig(retry=RetryPolicy(max_retries=5, delay=0.0))
    )
    result = await handler.handle_node_execution("flaky", flaky, {})

    assert result == {"ok": True}
    assert attempts["count"] == 3


@pytest.mark.asyncio
async def test_handler_raises_after_exhausting_retries_with_no_fallback():
    async def always_fails(data: dict) -> dict:
        raise NodeExecutionError("permanent failure", node_name="broken")

    handler = ErrorHandler(
        config=ErrorHandlerConfig(retry=RetryPolicy(max_retries=1, delay=0.0))
    )

    with pytest.raises(GraphExecutionError):
        await handler.handle_node_execution("broken", always_fails, {})


@pytest.mark.asyncio
async def test_circuit_breaker_trips_after_failure_threshold_and_routes_to_fallback():
    async def always_fails(data: dict) -> dict:
        raise NodeExecutionError("down", node_name="unstable")

    async def fallback(data: dict, error_context) -> dict:
        return {"fallback": True}

    handler = ErrorHandler(
        config=ErrorHandlerConfig(
            retry=RetryPolicy(max_retries=0, delay=0.0),
            circuit_breaker=CircuitBreakerConfig(failure_threshold=2, recovery_timeout=999.0),
            fallback={"fallbacks": {"unstable": "backup"}},
        )
    )
    handler.register_fallback_executor("backup", fallback)

    # First two calls trigger a failure each; both have a fallback registered.
    result1 = await handler.handle_node_execution("unstable", always_fails, {})
    result2 = await handler.handle_node_execution("unstable", always_fails, {})
    assert result1 == {"fallback": True}
    assert result2 == {"fallback": True}

    state = await handler.store.get_state("unstable")
    assert state.state == CircuitState.OPEN

    # Third call: circuit is open, should route straight to fallback.
    result3 = await handler.handle_node_execution("unstable", always_fails, {})
    assert result3 == {"fallback": True}

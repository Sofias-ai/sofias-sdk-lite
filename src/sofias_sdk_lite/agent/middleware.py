"""Agent-level middleware for cross-cutting concerns.

Middleware hooks are called before and after each node execution,
allowing logging, metrics, tracing, and input/output transformation
without modifying node code.

Example::

    class MetricsMiddleware:
        async def before_node(self, node_name, input_data):
            self.start = time.perf_counter()
            return None

        async def after_node(self, node_name, input_data, output_data):
            elapsed = (time.perf_counter() - self.start) * 1000
            metrics.record(node_name, elapsed)
            return None

        async def on_error(self, node_name, error):
            metrics.record_error(node_name)

    agent = (
        AgentBuilder(...)
        .with_middleware(MetricsMiddleware())
        .build()
    )
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

from sofias_sdk_lite.observability._log import get_logger

logger = get_logger(__name__)

__all__ = [
    "AgentMiddleware",
    "LoggingMiddleware",
    "TimingMiddleware",
]


@runtime_checkable
class AgentMiddleware(Protocol):
    """Protocol for agent-level middleware.

    All methods are optional — implement only the hooks you need.
    Return None from before_node/after_node to keep original data,
    or return a dict to replace it.
    """

    async def before_node(
        self, node_name: str, input_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Called before each node execution."""
        ...

    async def after_node(
        self, node_name: str, input_data: dict[str, Any], output_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Called after each node execution."""
        ...

    async def on_error(
        self, node_name: str, error: Exception,
    ) -> None:
        """Called when a node fails (before error handler retries)."""
        ...


class LoggingMiddleware:
    """Built-in middleware that logs node execution with timing."""

    async def before_node(
        self, node_name: str, input_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        logger.info("node_start", node_name=node_name)
        return None

    async def after_node(
        self, node_name: str, input_data: dict[str, Any], output_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        logger.info("node_complete", node_name=node_name)
        return None

    async def on_error(
        self, node_name: str, error: Exception,
    ) -> None:
        logger.error("node_error", node_name=node_name, error=str(error))


class TimingMiddleware:
    """Built-in middleware that records per-node execution time."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self._starts: dict[str, float] = {}

    async def before_node(
        self, node_name: str, input_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        self._starts[node_name] = time.perf_counter()
        return None

    async def after_node(
        self, node_name: str, input_data: dict[str, Any], output_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        start = self._starts.pop(node_name, None)
        if start is not None:
            self.timings[node_name] = (time.perf_counter() - start) * 1000
        return None

    async def on_error(
        self, node_name: str, error: Exception,
    ) -> None:
        self._starts.pop(node_name, None)

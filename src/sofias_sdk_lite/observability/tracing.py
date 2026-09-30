"""Tracing protocols for agent execution observability.

Defines the ``SpanType`` taxonomy plus the ``Span``/``TracingProvider``
protocols the SDK expects from an external tracing system, the
``TracingAdapter`` convenience wrapper (context managers for tracing agent,
node, tool, and LLM operations), and no-op / logging-only implementations
for when a real tracing backend isn't configured.

Protocol-only: no concrete OpenTelemetry/OTLP exporter is vendored.
Bring your own backend by implementing ``TracingProvider``.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager, contextmanager
from enum import StrEnum
from typing import Any, AsyncIterator, Iterator, Protocol, runtime_checkable

from sofias_sdk_lite.observability._log import get_logger


class SpanType(StrEnum):
    """Types of spans, aligned with sofias_sdk_lite node types."""

    AGENT = "agent"
    """Top-level agent execution (root span)."""

    LLM_NODE = "llm_node"
    """Execution of an LLMNode (classifier, responder, etc.)."""

    PLANNER_NODE = "planner_node"
    """Execution of an LLMNode configured as a planner."""

    FUNCTION_NODE = "function_node"
    """Execution of a FunctionNode (deterministic logic)."""

    DELEGATION_NODE = "delegation_node"
    """Execution of a DelegationNode (RPC orchestration)."""

    LLM_CALL = "llm_call"
    """Individual LLM invocation within a node's tool loop."""

    TOOL_CALL = "tool_call"
    """Individual tool execution within a node's tool loop."""

    DELEGATION_CALL = "delegation_call"
    """Individual RPC call to an external agent."""


@runtime_checkable
class Span(Protocol):
    """Protocol defining the interface for a tracing span.

    A span represents a unit of work or operation. It tracks timing,
    attributes, and events within that operation.

    Example implementation:
        class OTelSpan:
            def __init__(self, otel_span):
                self._span = otel_span

            def add_event(self, name: str, attributes: dict | None = None):
                self._span.add_event(name, attributes=attributes)

            def set_attribute(self, key: str, value: Any):
                self._span.set_attribute(key, value)
    """

    def add_event(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Add an event to the span.

        Events are timestamped annotations within a span, useful for
        logging significant occurrences during the operation.

        Args:
            name: Name of the event (e.g., "cache_hit", "retry_attempt").
            attributes: Optional attributes for the event.
        """
        ...

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the span.

        Attributes are key-value pairs that describe the span
        (e.g., "http.status_code", "db.query").

        Args:
            key: The attribute key.
            value: The attribute value (should be primitive type).
        """
        ...


@runtime_checkable
class TracingProvider(Protocol):
    """Protocol defining the interface for tracing providers.

    This protocol defines what the SDK expects from an external
    tracing system. The concrete implementation connects to
    OpenTelemetry, Jaeger, Datadog, or another tracing backend.

    Example implementation:
        from opentelemetry import trace

        class OTelTracingProvider:
            def __init__(self, service_name: str):
                self._tracer = trace.get_tracer(service_name)

            def start_span(
                self, name: str, attributes: dict | None = None
            ) -> Span:
                span = self._tracer.start_span(name, attributes=attributes)
                return OTelSpan(span)

            def end_span(
                self,
                span: Span,
                status: str = "ok",
                error: Exception | None = None,
            ) -> None:
                if error:
                    span._span.record_exception(error)
                    span._span.set_status(StatusCode.ERROR)
                else:
                    span._span.set_status(StatusCode.OK)
                span._span.end()
    """

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a new span.

        Args:
            name: Name of the span (e.g., "agent.execute", "node.classifier").
            attributes: Optional initial attributes for the span.

        Returns:
            A Span instance representing the operation.
        """
        ...

    def end_span(
        self,
        span: Span,
        status: str = "ok",
        error: Exception | None = None,
    ) -> None:
        """End a span with status.

        Args:
            span: The span to end.
            status: Status of the operation ("ok" or "error").
            error: Optional exception if the operation failed.
        """
        ...


class NullSpan:
    """A no-op span implementation for when tracing is disabled."""

    def add_event(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """No-op event."""
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        """No-op attribute."""
        pass


class NullTracingProvider:
    """A no-op tracing provider for when observability is disabled.

    This provider creates NullSpan instances that do nothing.
    It allows the SDK to function without any tracing overhead
    when observability is not configured.
    """

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Return a null span."""
        return NullSpan()

    def end_span(
        self,
        span: Span,
        status: str = "ok",
        error: Exception | None = None,
    ) -> None:
        """No-op end."""
        pass


class TracingAdapter:
    """Adapter that provides convenient tracing methods for SDK operations.

    The TracingAdapter wraps a TracingProvider with convenience methods
    specific to agent SDK operations. It provides:
    - Context managers for common operations (agent, node, tool, LLM)
    - Automatic timing and error recording
    - No-op behavior when provider is None

    Example:
        provider = OTelTracingProvider("my-agent-service")
        tracing = TracingAdapter(provider)

        # Trace agent execution
        async with tracing.trace_agent_execution_async("inbox_assist") as span:
            span.set_attribute("input.length", len(str(message.content)))
            response = await agent.execute(message)

        # Trace node execution
        with tracing.trace_node_execution("classifier", input_data) as span:
            result = await node.execute(input_data)
    """

    def __init__(self, provider: TracingProvider | None = None) -> None:
        """Initialize the tracing adapter.

        Args:
            provider: Optional tracing provider. If None, all tracing
                methods become no-ops with zero overhead.
        """
        self._provider = provider or NullTracingProvider()
        self._is_null = isinstance(self._provider, NullTracingProvider)

    @property
    def provider(self) -> TracingProvider:
        """The underlying tracing provider."""
        return self._provider

    @property
    def is_enabled(self) -> bool:
        """Whether tracing is enabled (non-null provider)."""
        return not self._is_null

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a new span.

        Args:
            name: Name of the span.
            attributes: Optional initial attributes.

        Returns:
            A Span instance.
        """
        return self._provider.start_span(name, attributes)

    def end_span(
        self,
        span: Span,
        status: str = "ok",
        error: Exception | None = None,
    ) -> None:
        """End a span with status.

        Args:
            span: The span to end.
            status: Status of the operation.
            error: Optional exception if failed.
        """
        self._provider.end_span(span, status, error)

    @contextmanager
    def trace_agent_execution(
        self,
        agent_name: str,
        agent_version: str | None = None,
    ) -> Iterator[Span]:
        """Context manager for tracing agent execution.

        Creates a span for the entire agent execution, automatically
        recording timing and errors.

        Args:
            agent_name: Name of the agent being executed.
            agent_version: Optional version of the agent.

        Yields:
            The span for adding events and attributes.

        Example:
            with tracing.trace_agent_execution("my_agent", "1.0.0") as span:
                span.set_attribute("input.type", "email")
                response = agent.execute(message)
        """
        attributes = {
            "agent.name": agent_name,
            "agent.operation": "execute",
        }
        if agent_version:
            attributes["agent.version"] = agent_version

        span = self.start_span(f"agent.{agent_name}.execute", attributes)

        try:
            yield span
            self.end_span(span, status="ok")
        except Exception as e:
            self.end_span(span, status="error", error=e)
            raise

    @asynccontextmanager
    async def trace_agent_execution_async(
        self,
        agent_name: str,
        agent_version: str | None = None,
    ) -> AsyncIterator[Span]:
        """Async context manager for tracing agent execution.

        Same as trace_agent_execution but for async contexts.

        Args:
            agent_name: Name of the agent being executed.
            agent_version: Optional version of the agent.

        Yields:
            The span for adding events and attributes.
        """
        attributes = {
            "agent.name": agent_name,
            "agent.operation": "execute",
        }
        if agent_version:
            attributes["agent.version"] = agent_version

        span = self.start_span(f"agent.{agent_name}.execute", attributes)

        try:
            yield span
            self.end_span(span, status="ok")
        except Exception as e:
            self.end_span(span, status="error", error=e)
            raise

    @contextmanager
    def trace_node_execution(
        self,
        node_name: str,
        input_data: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        """Context manager for tracing node execution.

        Args:
            node_name: Name of the node being executed.
            input_data: Optional input data (for logging key count/size).

        Yields:
            The span for adding events and attributes.
        """
        attributes: dict[str, Any] = {
            "node.name": node_name,
            "node.operation": "execute",
        }
        if input_data:
            attributes["node.input.keys"] = list(input_data.keys())

        span = self.start_span(f"node.{node_name}.execute", attributes)

        try:
            yield span
            self.end_span(span, status="ok")
        except Exception as e:
            self.end_span(span, status="error", error=e)
            raise

    @asynccontextmanager
    async def trace_node_execution_async(
        self,
        node_name: str,
        input_data: dict[str, Any] | None = None,
    ) -> AsyncIterator[Span]:
        """Async context manager for tracing node execution."""
        attributes: dict[str, Any] = {
            "node.name": node_name,
            "node.operation": "execute",
        }
        if input_data:
            attributes["node.input.keys"] = list(input_data.keys())

        span = self.start_span(f"node.{node_name}.execute", attributes)

        try:
            yield span
            self.end_span(span, status="ok")
        except Exception as e:
            self.end_span(span, status="error", error=e)
            raise

    @contextmanager
    def trace_tool_execution(
        self,
        tool_name: str,
        tool_input: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        """Context manager for tracing tool execution.

        Args:
            tool_name: Name of the tool being executed.
            tool_input: Optional tool input (for logging).

        Yields:
            The span for adding events and attributes.
        """
        attributes: dict[str, Any] = {
            "tool.name": tool_name,
            "tool.operation": "execute",
        }
        if tool_input:
            attributes["tool.input.keys"] = list(tool_input.keys())

        span = self.start_span(f"tool.{tool_name}.execute", attributes)

        try:
            yield span
            self.end_span(span, status="ok")
        except Exception as e:
            self.end_span(span, status="error", error=e)
            raise

    @asynccontextmanager
    async def trace_tool_execution_async(
        self,
        tool_name: str,
        tool_input: dict[str, Any] | None = None,
    ) -> AsyncIterator[Span]:
        """Async context manager for tracing tool execution."""
        attributes: dict[str, Any] = {
            "tool.name": tool_name,
            "tool.operation": "execute",
        }
        if tool_input:
            attributes["tool.input.keys"] = list(tool_input.keys())

        span = self.start_span(f"tool.{tool_name}.execute", attributes)

        try:
            yield span
            self.end_span(span, status="ok")
        except Exception as e:
            self.end_span(span, status="error", error=e)
            raise

    @contextmanager
    def trace_llm_call(
        self,
        model: str,
        prompt_length: int,
        *,
        provider: str | None = None,
    ) -> Iterator[Span]:
        """Context manager for tracing LLM calls.

        Args:
            model: The model being called (e.g., "gpt-4o").
            prompt_length: Length of the prompt in characters.
            provider: Optional provider name (e.g., "openai").

        Yields:
            The span for adding events and attributes.
        """
        attributes: dict[str, Any] = {
            "llm.model": model,
            "llm.prompt.length": prompt_length,
            "llm.operation": "invoke",
        }
        if provider:
            attributes["llm.provider"] = provider

        span = self.start_span(f"llm.{model}.invoke", attributes)
        start_time = time.perf_counter()

        try:
            yield span
            duration = time.perf_counter() - start_time
            span.set_attribute("llm.duration_ms", duration * 1000)
            self.end_span(span, status="ok")
        except Exception as e:
            duration = time.perf_counter() - start_time
            span.set_attribute("llm.duration_ms", duration * 1000)
            self.end_span(span, status="error", error=e)
            raise

    @asynccontextmanager
    async def trace_llm_call_async(
        self,
        model: str,
        prompt_length: int,
        *,
        provider: str | None = None,
    ) -> AsyncIterator[Span]:
        """Async context manager for tracing LLM calls."""
        attributes: dict[str, Any] = {
            "llm.model": model,
            "llm.prompt.length": prompt_length,
            "llm.operation": "invoke",
        }
        if provider:
            attributes["llm.provider"] = provider

        span = self.start_span(f"llm.{model}.invoke", attributes)
        start_time = time.perf_counter()

        try:
            yield span
            duration = time.perf_counter() - start_time
            span.set_attribute("llm.duration_ms", duration * 1000)
            self.end_span(span, status="ok")
        except Exception as e:
            duration = time.perf_counter() - start_time
            span.set_attribute("llm.duration_ms", duration * 1000)
            self.end_span(span, status="error", error=e)
            raise


class LoggingTracingProvider:
    """A tracing provider that logs spans for debugging.

    Useful for local development when a full tracing system
    is not available. Logs span start/end with timing.
    """

    def __init__(self, logger_name: str = "tracing") -> None:
        """Initialize with a logger name."""
        self._logger = get_logger(logger_name)
        self._spans: dict[int, dict[str, Any]] = {}

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a span and log it."""
        span = LoggingSpan(name, attributes, self._logger)
        self._logger.debug(f"[SPAN START] {name} attrs={attributes}")
        return span

    def end_span(
        self,
        span: Span,
        status: str = "ok",
        error: Exception | None = None,
    ) -> None:
        """End a span and log it."""
        if isinstance(span, LoggingSpan):
            duration = span.get_duration_ms()
            if error:
                self._logger.debug(
                    f"[SPAN END] {span.name} status={status} "
                    f"duration={duration:.2f}ms error={error}"
                )
            else:
                self._logger.debug(
                    f"[SPAN END] {span.name} status={status} "
                    f"duration={duration:.2f}ms"
                )


class LoggingSpan:
    """A span implementation that logs events and attributes."""

    def __init__(
        self,
        name: str,
        attributes: dict[str, Any] | None,
        log: Any,
    ) -> None:
        """Initialize the logging span."""
        self.name = name
        self._attributes = dict(attributes or {})
        self._logger = log
        self._start_time = time.perf_counter()

    def add_event(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Log an event."""
        self._logger.debug(f"[SPAN EVENT] {self.name}: {name} attrs={attributes}")

    def set_attribute(self, key: str, value: Any) -> None:
        """Store and log an attribute."""
        self._attributes[key] = value
        self._logger.debug(f"[SPAN ATTR] {self.name}: {key}={value}")

    def get_duration_ms(self) -> float:
        """Get the span duration in milliseconds."""
        return (time.perf_counter() - self._start_time) * 1000


__all__ = [
    "SpanType",
    "Span",
    "TracingProvider",
    "NullSpan",
    "NullTracingProvider",
    "TracingAdapter",
    "LoggingTracingProvider",
    "LoggingSpan",
]

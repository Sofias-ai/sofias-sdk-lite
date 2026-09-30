"""Observability protocols for sofias_sdk_lite: tracing and events.

Protocol-only — no concrete OTLP exporter or message broker client
ships here. Bring your own tracing backend (OpenTelemetry, Jaeger,
Datadog, ...) and event broker (RabbitMQ, Kafka, ...) by implementing the
protocols below.
"""

from __future__ import annotations

from sofias_sdk_lite.observability._log import get_logger
from sofias_sdk_lite.observability.events import (
    EventAdapter,
    EventHandler,
    EventListener,
    EventPublisher,
    NullEventListener,
    NullEventPublisher,
)
from sofias_sdk_lite.observability.trace_context import (
    SpanHandle,
    active_traceparent,
    consumer_span,
    has_otel,
)
from sofias_sdk_lite.observability.tracing import (
    LoggingSpan,
    LoggingTracingProvider,
    NullSpan,
    NullTracingProvider,
    Span,
    SpanType,
    TracingAdapter,
    TracingProvider,
)

__all__ = [
    "get_logger",
    "SpanHandle",
    "active_traceparent",
    "consumer_span",
    "has_otel",
    "EventAdapter",
    "EventHandler",
    "EventListener",
    "EventPublisher",
    "NullEventListener",
    "NullEventPublisher",
    "LoggingSpan",
    "LoggingTracingProvider",
    "NullSpan",
    "NullTracingProvider",
    "Span",
    "SpanType",
    "TracingAdapter",
    "TracingProvider",
]

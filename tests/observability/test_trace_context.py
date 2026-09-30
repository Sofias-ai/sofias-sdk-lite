"""Trace-context helpers degrade to pure echo / no-op without OpenTelemetry."""

from __future__ import annotations

import pytest

from sofias_sdk_lite.observability import trace_context
from sofias_sdk_lite.observability.trace_context import active_traceparent, consumer_span, has_otel


def test_active_traceparent_echoes_fallback_without_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trace_context, "_otel", lambda: None)
    assert active_traceparent("inbound") == "inbound"
    assert active_traceparent(None) is None


def test_consumer_span_is_a_noop_without_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trace_context, "_otel", lambda: None)
    with consumer_span("agent.handle", traceparent="x", attributes={"a": 1}) as span:
        span.set_attribute("k", "v")
        span.record_error(RuntimeError("boom"))


def test_has_otel_reports_import_state() -> None:
    assert isinstance(has_otel(), bool)


@pytest.mark.skipif(not has_otel(), reason="opentelemetry-api not installed")
def test_active_span_wins_over_inbound_with_otel() -> None:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    trace.set_tracer_provider(TracerProvider())
    with consumer_span("agent.handle", traceparent=None) as span:
        span.set_attribute("agent.name", "a")
        span.record_error(RuntimeError("boom"))
        tp = active_traceparent("inbound")
    assert tp is not None and tp != "inbound" and tp.startswith("00-")
    assert active_traceparent("inbound") == "inbound"  # no active span afterwards


@pytest.mark.skipif(not has_otel(), reason="opentelemetry-api not installed")
def test_child_span_continues_the_inbound_trace() -> None:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    trace.set_tracer_provider(TracerProvider())
    inbound = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    with consumer_span("agent.handle", traceparent=inbound):
        tp = active_traceparent(None)
    assert tp is not None
    assert tp.split("-")[1] == "0af7651916cd43dd8448eb211c80319c"  # same trace id
    assert tp.split("-")[2] != "b7ad6b7169203331"  # new span id

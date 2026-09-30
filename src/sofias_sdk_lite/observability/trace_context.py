"""W3C trace-context helpers with OpenTelemetry as an *optional* dependency.

The SDK never requires ``opentelemetry-api``. When it is installed (the
``[otel]`` extra), the runner opens a root ``agent.handle`` span per turn and
the response workflows echo the *active* span's ``traceparent`` on every
published fragment, so a chat turn stays one trace end to end. Without it,
everything here degrades to a pure echo of the inbound ``traceparent`` (or to
no-ops), with no behavioral change on the wire.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "SpanHandle",
    "active_traceparent",
    "consumer_span",
    "has_otel",
]

_TRACER_NAME = "sofias_sdk_lite"


@dataclass(frozen=True)
class _OTel:
    """The two OpenTelemetry API modules this SDK touches, loaded lazily."""

    propagate: Any
    trace: Any


_otel_cache: _OTel | None = None
_otel_probed = False


def _otel() -> _OTel | None:
    """Return the OpenTelemetry API modules, or ``None`` when not installed.

    Loaded through ``importlib`` so the SDK type-checks and runs identically
    with or without the optional dependency.
    """
    global _otel_cache, _otel_probed
    if not _otel_probed:
        _otel_probed = True
        if importlib.util.find_spec("opentelemetry") is not None:
            _otel_cache = _OTel(
                propagate=importlib.import_module("opentelemetry.propagate"),
                trace=importlib.import_module("opentelemetry.trace"),
            )
    return _otel_cache


def has_otel() -> bool:
    """Whether ``opentelemetry-api`` is importable in this environment."""
    return _otel() is not None


def active_traceparent(fallback: str | None = None) -> str | None:
    """Return the ``traceparent`` of the active OTel span, else ``fallback``.

    Response workflows call this right before publishing so the response
    continues the agent's own span rather than the inbound request's. With no
    active span (or no OTel at all) this is a pure echo of ``fallback``.
    """
    otel = _otel()
    if otel is None:
        return fallback
    carrier: dict[str, str] = {}
    otel.propagate.inject(carrier)
    return carrier.get("traceparent") or fallback


class SpanHandle(Protocol):
    """Minimal span surface the runner needs; satisfied by OTel and by the no-op."""

    def set_attribute(self, key: str, value: Any) -> None: ...

    def record_error(self, exc: BaseException) -> None: ...


class _NullSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def record_error(self, exc: BaseException) -> None:
        return None


class _OTelSpan:
    def __init__(self, span: Any, trace: Any) -> None:
        self._span = span
        self._trace = trace

    def set_attribute(self, key: str, value: Any) -> None:
        self._span.set_attribute(key, value)

    def record_error(self, exc: BaseException) -> None:
        # Type + message on the span, not just an ERROR status: otherwise the
        # tracing UI shows "Error" with no detail and one has to jump to logs.
        self._span.record_exception(exc)
        self._span.set_status(self._trace.Status(self._trace.StatusCode.ERROR, str(exc)))


@contextmanager
def consumer_span(
    name: str,
    *,
    traceparent: str | None,
    attributes: dict[str, Any] | None = None,
) -> Generator[SpanHandle, None, None]:
    """Open a CONSUMER span as the *active* context, child of ``traceparent``.

    No-op (yields a null handle) when OpenTelemetry is not installed or no
    ``TracerProvider`` is configured, so it is safe to call unconditionally.
    """
    otel = _otel()
    if otel is None:
        yield _NullSpan()
        return

    tracer = otel.trace.get_tracer(_TRACER_NAME)
    parent = otel.propagate.extract({"traceparent": traceparent}) if traceparent else None
    with tracer.start_as_current_span(
        name, context=parent, kind=otel.trace.SpanKind.CONSUMER
    ) as span:
        for key, value in (attributes or {}).items():
            span.set_attribute(key, value)
        yield _OTelSpan(span, otel.trace)

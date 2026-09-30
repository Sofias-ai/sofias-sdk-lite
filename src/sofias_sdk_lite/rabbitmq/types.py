"""Shared types for the RabbitMQ transport layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import aio_pika

__all__ = [
    "AckPolicy",
    "MessageContext",
    "PublishOptions",
    "with_traceparent",
]


class AckPolicy(Enum):
    """Controls how message acknowledgement is handled.

    AUTO   - The library acks after a successful callback and nacks on exception.
    MANUAL - The callback is responsible for calling ctx.ack() / ctx.nack() / ctx.reject().
    """

    AUTO = "auto"
    MANUAL = "manual"


@dataclass
class MessageContext:
    """Rich context passed to consumer handlers.

    Provides the deserialized payload, AMQP metadata, and methods
    for explicit acknowledgement when using ``AckPolicy.MANUAL``.

    ``traceparent`` is the W3C trace-context header (if the producer set it),
    surfaced directly and typed for trace continuity; it stays available in
    ``headers`` too.
    """

    payload: dict
    headers: dict
    traceparent: str | None
    routing_key: str
    exchange: str
    delivery_tag: int
    correlation_id: str | None
    reply_to: str | None
    timestamp: datetime | None
    _message: aio_pika.IncomingMessage = field(repr=False, compare=False)

    async def ack(self) -> None:
        """Acknowledge the message."""
        await self._message.ack()

    async def nack(self, requeue: bool = True) -> None:
        """Negatively acknowledge the message."""
        await self._message.nack(requeue=requeue)

    async def reject(self) -> None:
        """Reject the message (no requeue)."""
        await self._message.reject(requeue=False)


@dataclass
class PublishOptions:
    """Optional parameters for publish() and publish_stream()."""

    correlation_id: str | None = None
    headers: dict[str, Any] | None = None
    ttl: int | None = None  # milliseconds
    priority: int | None = None
    persistent: bool = True
    traceparent: str | None = None
    """W3C trace context. On ``publish()`` it travels as the ``traceparent``
    AMQP header; on ``publish_stream()`` as an AMQP 1.0 application property."""


def with_traceparent(
    headers: dict[str, Any] | None,
    traceparent: str | None,
) -> dict[str, Any] | None:
    """Return ``headers`` with the W3C ``traceparent`` added, only if present.

    The same header name is used on the queue side (AMQP 0.9.1 headers) and on
    the stream side (AMQP 1.0 ``application_properties``), so a consumer reads
    it back the same way regardless of transport. With no ``traceparent`` the
    headers are returned untouched, keeping the historical wire format.
    """
    if traceparent is None:
        return headers
    return {**(headers or {}), "traceparent": traceparent}

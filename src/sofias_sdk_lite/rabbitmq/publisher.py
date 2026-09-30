"""Publisher for RabbitMQ with support for exchanges and streams."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional, Union

import aio_pika
from aio_pika import Exchange
from rstream import AMQPMessage, Producer

from sofias_sdk_lite.observability._log import get_logger

from .client import RabbitMQClient
from .types import PublishOptions, with_traceparent

__all__ = [
    "RabbitMQPublisher",
    "publish_reply",
    "Exchange",
    "Producer",
    "PublishOptions",
]

logger = get_logger("rabbitmq.publisher")

# Retry settings for RabbitMQ Streams (port 5552) connections and sends.
_STREAM_MAX_RETRIES = 3
_STREAM_BASE_DELAY = 1.0  # seconds; doubles each attempt (1, 2, 4)


async def publish_reply(
    client: RabbitMQClient,
    reply_to: str,
    correlation_id: str,
    payload: Union[Dict[str, Any], bytes, str],
    *,
    content_type: str = "application/json",
) -> None:
    """Publish a reply to the default exchange with a dynamic routing key.

    Common pattern for responding to a message that includes ``reply_to``
    and ``correlation_id`` (e.g. delegation responses, RPC replies).

    Args:
        client: Connected RabbitMQ client.
        reply_to: Destination queue (used as the routing key on the default exchange).
        correlation_id: Correlation ID so the receiver can associate the response.
        payload: Message body. Accepts a dict (serialized to JSON),
                 a str (encoded as UTF-8), or bytes (sent as-is).
        content_type: Message content-type (default ``application/json``).
    """
    if isinstance(payload, dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    elif isinstance(payload, str):
        body = payload.encode("utf-8")
    else:
        body = payload

    message = aio_pika.Message(
        body=body,
        content_type=content_type,
        correlation_id=correlation_id,
        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
    )

    async with client.acquire_channel() as channel:
        await channel.default_exchange.publish(
            message,
            routing_key=reply_to,
        )

    logger.info(
        "rabbitmq_publish_reply",
        reply_to=reply_to,
        correlation_id=correlation_id,
    )


class RabbitMQPublisher:
    """Publishes messages to RabbitMQ exchanges and streams."""

    def __init__(
        self,
        *,
        client: RabbitMQClient,
        exchange_name: str,
        routing_key: str,
        content_type: str = "application/json",
    ) -> None:
        self._client = client
        self._exchange_name = exchange_name
        self._routing_key = routing_key
        self._content_type = content_type
        self._exchange: Optional[aio_pika.Exchange] = None
        self._stream_producer: Optional[Producer] = None
        self._stream_producer_lock = asyncio.Lock()
        self._known_streams: set[str] = set()

    async def _ensure_exchange(self) -> aio_pika.Exchange:
        """Ensure the exchange exists and return it."""
        if self._exchange:
            return self._exchange

        async with self._client.acquire_channel() as channel:
            self._exchange = await channel.declare_exchange(
                self._exchange_name,
                aio_pika.ExchangeType.TOPIC,
                durable=True,
            )
        return self._exchange

    async def publish(
        self,
        payload: Dict[str, Any],
        options: Optional[PublishOptions] = None,
    ) -> None:
        """Publish a message to the exchange.

        Args:
            payload: Dictionary with the message to publish.
            options: Publishing options (correlation_id, headers, ttl, priority, persistent).
        """
        opts = options or PublishOptions()
        exchange = await self._ensure_exchange()

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        msg = aio_pika.Message(
            body=body,
            content_type=self._content_type,
            correlation_id=opts.correlation_id,
            headers=with_traceparent(opts.headers, opts.traceparent),
            expiration=opts.ttl,
            priority=opts.priority,
            delivery_mode=(
                aio_pika.DeliveryMode.PERSISTENT
                if opts.persistent
                else aio_pika.DeliveryMode.NOT_PERSISTENT
            ),
        )

        logger.info(
            "rabbitmq_publish",
            exchange=self._exchange_name,
            routing_key=self._routing_key,
            correlation_id=opts.correlation_id,
        )

        await exchange.publish(msg, routing_key=self._routing_key)

    async def _ensure_stream_producer(self) -> Producer:
        """Return the reusable Producer, creating it if necessary.

        Retries connection up to ``_STREAM_MAX_RETRIES`` times with
        exponential backoff to survive transient RabbitMQ Streams outages.
        """
        async with self._stream_producer_lock:
            if self._stream_producer is not None:
                return self._stream_producer

            cfg = self._client.config
            last_err: Exception | None = None

            for attempt in range(1, _STREAM_MAX_RETRIES + 1):
                try:
                    producer = Producer(
                        host=cfg.host,
                        port=cfg.stream_port,
                        username=cfg.username,
                        password=cfg.password,
                        vhost=cfg.virtual_host,
                    )
                    await producer.start()
                    self._stream_producer = producer
                    if attempt > 1:
                        logger.info(
                            "stream_producer_connected_after_retry",
                            attempt=attempt,
                        )
                    return self._stream_producer
                except Exception as exc:
                    last_err = exc
                    delay = _STREAM_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "stream_producer_connect_failed",
                        attempt=attempt,
                        max_retries=_STREAM_MAX_RETRIES,
                        delay=delay,
                        error=str(exc),
                    )
                    if attempt < _STREAM_MAX_RETRIES:
                        await asyncio.sleep(delay)

            raise ConnectionError(
                f"Could not connect to RabbitMQ Streams after "
                f"{_STREAM_MAX_RETRIES} attempts"
            ) from last_err

    async def _reset_stream_producer(self) -> None:
        """Close and discard the current stream producer so the next call
        to ``_ensure_stream_producer`` creates a fresh connection."""
        async with self._stream_producer_lock:
            if self._stream_producer is not None:
                try:
                    await self._stream_producer.close()
                except Exception:
                    pass
                self._stream_producer = None
                self._known_streams.clear()

    async def publish_stream(
        self,
        payload: Dict[str, Any],
        stream_name: str,
        options: Optional[PublishOptions] = None,
    ) -> None:
        """Publish a message to a RabbitMQ stream.

        If the send fails (e.g. broken connection), the producer is
        discarded and the operation is retried with a fresh connection
        up to ``_STREAM_MAX_RETRIES`` times.

        Args:
            payload: Dictionary with the message to publish.
            stream_name: Name of the stream (required).
            options: Publishing options. ``options.traceparent`` is echoed as
                the ``traceparent`` AMQP 1.0 application property.
        """
        opts = options or PublishOptions()
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_err: Exception | None = None

        # Wire format: with a traceparent, wrap the body in an AMQP 1.0 message
        # carrying it in application_properties so the trace continues across
        # the stream (the consumer reads it back). Without one, keep the
        # historical raw JSON bytes — byte-for-byte backward compatible.
        message: Any = (
            AMQPMessage(
                body=body,
                application_properties={"traceparent": opts.traceparent},
            )
            if opts.traceparent
            else body
        )

        for attempt in range(1, _STREAM_MAX_RETRIES + 1):
            try:
                producer = await self._ensure_stream_producer()

                if stream_name not in self._known_streams:
                    await producer.create_stream(stream_name, exists_ok=True)
                    self._known_streams.add(stream_name)

                await producer.send(stream=stream_name, message=message)

                logger.info(
                    "rabbitmq_stream_publish",
                    stream=stream_name,
                    correlation_id=opts.correlation_id,
                )
                return
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "rabbitmq_stream_publish_failed",
                    stream=stream_name,
                    attempt=attempt,
                    max_retries=_STREAM_MAX_RETRIES,
                    error=str(exc),
                )
                await self._reset_stream_producer()
                if attempt < _STREAM_MAX_RETRIES:
                    delay = _STREAM_BASE_DELAY * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)

        raise ConnectionError(
            f"Could not publish to stream '{stream_name}' after "
            f"{_STREAM_MAX_RETRIES} attempts"
        ) from last_err

    async def close(self) -> None:
        """Close the stream Producer if it exists."""
        async with self._stream_producer_lock:
            if self._stream_producer is not None:
                await self._stream_producer.close()
                self._stream_producer = None
                self._known_streams.clear()

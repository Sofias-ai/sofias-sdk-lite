"""RabbitMQ consumer built on aio-pika."""
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Optional, Type, cast

import aio_pika
from pydantic import BaseModel, ValidationError

from sofias_sdk_lite.messaging.models import AgentTaskMessage
from sofias_sdk_lite.observability._log import get_logger

from .client import RabbitMQClient
from .types import AckPolicy, MessageContext

__all__ = ["RabbitMQConsumer", "MessageHandler"]

logger = get_logger("rabbitmq.consumer")

# Public callback type
MessageHandler = Callable[[MessageContext], Awaitable[None]]


class RabbitMQConsumer:
    """Consumes messages from a queue and dispatches them to a handler.

    The handler receives a :class:`MessageContext` with the deserialized
    payload, AMQP metadata, and ack/nack helpers.

    Example:
        ```python

        async with RabbitMQClient(config) as client:
            consumer = RabbitMQConsumer(
                client=client,
                queue_name="my.queue",
                handler=my_handler,
            )
            await consumer.start()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await consumer.stop()
        ```
    """

    def __init__(
        self,
        *,
        client: RabbitMQClient,
        queue_name: str,
        handler: MessageHandler,
        ack_policy: AckPolicy = AckPolicy.AUTO,
        message_model: Type[BaseModel] = AgentTaskMessage,
    ) -> None:
        self._client = client
        self._queue_name = queue_name
        self._handler = handler
        self._ack_policy = ack_policy
        self._message_model = message_model

        self._queue: Optional[aio_pika.Queue] = None
        self._consumer_tag: Optional[str] = None
        self._started = False

    async def start(self) -> None:
        """Declare the queue and begin consuming messages."""
        if self._started:
            logger.warning("rabbitmq_consumer_already_running")
            return

        await self._client.ensure_connected()
        channel = self._client.channel

        self._queue = await channel.declare_queue(
            self._queue_name, durable=True,
        )

        self._consumer_tag = await self._queue.consume(
            self._on_message, no_ack=False,
        )

        self._started = True
        logger.info("rabbitmq_consumer_start", queue=self._queue_name)

    async def stop(self) -> None:
        """Cancel the consumer (does **not** close the connection)."""
        if self._consumer_tag and self._queue:
            await self._queue.cancel(self._consumer_tag)

        self._consumer_tag = None
        self._started = False
        logger.info("rabbitmq_consumer_stop", queue=self._queue_name)

    # -- internal -------------------------------------------------------------

    async def _on_message(self, message: aio_pika.IncomingMessage) -> None:
        """Dispatch an incoming AMQP message to the handler."""
        correlation_id = message.correlation_id or "unknown"

        # Deserialize body
        try:
            payload = json.loads(message.body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.error(
                "rabbitmq_message_invalid",
                error=str(exc),
                correlation_id=correlation_id,
            )
            await message.nack(requeue=False)
            return

        # Validate incoming payload against canonical schema
        try:
            self._message_model.model_validate(payload)
        except ValidationError as exc:
            logger.error(
                "rabbitmq_message_validation_failed",
                error=str(exc),
                correlation_id=correlation_id,
            )
            await message.nack(requeue=False)
            return

        # Trace context (W3C `traceparent`) is observability metadata: expose it
        # directly and typed. aio_pika may deliver header values as bytes or str
        # depending on the producer, so normalize defensively.
        headers = cast(dict[str, Any], message.headers or {})
        raw_tp: Any = headers.get("traceparent")
        traceparent: str | None = (
            raw_tp.decode()
            if isinstance(raw_tp, bytes)
            else (str(raw_tp) if raw_tp is not None else None)
        )

        ctx = MessageContext(
            payload=payload,
            headers=dict(message.headers or {}),
            traceparent=traceparent,
            routing_key=message.routing_key or "",
            exchange=message.exchange or "",
            delivery_tag=message.delivery_tag,
            correlation_id=message.correlation_id,
            reply_to=message.reply_to,
            timestamp=message.timestamp,
            _message=message,
        )

        if self._ack_policy is AckPolicy.AUTO:
            try:
                await self._handler(ctx)
                await message.ack()
            except Exception as exc:
                logger.error(
                    "rabbitmq_message_processing_failed",
                    correlation_id=correlation_id,
                    error=str(exc),
                    exc_info=True,
                )
                await message.nack(requeue=False)
        else:
            # MANUAL — handler is responsible for ack/nack via ctx
            await self._handler(ctx)

"""W3C ``traceparent`` on the wire: queue headers, stream application properties,
and the consumer surfacing it on ``MessageContext``."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from rstream import AMQPMessage

from sofias_sdk_lite.rabbitmq.consumer import RabbitMQConsumer
from sofias_sdk_lite.rabbitmq.publisher import RabbitMQPublisher
from sofias_sdk_lite.rabbitmq.types import MessageContext, PublishOptions, with_traceparent

TP = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


class TestWithTraceparent:
    def test_none_leaves_headers_untouched(self) -> None:
        assert with_traceparent(None, None) is None
        assert with_traceparent({"a": 1}, None) == {"a": 1}

    def test_adds_header_without_mutating_input(self) -> None:
        headers = {"a": 1}
        result = with_traceparent(headers, TP)
        assert result == {"a": 1, "traceparent": TP}
        assert headers == {"a": 1}

    def test_publish_options_default_has_no_traceparent(self) -> None:
        assert PublishOptions().traceparent is None


def _publisher_with_fake_producer() -> tuple[RabbitMQPublisher, AsyncMock]:
    publisher = RabbitMQPublisher(client=MagicMock(), exchange_name="", routing_key="")
    producer = AsyncMock()
    publisher._ensure_stream_producer = AsyncMock(return_value=producer)  # type: ignore[method-assign]
    return publisher, producer


class TestPublishStream:
    async def test_without_traceparent_sends_raw_json_bytes(self) -> None:
        publisher, producer = _publisher_with_fake_producer()

        await publisher.publish_stream({"k": "v"}, "s")

        sent = producer.send.await_args.kwargs["message"]
        assert isinstance(sent, bytes)
        assert json.loads(sent) == {"k": "v"}

    async def test_with_traceparent_wraps_in_amqp_message(self) -> None:
        publisher, producer = _publisher_with_fake_producer()

        await publisher.publish_stream({"k": "v"}, "s", PublishOptions(traceparent=TP))

        sent = producer.send.await_args.kwargs["message"]
        assert isinstance(sent, AMQPMessage)
        assert sent.application_properties == {"traceparent": TP}
        assert json.loads(bytes(sent.body)) == {"k": "v"}


class TestPublishQueue:
    async def test_traceparent_travels_as_amqp_header(self) -> None:
        publisher = RabbitMQPublisher(client=MagicMock(), exchange_name="x", routing_key="rk")
        exchange = AsyncMock()
        publisher._ensure_exchange = AsyncMock(return_value=exchange)  # type: ignore[method-assign]

        await publisher.publish({"k": "v"}, PublishOptions(headers={"h": "1"}, traceparent=TP))

        msg = exchange.publish.await_args.args[0]
        assert msg.headers["traceparent"] == TP
        assert msg.headers["h"] == "1"


def _incoming(headers: dict | None) -> MagicMock:
    message = MagicMock()
    message.body = json.dumps(
        {
            "content": "hi",
            "agent": "a",
            "tenant_identifier": "t",
            "conversation_id": "c",
            "message_id": 1,
        }
    ).encode()
    message.headers = headers
    message.correlation_id = None
    message.routing_key = "rk"
    message.exchange = ""
    message.delivery_tag = 1
    message.reply_to = None
    message.timestamp = None
    message.ack = AsyncMock()
    message.nack = AsyncMock()
    return message


class TestConsumerSurfacesTraceparent:
    @pytest.mark.parametrize(
        ("headers", "expected"),
        [
            ({"traceparent": TP}, TP),
            ({"traceparent": TP.encode()}, TP),
            ({}, None),
            (None, None),
        ],
    )
    async def test_context_carries_normalized_traceparent(self, headers, expected) -> None:
        seen: list[MessageContext] = []

        async def handler(ctx: MessageContext) -> None:
            seen.append(ctx)

        consumer = RabbitMQConsumer(client=MagicMock(), queue_name="q", handler=handler)
        message = _incoming(headers)

        await consumer._on_message(message)

        assert len(seen) == 1
        assert seen[0].traceparent == expected
        message.ack.assert_awaited_once()

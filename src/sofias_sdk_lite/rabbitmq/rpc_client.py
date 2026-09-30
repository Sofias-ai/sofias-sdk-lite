"""RabbitMQ RPC client for request-reply patterns."""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, AsyncIterator, Dict, Optional, Union

import aio_pika

from sofias_sdk_lite.observability._log import get_logger

from .client import RabbitMQClient

__all__ = ["RabbitMQRPCClient", "RPCError", "RPCTimeoutError"]

logger = get_logger("rabbitmq.rpc_client")


class RPCTimeoutError(Exception):
    """Raised when an RPC call exceeds the specified timeout."""

    pass


class RPCError(Exception):
    """Raised when an RPC call fails."""

    pass


class RabbitMQRPCClient:
    """RPC client for request-reply communication with RabbitMQ.

    This client implements the AMQP RPC pattern:
    1. Creates an exclusive reply queue for receiving responses
    2. Sends requests with reply_to and correlation_id properties
    3. Waits for responses matching the correlation_id

    Example:
        client = RabbitMQClient(config)
        await client.connect()

        async with RabbitMQRPCClient(client=client) as rpc:
            response = await rpc.call(
                queue="executor.tasks",
                payload={"task": "process_data", "data": {...}},
                timeout=30.0,
            )

        await client.disconnect()
    """

    def __init__(
        self,
        *,
        client: RabbitMQClient,
        content_type: str = "application/json",
    ) -> None:
        """Initialize the RPC client.

        Args:
            client: RabbitMQ client for connection management
            content_type: Content type for messages (default: application/json)
        """
        self._client = client
        self._content_type = content_type

        self._reply_queue: Optional[aio_pika.Queue] = None
        self._reply_queue_name: Optional[str] = None
        self._pending_calls: Dict[str, Union[asyncio.Future, asyncio.Queue]] = {}
        self._consumer_tag: Optional[str] = None
        self._channel: Optional[aio_pika.RobustChannel] = None
        self._setup_complete = False

    async def setup(self) -> None:
        """Set up the reply queue and start consuming responses.

        Must be called before making RPC calls.
        """
        if self._setup_complete:
            return

        await self._client.ensure_connected()

        # Get a dedicated channel for RPC
        self._channel = await self._client._connection.channel()

        # Declare exclusive auto-delete reply queue
        self._reply_queue = await self._channel.declare_queue(
            name="",  # Auto-generated name
            exclusive=True,
            auto_delete=True,
        )
        self._reply_queue_name = self._reply_queue.name

        # Start consuming responses
        self._consumer_tag = await self._reply_queue.consume(
            self._on_response,
            no_ack=True,  # Auto-ack since we're handling responses
        )

        self._setup_complete = True
        logger.info(
            "rpc_client_setup",
            reply_queue=self._reply_queue_name,
        )

    async def call(
        self,
        queue: str,
        payload: Dict[str, Any],
        timeout: float = 30.0,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make an RPC call and wait for the response.

        Args:
            queue: Target queue name (executor's task queue)
            payload: Message payload to send
            timeout: Maximum time to wait for response (seconds)
            correlation_id: Optional correlation ID (auto-generated if not provided)

        Returns:
            Response payload as a dictionary

        Raises:
            RPCTimeoutError: If response is not received within timeout
            RPCError: If the call fails for other reasons
        """
        if not self._setup_complete:
            await self.setup()

        # Generate correlation ID if not provided
        corr_id = correlation_id or str(uuid.uuid4())

        # Create a Future for this call
        future: asyncio.Future[Dict[str, Any]] = asyncio.Future()
        self._pending_calls[corr_id] = future

        try:
            # Prepare message
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            message = aio_pika.Message(
                body=body,
                content_type=self._content_type,
                correlation_id=corr_id,
                reply_to=self._reply_queue_name,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            )

            # Publish to default exchange with queue name as routing key
            await self._channel.default_exchange.publish(
                message,
                routing_key=queue,
            )

            logger.info(
                "rpc_call_sent",
                queue=queue,
                correlation_id=corr_id,
                reply_to=self._reply_queue_name,
            )

            # Wait for response with timeout
            try:
                response = await asyncio.wait_for(future, timeout=timeout)
                logger.info(
                    "rpc_call_completed",
                    queue=queue,
                    correlation_id=corr_id,
                )
                return response
            except asyncio.TimeoutError:
                logger.error(
                    "rpc_call_timeout",
                    queue=queue,
                    correlation_id=corr_id,
                    timeout=timeout,
                )
                raise RPCTimeoutError(
                    f"RPC call to '{queue}' timed out after {timeout}s"
                )
        finally:
            # Clean up pending call
            self._pending_calls.pop(corr_id, None)

    async def call_streaming(
        self,
        queue: str,
        payload: Dict[str, Any],
        timeout: float = 60.0,
        correlation_id: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Make an RPC call that expects multiple response fragments.

        Yields response fragments as they arrive until a fragment
        with is_last=True or complete=True is received.

        Args:
            queue: Target queue name
            payload: Message payload to send
            timeout: Maximum time to wait for complete response
            correlation_id: Optional correlation ID

        Yields:
            Response fragments as dictionaries

        Raises:
            RPCTimeoutError: If complete response not received within timeout
        """
        if not self._setup_complete:
            await self.setup()

        corr_id = correlation_id or str(uuid.uuid4())

        # Use an asyncio.Queue for streaming responses
        response_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._pending_calls[corr_id] = response_queue

        try:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            message = aio_pika.Message(
                body=body,
                content_type=self._content_type,
                correlation_id=corr_id,
                reply_to=self._reply_queue_name,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            )

            await self._channel.default_exchange.publish(
                message,
                routing_key=queue,
            )

            logger.info(
                "rpc_streaming_call_sent",
                queue=queue,
                correlation_id=corr_id,
            )

            deadline = asyncio.get_event_loop().time() + timeout

            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise RPCTimeoutError(
                        f"Streaming RPC to '{queue}' timed out after {timeout}s"
                    )

                try:
                    fragment = await asyncio.wait_for(
                        response_queue.get(),
                        timeout=remaining,
                    )
                    yield fragment

                    # Check if this is the last fragment
                    if fragment.get("is_last") or fragment.get("complete"):
                        break
                except asyncio.TimeoutError:
                    raise RPCTimeoutError(
                        f"Streaming RPC to '{queue}' timed out after {timeout}s"
                    )
        finally:
            self._pending_calls.pop(corr_id, None)

    async def _on_response(self, message: aio_pika.IncomingMessage) -> None:
        """Handle incoming response messages.

        Routes responses to the correct pending call based on correlation_id.
        """
        corr_id = message.correlation_id
        if not corr_id:
            logger.warning("rpc_response_no_correlation_id")
            return

        pending = self._pending_calls.get(corr_id)
        if not pending:
            logger.warning(
                "rpc_response_unknown_correlation_id",
                correlation_id=corr_id,
            )
            return

        try:
            payload = json.loads(message.body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            logger.error(
                "rpc_response_invalid_json",
                correlation_id=corr_id,
                error=str(exc),
            )
            if isinstance(pending, asyncio.Future):
                pending.set_exception(RPCError(f"Invalid JSON response: {exc}"))
            return

        logger.debug(
            "rpc_response_received",
            correlation_id=corr_id,
            is_last=payload.get("is_last"),
        )

        # Handle based on pending type (Future for single, Queue for streaming)
        if isinstance(pending, asyncio.Future):
            if not pending.done():
                pending.set_result(payload)
        elif isinstance(pending, asyncio.Queue):
            await pending.put(payload)

    async def close(self) -> None:
        """Close the RPC client and clean up resources."""
        if self._consumer_tag and self._reply_queue:
            await self._reply_queue.cancel(self._consumer_tag)

        if self._channel and not self._channel.is_closed:
            await self._channel.close()

        # Cancel any pending calls
        for corr_id, pending in self._pending_calls.items():
            if isinstance(pending, asyncio.Future) and not pending.done():
                pending.set_exception(
                    RPCError("RPC client closed while call was pending")
                )

        self._pending_calls.clear()
        self._setup_complete = False

        logger.info("rpc_client_closed")

    async def __aenter__(self) -> "RabbitMQRPCClient":
        """Async context manager entry."""
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

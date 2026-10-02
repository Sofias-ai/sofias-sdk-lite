"""RabbitMQ transport for delegation between agents."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

import aio_pika

from sofias_sdk_lite.messaging.models import AgentTaskMessage
from sofias_sdk_lite.nodes.delegation_transport import DelegationTransport
from sofias_sdk_lite.observability._log import get_logger

from .client import RabbitMQClient

__all__ = ["RabbitMQDelegationTransport", "DelegationTransportError"]

logger = get_logger("rabbitmq.delegation_transport")


class DelegationTransportError(Exception):
    """Base exception for delegation transport errors."""

    pass


class RabbitMQDelegationTransport:
    """Transport for delegation requests between agents via RabbitMQ.

    Implements the :class:`~sofias_sdk_lite.nodes.delegation_transport.DelegationTransport`
    protocol, separating the send and wait operations to support DAG execution
    patterns where multiple requests can be sent in parallel and responses
    collected as they arrive.

    Key differences from ``RabbitMQRPCClient``:
    - send_request() does NOT wait for response
    - wait_any_response() can wait for any of multiple correlation_ids
    - Responses arriving before wait_* are buffered, not lost

    Example:
        ```python
        client = RabbitMQClient(config)
        await client.connect()

        async with RabbitMQDelegationTransport(client=client) as transport:
            # Send multiple requests (tenant_identifier and conversation_id
            # are required, and the target queue must be an explicit
            # routing key — this SDK does not assume any naming convention)
            await transport.send_request(
                "agents.translator.tasks",
                {"text": "Hello"},
                "cid-1",
                tenant_identifier="tenant-123",
                conversation_id="conv-456",
            )
            await transport.send_request(
                "agents.summarizer.tasks",
                {"text": "..."},
                "cid-2",
                tenant_identifier="tenant-123",
                conversation_id="conv-456",
            )

            # Wait for any to complete (DAG pattern)
            cid, response = await transport.wait_any_response({"cid-1", "cid-2"}, timeout=30.0)
            print(f"{cid} responded with: {response}")

        await client.disconnect()
        ```
    """

    def __init__(
        self,
        *,
        client: RabbitMQClient,
        content_type: str = "application/json",
    ) -> None:
        """Initialize the delegation transport.

        Args:
            client: RabbitMQ client for connection management
            content_type: Content type for messages (default: application/json)
        """
        self._client = client
        self._content_type = content_type

        # Connection state
        self._channel: Optional[aio_pika.RobustChannel] = None
        self._reply_queue: Optional[aio_pika.Queue] = None
        self._reply_queue_name: Optional[str] = None
        self._consumer_tag: Optional[str] = None
        self._started = False

        # Response management
        self._received_responses: Dict[str, dict] = {}
        self._waiters: Dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Initialize the reply queue and start consuming responses.

        Creates an exclusive auto-delete queue for receiving responses
        and starts the consumer. Must be called before send_request().
        """
        if self._started:
            return

        await self._client.ensure_connected()

        # Get a dedicated channel for the transport
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
            no_ack=True,
        )

        self._started = True
        logger.info(
            "delegation_transport_started",
            reply_queue=self._reply_queue_name,
        )

    async def stop(self) -> None:
        """Stop consuming and clean up resources.

        Cancels all pending waiters and clears the response buffer.
        """
        if self._consumer_tag and self._reply_queue:
            await self._reply_queue.cancel(self._consumer_tag)

        if self._channel and not self._channel.is_closed:
            await self._channel.close()

        # Wake up and clean pending waiters
        async with self._lock:
            for event in self._waiters.values():
                event.set()
            self._waiters.clear()
            self._received_responses.clear()

        self._started = False
        logger.info("delegation_transport_stopped")

    async def send_request(
        self,
        target_queue: str,
        payload: dict,
        correlation_id: str,
        *,
        tenant_identifier: str,
        conversation_id: str,
        metadata: dict | None = None,
    ) -> None:
        """Send a delegation request without waiting for response.

        Builds a canonical ``AgentTaskMessage`` so the receiving agent gets
        the same format regardless of whether the message comes from the
        platform publisher or from another agent.

        Args:
            target_queue: Explicit routing key/queue name of the target
                agent. The SDK does not assume any naming convention —
                callers choose their own topology.
            payload: The task payload to send (serialised as JSON string in ``content``)
            correlation_id: Unique ID to correlate request/response
            tenant_identifier: Tenant identifier (required for routing responses)
            conversation_id: Conversation ID (required for context continuity)
            metadata: Optional metadata (trace_id, source_agent, etc.)

        Raises:
            DelegationTransportError: If transport is not started
        """
        if not self._started:
            raise DelegationTransportError(
                "Transport not started. Call start() first."
            )

        # Build canonical message — the receiver gets AgentTaskMessage
        # regardless of whether the sender is the platform or another agent.
        task_message = AgentTaskMessage(
            conversation_id=conversation_id,
            message_id=correlation_id,
            content=json.dumps(payload, ensure_ascii=False) if isinstance(payload, dict) else str(payload),
            agent=target_queue,
            tenant_identifier=tenant_identifier,
            reply_to=self._reply_queue_name,
            correlation_id=correlation_id,
        )

        body = task_message.model_dump_json().encode("utf-8")
        message = aio_pika.Message(
            body=body,
            content_type=self._content_type,
            correlation_id=correlation_id,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )

        await self._channel.default_exchange.publish(
            message,
            routing_key=target_queue,
        )

        logger.info(
            "delegation_request_sent",
            target_queue=target_queue,
            correlation_id=correlation_id,
        )

    async def wait_response(
        self,
        correlation_id: str,
        timeout: float,
    ) -> dict | None:
        """Wait for a specific response by correlation_id.

        Args:
            correlation_id: The correlation ID to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            The response payload dict, or None if timeout was reached.

        Raises:
            DelegationTransportError: If transport is not started or closed while waiting
        """
        if not self._started:
            raise DelegationTransportError("Transport not started")

        # Fast path: response already arrived
        async with self._lock:
            if correlation_id in self._received_responses:
                return self._received_responses.pop(correlation_id)

            # Create event for waiting
            event = asyncio.Event()
            self._waiters[correlation_id] = event

        try:
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                return None

            # Get response
            async with self._lock:
                if correlation_id not in self._received_responses:
                    raise DelegationTransportError(
                        "Transport closed while waiting for response"
                    )
                return self._received_responses.pop(correlation_id)

        finally:
            async with self._lock:
                self._waiters.pop(correlation_id, None)

    async def wait_any_response(
        self,
        correlation_ids: set[str],
        timeout: float,
    ) -> tuple[str, dict] | None:
        """Wait for any response from a set of correlation_ids.

        This is critical for DAG execution where multiple nodes run in parallel
        and we need to process whichever finishes first.

        Args:
            correlation_ids: Set of correlation IDs to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            Tuple of (correlation_id, response_payload) for the first response,
            or None if timeout was reached without any response.

        Raises:
            DelegationTransportError: If transport is not started or correlation_ids is empty
        """
        if not self._started:
            raise DelegationTransportError("Transport not started")

        if not correlation_ids:
            raise DelegationTransportError("correlation_ids cannot be empty")

        # Fast path: check if any response already arrived
        async with self._lock:
            for corr_id in correlation_ids:
                if corr_id in self._received_responses:
                    response = self._received_responses.pop(corr_id)
                    return (corr_id, response)

            # Create events for all IDs that don't have a response yet
            events_to_wait: Dict[str, asyncio.Event] = {}
            for corr_id in correlation_ids:
                if corr_id not in self._waiters:
                    event = asyncio.Event()
                    self._waiters[corr_id] = event
                    events_to_wait[corr_id] = event
                else:
                    events_to_wait[corr_id] = self._waiters[corr_id]

        try:
            wait_tasks: Dict[asyncio.Task, str] = {
                asyncio.create_task(event.wait()): corr_id
                for corr_id, event in events_to_wait.items()
            }

            try:
                done, pending = await asyncio.wait(
                    wait_tasks.keys(),
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in pending:
                    task.cancel()

                if not done:
                    return None

                async with self._lock:
                    for task in done:
                        corr_id = wait_tasks[task]
                        if corr_id in self._received_responses:
                            response = self._received_responses.pop(corr_id)
                            return (corr_id, response)

                raise DelegationTransportError(
                    "Response was consumed by another waiter"
                )

            except asyncio.CancelledError:
                for task in wait_tasks.keys():
                    task.cancel()
                raise

        finally:
            async with self._lock:
                for corr_id in correlation_ids:
                    if corr_id not in self._received_responses:
                        self._waiters.pop(corr_id, None)

    async def _on_response(self, message: aio_pika.IncomingMessage) -> None:
        """Handle incoming response messages.

        Stores the response in the buffer and notifies any waiting coroutines.
        """
        corr_id = message.correlation_id
        if not corr_id:
            logger.warning("delegation_response_no_correlation_id")
            return

        try:
            payload = json.loads(message.body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            logger.error(
                "delegation_response_invalid_json",
                correlation_id=corr_id,
                error=str(exc),
            )
            return

        async with self._lock:
            # Store response in buffer
            self._received_responses[corr_id] = payload

            # Notify waiter if exists
            if corr_id in self._waiters:
                self._waiters[corr_id].set()

        logger.debug(
            "delegation_response_received",
            correlation_id=corr_id,
        )

    async def __aenter__(self) -> "RabbitMQDelegationTransport":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.stop()


# Keep the DelegationTransport protocol import referenced so static analysis
# and runtime `isinstance()` checks can confirm RabbitMQDelegationTransport
# satisfies the protocol shape it is documented against.
_: type[DelegationTransport] = RabbitMQDelegationTransport

"""Event publisher/listener protocols for agent event integration.

Defines the ``EventPublisher``/``EventListener`` protocols the SDK expects
from an external message broker (RabbitMQ, Kafka, Redis Pub/Sub, ...), the
``EventAdapter`` convenience wrapper (standard event metadata, error
handling, agent-output/error helpers), and no-op ``Null*`` implementations
for local development and tests.

Protocol-only: no concrete broker client is vendored here (the RabbitMQ
transport lives in ``sofias_sdk_lite.rabbitmq``).
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Coroutine, Protocol, runtime_checkable

from sofias_sdk_lite.observability._log import get_logger

logger = get_logger("observability.events")

# Type alias for event handlers
EventHandler = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


@runtime_checkable
class EventPublisher(Protocol):
    """Protocol defining the interface for event publishing.

    This protocol defines what the SDK expects for publishing events
    to an external message queue. The concrete implementation should
    connect to RabbitMQ, Kafka, or another message broker.

    Example implementation:
        class RabbitMQPublisher:
            def __init__(self, connection):
                self._connection = connection
                self._channel = connection.channel()

            async def publish(
                self,
                event_type: str,
                payload: dict,
                routing_key: str | None = None,
            ) -> None:
                self._channel.basic_publish(
                    exchange="agents",
                    routing_key=routing_key or event_type,
                    body=json.dumps(payload),
                )
    """

    async def publish(
        self,
        event_type: str,
        payload: dict[str, Any],
        routing_key: str | None = None,
    ) -> None:
        """Publish an event to the message queue.

        Args:
            event_type: The type/name of the event.
            payload: The event data to publish.
            routing_key: Optional routing key for topic-based routing.
                If None, implementation may use event_type as routing key.
        """
        ...


@runtime_checkable
class EventListener(Protocol):
    """Protocol defining the interface for event subscription.

    This protocol defines what the SDK expects for subscribing to
    events from an external message queue.

    Example implementation:
        class RabbitMQListener:
            def __init__(self, connection):
                self._connection = connection
                self._channel = connection.channel()
                self._handlers = {}

            async def subscribe(
                self, event_type: str, handler: EventHandler
            ) -> None:
                queue = self._channel.queue_declare(queue=event_type)
                self._handlers[event_type] = handler

                def callback(ch, method, props, body):
                    asyncio.create_task(handler(json.loads(body)))

                self._channel.basic_consume(
                    queue=event_type,
                    on_message_callback=callback,
                    auto_ack=True,
                )

            async def unsubscribe(self, event_type: str) -> None:
                self._channel.queue_delete(queue=event_type)
                self._handlers.pop(event_type, None)
    """

    async def subscribe(
        self,
        event_type: str,
        handler: EventHandler,
    ) -> None:
        """Subscribe to events of a specific type.

        Args:
            event_type: The type of events to subscribe to.
            handler: Async function to call when events are received.
        """
        ...

    async def unsubscribe(self, event_type: str) -> None:
        """Unsubscribe from a specific event type.

        Args:
            event_type: The type of events to unsubscribe from.
        """
        ...


class EventAdapter:
    """Adapter that wraps EventPublisher and EventListener with SDK conventions.

    The EventAdapter mediates between SDK agents and the external event
    system. It adds:
    - Standard metadata to all published events (timestamp, agent name, etc.)
    - Convenience methods for common agent event patterns
    - Error handling and logging

    Example:
        ```python
        publisher = RabbitMQPublisher(connection)
        listener = RabbitMQListener(connection)
        events = EventAdapter(publisher, listener)

        # Publish agent output
        await events.publish_agent_output(
            agent_name="inbox_assist",
            agent_version="1.0.0",
            output={"response": "Hello!"},
            routing_key="inbox.response",
        )

        # Subscribe to agent input
        await events.subscribe_to_input(
            agent_name="inbox_assist",
            handler=my_handler,
        )
        ```
    """

    def __init__(
        self,
        publisher: EventPublisher | None = None,
        listener: EventListener | None = None,
    ) -> None:
        """Initialize the event adapter.

        Args:
            publisher: Optional event publisher for sending events.
            listener: Optional event listener for receiving events.

        Note:
            Either or both can be None if the agent only publishes
            or only subscribes.
        """
        self._publisher = publisher
        self._listener = listener

    @property
    def publisher(self) -> EventPublisher | None:
        """The underlying event publisher."""
        return self._publisher

    @property
    def listener(self) -> EventListener | None:
        """The underlying event listener."""
        return self._listener

    async def publish(
        self,
        event_type: str,
        payload: dict[str, Any],
        routing_key: str | None = None,
    ) -> bool:
        """Publish an event with error handling.

        Args:
            event_type: The type/name of the event.
            payload: The event data to publish.
            routing_key: Optional routing key.

        Returns:
            True if published successfully, False on error or no publisher.
        """
        if self._publisher is None:
            logger.warning(
                f"Cannot publish event '{event_type}': no publisher configured"
            )
            return False

        try:
            await self._publisher.publish(event_type, payload, routing_key)
            logger.debug(f"Published event '{event_type}' with key '{routing_key}'")
            return True
        except Exception as e:
            logger.error(
                f"Failed to publish event '{event_type}': {e}",
                exc_info=True,
            )
            return False

    async def subscribe(
        self,
        event_type: str,
        handler: EventHandler,
    ) -> bool:
        """Subscribe to events with error handling.

        Args:
            event_type: The type of events to subscribe to.
            handler: Async function to call when events are received.

        Returns:
            True if subscribed successfully, False on error or no listener.
        """
        if self._listener is None:
            logger.warning(
                f"Cannot subscribe to '{event_type}': no listener configured"
            )
            return False

        try:
            await self._listener.subscribe(event_type, handler)
            logger.debug(f"Subscribed to event type '{event_type}'")
            return True
        except Exception as e:
            logger.error(
                f"Failed to subscribe to '{event_type}': {e}",
                exc_info=True,
            )
            return False

    async def unsubscribe(self, event_type: str) -> bool:
        """Unsubscribe from events with error handling.

        Args:
            event_type: The type of events to unsubscribe from.

        Returns:
            True if unsubscribed successfully, False on error or no listener.
        """
        if self._listener is None:
            return False

        try:
            await self._listener.unsubscribe(event_type)
            logger.debug(f"Unsubscribed from event type '{event_type}'")
            return True
        except Exception as e:
            logger.error(
                f"Failed to unsubscribe from '{event_type}': {e}",
                exc_info=True,
            )
            return False

    async def publish_agent_output(
        self,
        agent_name: str,
        output: dict[str, Any],
        routing_key: str | None = None,
        *,
        agent_version: str | None = None,
        trace_id: str | None = None,
        execution_id: str | None = None,
    ) -> bool:
        """Publish an agent's output as a standardized event.

        This method wraps the output with standard metadata following
        the ecosystem's event conventions.

        Args:
            agent_name: Name of the agent producing the output.
            output: The agent's output dictionary.
            routing_key: Optional routing key for message routing.
            agent_version: Optional agent version string.
            trace_id: Optional trace ID for distributed tracing.
            execution_id: Optional unique ID for this execution.

        Returns:
            True if published successfully, False otherwise.
        """
        event_payload = {
            "event_type": "agent.output",
            "timestamp": time.time(),
            "agent_name": agent_name,
            "agent_version": agent_version,
            "trace_id": trace_id,
            "execution_id": execution_id or str(uuid.uuid4()),
            "output": output,
        }

        # Remove None values
        event_payload = {k: v for k, v in event_payload.items() if v is not None}

        return await self.publish(
            event_type="agent.output",
            payload=event_payload,
            routing_key=routing_key or f"{agent_name}.output",
        )

    async def subscribe_to_input(
        self,
        agent_name: str,
        handler: EventHandler,
        queue_name: str | None = None,
    ) -> bool:
        """Subscribe an agent to its input queue.

        Args:
            agent_name: Name of the agent to subscribe.
            handler: Async function to handle incoming input events.
            queue_name: Optional custom queue name. If None, uses
                "{agent_name}.input" as the queue/event type.

        Returns:
            True if subscribed successfully, False otherwise.
        """
        event_type = queue_name or f"{agent_name}.input"
        return await self.subscribe(event_type, handler)

    async def publish_agent_error(
        self,
        agent_name: str,
        error: Exception,
        context: dict[str, Any] | None = None,
        routing_key: str | None = None,
        *,
        agent_version: str | None = None,
        trace_id: str | None = None,
    ) -> bool:
        """Publish an agent error event for monitoring/alerting.

        Args:
            agent_name: Name of the agent that encountered the error.
            error: The exception that occurred.
            context: Optional context about the error (input, node, etc.).
            routing_key: Optional routing key.
            agent_version: Optional agent version string.
            trace_id: Optional trace ID for distributed tracing.

        Returns:
            True if published successfully, False otherwise.
        """
        event_payload = {
            "event_type": "agent.error",
            "timestamp": time.time(),
            "agent_name": agent_name,
            "agent_version": agent_version,
            "trace_id": trace_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
        }

        event_payload = {k: v for k, v in event_payload.items() if v is not None}

        return await self.publish(
            event_type="agent.error",
            payload=event_payload,
            routing_key=routing_key or f"{agent_name}.error",
        )


class NullEventPublisher:
    """A no-op event publisher for testing or local development."""

    async def publish(
        self,
        event_type: str,
        payload: dict[str, Any],
        routing_key: str | None = None,
    ) -> None:
        """Log the event instead of publishing."""
        logger.debug(
            f"[NullPublisher] Would publish '{event_type}' "
            f"(routing_key={routing_key}): {payload}"
        )


class NullEventListener:
    """A no-op event listener for testing or local development."""

    def __init__(self) -> None:
        self._handlers: dict[str, EventHandler] = {}

    async def subscribe(
        self,
        event_type: str,
        handler: EventHandler,
    ) -> None:
        """Store handler but don't actually subscribe."""
        self._handlers[event_type] = handler
        logger.debug(f"[NullListener] Registered handler for '{event_type}'")

    async def unsubscribe(self, event_type: str) -> None:
        """Remove handler."""
        self._handlers.pop(event_type, None)
        logger.debug(f"[NullListener] Unregistered handler for '{event_type}'")

    async def simulate_event(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Simulate receiving an event (for testing).

        Args:
            event_type: The event type to simulate.
            payload: The event payload.
        """
        handler = self._handlers.get(event_type)
        if handler:
            await handler(payload)


__all__ = [
    "EventHandler",
    "EventPublisher",
    "EventListener",
    "EventAdapter",
    "NullEventPublisher",
    "NullEventListener",
]

"""DelegationTransport protocol for delegation node communication.

This module defines the DelegationTransport protocol which abstracts
the transport layer for delegation operations. Implementations handle
the actual message passing (e.g., RabbitMQ, HTTP, in-memory for testing).

The DelegationNode uses this protocol to send requests and wait for responses
without knowing about the underlying transport mechanism. The concrete
RabbitMQ implementation lives in `sofias_sdk_lite.rabbitmq` and is never
imported from this module — only the protocol is defined here, plus a
NullDelegationTransport for local development and tests.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["DelegationTransport", "NullDelegationTransport"]


@runtime_checkable
class DelegationTransport(Protocol):
    """Protocol defining the interface for delegation transport.

    This protocol abstracts the transport layer for delegation operations,
    allowing DelegationNode to work with any implementation (RabbitMQ, HTTP,
    in-memory for testing, etc.).

    The transport is responsible for:
    - Managing its own reply queue/endpoint
    - Correlating requests with responses via correlation_id
    - Buffering responses until they are requested

    Example implementation:
        class RabbitMQDelegationTransport:
            async def start(self) -> None:
                # Create exclusive reply queue
                # Start consuming responses

            async def stop(self) -> None:
                # Clean up resources

            async def send_request(
                self,
                target_queue: str,
                payload: dict[str, Any],
                correlation_id: str,
                *,
                tenant_identifier: str,
                conversation_id: str,
                metadata: dict[str, Any] | None = None,
            ) -> None:
                # Publish to target queue with reply_to set

            async def wait_response(
                self,
                correlation_id: str,
                timeout: float,
            ) -> dict[str, Any] | None:
                # Wait for specific response, return None on timeout

            async def wait_any_response(
                self,
                correlation_ids: set[str],
                timeout: float,
            ) -> tuple[str, dict[str, Any]] | None:
                # Wait for any of the specified responses, return None on timeout
    """

    async def start(self) -> None:
        """Initialize the transport.

        Must be called before sending requests. Creates necessary
        resources (e.g., reply queues, connections).
        """
        ...

    async def stop(self) -> None:
        """Clean up the transport.

        Release all resources. Should be called when the transport
        is no longer needed.
        """
        ...

    async def send_request(
        self,
        target_queue: str,
        payload: dict[str, Any],
        correlation_id: str,
        *,
        tenant_identifier: str,
        conversation_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Send a delegation request without waiting for response.

        Args:
            target_queue: The queue/endpoint to send the request to.
            payload: The request payload to send.
            correlation_id: Unique ID to correlate request with response.
            tenant_identifier: Tenant identifier for routing and context.
            conversation_id: Conversation ID for context continuity.
            metadata: Optional metadata to include with the request.
        """
        ...

    async def wait_response(
        self,
        correlation_id: str,
        timeout: float,
    ) -> dict[str, Any] | None:
        """Wait for a specific response by correlation_id.

        Args:
            correlation_id: The correlation_id to wait for.
            timeout: Maximum time to wait in seconds.

        Returns:
            The response payload, or None if timeout was reached.
        """
        ...

    async def wait_any_response(
        self,
        correlation_ids: set[str],
        timeout: float,
    ) -> tuple[str, dict[str, Any]] | None:
        """Wait for any response from a set of correlation_ids.

        This method is critical for DAG execution where multiple
        requests are in flight and we need to process responses
        as they arrive.

        Args:
            correlation_ids: Set of correlation_ids to wait for.
            timeout: Maximum time to wait in seconds.

        Returns:
            Tuple of (correlation_id, response_payload) for the
            first response received, or None if timeout was reached.
        """
        ...


class NullDelegationTransport:
    """A no-op transport for testing or local development.

    This transport stores requests and allows simulating responses
    for testing purposes.
    """

    def __init__(self) -> None:
        self._started = False
        self._requests: list[dict[str, Any]] = []
        self._simulated_responses: dict[str, dict[str, Any]] = {}

    async def start(self) -> None:
        """Mark transport as started."""
        self._started = True

    async def stop(self) -> None:
        """Mark transport as stopped and clear state."""
        self._started = False
        self._requests.clear()
        self._simulated_responses.clear()

    async def send_request(
        self,
        target_queue: str,
        payload: dict[str, Any],
        correlation_id: str,
        *,
        tenant_identifier: str = "",
        conversation_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store the request for later inspection."""
        self._requests.append({
            "target_queue": target_queue,
            "payload": payload,
            "correlation_id": correlation_id,
            "tenant_identifier": tenant_identifier,
            "conversation_id": conversation_id,
            "metadata": metadata,
        })

    async def wait_response(
        self,
        correlation_id: str,
        timeout: float,
    ) -> dict[str, Any] | None:
        """Return simulated response if available, None otherwise."""
        if correlation_id in self._simulated_responses:
            return self._simulated_responses.pop(correlation_id)

        return None

    async def wait_any_response(
        self,
        correlation_ids: set[str],
        timeout: float,
    ) -> tuple[str, dict[str, Any]] | None:
        """Return first available simulated response, None otherwise."""
        for cid in correlation_ids:
            if cid in self._simulated_responses:
                response = self._simulated_responses.pop(cid)
                return (cid, response)

        return None

    def simulate_response(
        self,
        correlation_id: str,
        response: dict[str, Any],
    ) -> None:
        """Add a simulated response for testing.

        Args:
            correlation_id: The correlation_id to respond to.
            response: The response payload.
        """
        self._simulated_responses[correlation_id] = response

    @property
    def requests(self) -> list[dict[str, Any]]:
        """Get all captured requests."""
        return list(self._requests)

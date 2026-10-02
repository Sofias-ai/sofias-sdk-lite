"""Response workflow protocol for agent output delivery.

This module defines the ResponseWorkflow protocol — the abstraction that
determines how an agent delivers its response after execution. Concrete
implementations are provided by the consuming project:

- ChatResponseWorkflow: publishes to a chat stream (user-facing agents)
- DelegationResponseWorkflow: publishes to a reply queue (task agents)
- Any custom workflow the project requires

The SDK only defines the protocol contract. The Agent invokes the
workflow after building the AgentResponse, for both success and error
responses.

It also provides ``BaseDelegationResponseWorkflow``, a ready-to-use
concrete implementation for the delegation case. It builds a proper
``DelegationResponse`` (with token usage in ``extras``) and delegates
the actual publishing to a caller-supplied ``publish_fn``.

This module is transport-agnostic: it has no dependency on any specific
message broker. Wire it to whatever transport your application uses.

Example:
    ```python
    class MyRabbitMQWorkflow:
        async def send_response(
            self,
            response: AgentResponse,
            context: ExecutionContext,
        ) -> None:
            await self._publisher.publish(
                routing_key=f"agent.{context.agent_name}.responses",
                payload=response.model_dump(),
            )

    agent = (
        AgentBuilder("my_agent", version="1.0.0")
        ...
        .with_response_workflow(MyRabbitMQWorkflow())
        .build()
    )
    ```
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from sofias_sdk_lite.observability._log import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from sofias_sdk_lite.agent.execution_context import ExecutionContext
    from sofias_sdk_lite.agent.streaming import StreamEvent
    from sofias_sdk_lite.contracts.agent_contracts import AgentResponse

__all__ = [
    "ResponseWorkflow",
    "StreamingResponseWorkflow",
    "PublishFn",
    "BaseDelegationResponseWorkflow",
]


@runtime_checkable
class ResponseWorkflow(Protocol):
    """Protocol defining how an agent delivers its response.

    Implementations handle the transport-specific logic for sending
    the agent's response to the appropriate destination (chat stream,
    reply queue, webhook, etc.).

    The Agent calls send_response() after building the AgentResponse,
    for both success and error responses. Errors raised by the workflow
    are logged but do not affect the returned AgentResponse.
    """

    async def send_response(
        self,
        response: AgentResponse,
        context: ExecutionContext,
    ) -> None:
        """Deliver the agent's response.

        Args:
            response: The agent's output (success or error).
            context: The execution context with run metadata.
        """
        ...


@runtime_checkable
class StreamingResponseWorkflow(Protocol):
    """Protocol for workflows that receive streaming events in real time.

    Extends the ResponseWorkflow concept with an on_stream_event() method
    called for each StreamEvent during Agent.stream_execute(). The final
    send_response() is still called at the end with the complete response.

    Example:
        ```python
        class MySSEWorkflow:
            async def on_stream_event(
                self,
                event: StreamEvent,
                context: ExecutionContext,
            ) -> None:
                await self._sse_channel.send(event.model_dump_json())

            async def send_response(
                self,
                response: AgentResponse,
                context: ExecutionContext,
            ) -> None:
                await self._sse_channel.send_final(response.model_dump_json())
        ```
    """

    async def on_stream_event(
        self,
        event: StreamEvent,
        context: ExecutionContext,
    ) -> None:
        """Handle a streaming event during execution.

        Args:
            event: The streaming event (text chunk, tool call, etc.).
            context: The execution context with run metadata.
        """
        ...

    async def send_response(
        self,
        response: AgentResponse,
        context: ExecutionContext,
    ) -> None:
        """Deliver the agent's final response.

        Args:
            response: The agent's output (success or error).
            context: The execution context with run metadata.
        """
        ...


# Type alias for the transport-level publish function.
# Receives the serialized payload (dict) and must deliver it to the
# reply queue. The SDK does not care *how* — RabbitMQ, HTTP, in-memory, etc.
PublishFn = Callable[[dict[str, Any]], Awaitable[None]]


class BaseDelegationResponseWorkflow:
    """Ready-to-use ResponseWorkflow for agents that respond to delegations.

    Builds a proper ``DelegationResponse`` with token usage in ``extras``
    and delegates the actual publishing to a caller-supplied function.

    Args:
        publish_fn: Async callable that receives a dict payload and
            publishes it to the reply queue. The SDK is transport-agnostic;
            the caller wires this to whatever messaging system is needed.
        output_field: Optional key to extract from ``response.content`` as
            the ``result`` string. If *None*, the entire content dict is
            serialized as JSON. Defaults to *None*.

    Example:
        ```python

        async def publish(payload: dict) -> None:
            await my_transport.publish_reply(reply_to, correlation_id, payload)

        workflow = BaseDelegationResponseWorkflow(
            publish_fn=publish,
            output_field="report",
        )
        ```
    """

    def __init__(
        self,
        publish_fn: PublishFn,
        output_field: str | None = None,
    ) -> None:
        self._publish = publish_fn
        self._output_field = output_field

    def _extract_result(self, response: AgentResponse) -> str | None:
        """Extract the result string from an AgentResponse."""
        from sofias_sdk_lite.contracts.agent_contracts import ResponseStatus

        if response.status == ResponseStatus.ERROR:
            return None

        content = response.content
        if not content:
            return None

        if self._output_field is not None:
            return content.get(self._output_field, str(content))

        return json.dumps(content, ensure_ascii=False)

    async def send_response(
        self,
        response: AgentResponse,
        context: ExecutionContext,
    ) -> None:
        from sofias_sdk_lite.contracts.agent_contracts import ResponseStatus
        from sofias_sdk_lite.contracts.base_contracts import (
            DelegationResponse,
            DelegationStatus,
        )

        delegation_response = DelegationResponse(
            status=(
                DelegationStatus.SUCCESS
                if response.status == ResponseStatus.SUCCESS
                else DelegationStatus.ERROR
            ),
            result=self._extract_result(response),
            error_message=response.error_message,
            extras={
                "token_usage": response.usage,
            },
        )

        payload = delegation_response.model_dump()

        await self._publish(payload)

        logger.info(
            "delegation_response_published",
            agent_name=context.agent_name,
            status=delegation_response.status.value,
        )

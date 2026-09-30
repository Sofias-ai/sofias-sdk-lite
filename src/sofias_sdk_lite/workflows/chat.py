"""Chat response workflow — publishes agent responses as StreamFragment chunks.

Chunks the agent's answer and publishes it to a RabbitMQ stream. The stream
name is caller-provided (typically via ``RunnerConfig.response_stream``) —
this module does not assume any particular stream naming convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from sofias_sdk_lite.agent.execution_context import usage_state_from_list
from sofias_sdk_lite.messaging.models import StreamFragment
from sofias_sdk_lite.observability._log import get_logger
from sofias_sdk_lite.observability.trace_context import active_traceparent
from sofias_sdk_lite.rabbitmq.client import RabbitMQClient
from sofias_sdk_lite.rabbitmq.publisher import RabbitMQPublisher
from sofias_sdk_lite.rabbitmq.types import PublishOptions

if TYPE_CHECKING:
    from sofias_sdk_lite.agent.execution_context import ExecutionContext
    from sofias_sdk_lite.contracts.agent_contracts import AgentResponse

logger = get_logger("workflows.chat")

_DEFAULT_CHUNK_SIZE = 100

# A failed/empty AgentResponse (content={}) doesn't raise — the graph just
# returns nothing. Without this, the turn publishes a blank "completed"
# fragment: no error, no history entry, total silence. The caller has no way
# to know whether their request was ever handled (e.g. relaunching a survey
# they think never fired). Cross-agent bug, not specific to one graph.
EMPTY_RESPONSE_FALLBACK = (
    "Sorry, something went wrong processing this request. "
    "Nothing was saved. Please try again in a moment."
)

__all__ = ["ChatResponseWorkflow", "EMPTY_RESPONSE_FALLBACK"]


class ChatResponseWorkflow:
    """Publishes agent responses as StreamFragment chunks to a RabbitMQ stream.

    Implements the ``ResponseWorkflow`` protocol (``send_response``). Also
    provides ``send_error`` and ``close`` for infrastructure-level error
    handling and cleanup.

    Args:
        agent_name: Name used in ``StreamFragment.agent``.
        client: RabbitMQ client for publishing.
        stream_name: Target RabbitMQ stream name. No default naming
            convention is assumed — pass whatever stream your deployment uses
            (see ``RunnerConfig.response_stream``).
        conversation_id: Current conversation ID.
        message_id: Current message ID.
        chunk_size: Maximum characters per fragment (default 100).
        traceparent: W3C ``traceparent`` header from the inbound message. When
            set, it is echoed on every fragment's AMQP application properties
            so the streamed response continues the producer's trace end to end.
            If OpenTelemetry is installed and a span is active, that span's
            context wins over the inbound header.
    """

    def __init__(
        self,
        agent_name: str,
        client: RabbitMQClient,
        stream_name: str,
        conversation_id: str,
        message_id: str | int,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        traceparent: str | None = None,
    ) -> None:
        self._agent_name = agent_name
        self._publisher = RabbitMQPublisher(client=client, exchange_name="", routing_key="")
        self._stream_name = stream_name
        self._conversation_id = conversation_id
        self._message_id = message_id
        self._chunk_size = chunk_size
        self._traceparent = traceparent
        self._fragment_index: int = 0

    async def _publish_fragment(
        self,
        content: str,
        status: Literal["streaming", "completed", "error"],
        state: dict[str, Any] | None = None,
    ) -> None:
        is_last = status in ("completed", "error")
        fragment = StreamFragment(
            conversation_id=self._conversation_id,
            message_id=self._message_id,
            content=content,
            agent=self._agent_name,
            status=status,
            complete=is_last,
            fragment_index=self._fragment_index,
            is_last=is_last,
            state=state or {},
        )
        # The active span (this agent's own trace) wins over the inbound
        # traceparent; with no active span this is a pure echo of the inbound.
        await self._publisher.publish_stream(
            payload=fragment.model_dump(mode="json"),
            stream_name=self._stream_name,
            options=PublishOptions(traceparent=active_traceparent(self._traceparent)),
        )
        self._fragment_index += 1

    async def send_response(self, response: AgentResponse, context: ExecutionContext) -> None:
        """Publish the agent's answer as chunked StreamFragments.

        Extracts the answer from ``response.content`` (tries ``"answer"``
        then ``"content"`` keys) and chunks it into fragments. An empty answer
        is replaced by ``EMPTY_RESPONSE_FALLBACK`` so a failed turn is never
        delivered as silence. The turn's token usage (``response.usage``) is
        attached to the terminal fragment's ``state.usage``.
        """
        content: dict[str, Any] = response.content or {}
        answer = content.get("answer", "") or content.get("content", "")
        if not answer:
            logger.warning(
                "Empty response substituted with fallback message",
                conversation_id=self._conversation_id,
                status=getattr(response.status, "value", str(response.status)),
            )
            answer = EMPTY_RESPONSE_FALLBACK

        chunks = [answer[i : i + self._chunk_size] for i in range(0, len(answer), self._chunk_size)]
        usage_state = usage_state_from_list(getattr(response, "usage", None))
        for idx, chunk in enumerate(chunks):
            is_last = idx == len(chunks) - 1
            await self._publish_fragment(
                chunk,
                "completed" if is_last else "streaming",
                state={"usage": usage_state} if is_last and usage_state else None,
            )

        logger.info(
            "Response published",
            conversation_id=self._conversation_id,
            stream=self._stream_name,
            fragments=self._fragment_index,
        )

    async def send_error(self, error: str) -> None:
        """Publish an error fragment."""
        await self._publish_fragment(f"Error: {error}", "error")
        logger.error("Error response published", error=error)

    async def close(self) -> None:
        """Close the underlying stream producer."""
        await self._publisher.close()

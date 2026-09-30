"""Base incremental-streaming chat response workflow.

``ChatResponseWorkflow`` only implements the plain ``ResponseWorkflow``
protocol: it chunks the *final* answer after execution finishes. It does not
implement ``StreamingResponseWorkflow`` (``on_stream_event`` +
``send_response``, defined in ``sofias_sdk_lite.agent.response_workflow`` for
real-time, token-by-token publishing). Agents that needed true incremental
streaming had no base to subclass and each re-solved the same wire mechanics.

``BaseStreamingResponseWorkflow`` is that base. It owns everything transport-
and protocol-level; agents override a handful of hook points for their own
business logic:

- ``_final_text(response)``: pull the answer text out of ``response.content``.
- ``_extra_state(response)``: extra ``state`` keys for the terminal fragment
  (attachments, custom flags, ...).
- ``_empty_response_fallback(response)``: message substituted when a turn that
  never streamed anything resolves to empty text; ``None`` opts out.
- ``_safe_flush_len(buffer)``: cap how much of the buffer ``on_stream_event``
  may flush this round (e.g. holding back an unclosed structured tag).
- ``_should_buffer_text(event, context)``: whether to buffer a given
  ``TextChunkEvent`` at all (e.g. suppressing a launch-turn's ACK text).
- ``_on_tool_event(event, context)``: react to non-text StreamEvents (tool
  calls, artifacts, progress chips, ...). No-op by default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from sofias_sdk_lite.agent.execution_context import usage_state_from_list
from sofias_sdk_lite.agent.streaming import TextChunkEvent
from sofias_sdk_lite.messaging.models import StreamFragment
from sofias_sdk_lite.observability._log import get_logger
from sofias_sdk_lite.observability.trace_context import active_traceparent
from sofias_sdk_lite.rabbitmq.client import RabbitMQClient
from sofias_sdk_lite.rabbitmq.publisher import RabbitMQPublisher
from sofias_sdk_lite.rabbitmq.types import PublishOptions
from sofias_sdk_lite.workflows.chat import EMPTY_RESPONSE_FALLBACK

if TYPE_CHECKING:
    from sofias_sdk_lite.agent.execution_context import ExecutionContext
    from sofias_sdk_lite.agent.streaming import StreamEvent
    from sofias_sdk_lite.contracts.agent_contracts import AgentResponse

logger = get_logger("workflows.streaming_chat")

_DEFAULT_CHUNK_SIZE = 100
_FLUSH_SEPARATORS = " \n\t"

__all__ = ["BaseStreamingResponseWorkflow"]


def _flush_cut(buffer: str, limit: int) -> int:
    """Index at which to cut ``buffer`` so a fragment never splits a word.

    Returns the position right after the last separator inside
    ``buffer[:limit]``. With no separator in that window (a single word or
    URL longer than the limit), cuts at ``limit`` — the buffer must not grow
    unbounded while waiting for a separator that may never come.
    """
    window = buffer[:limit]
    for i in range(len(window) - 1, -1, -1):
        if window[i] in _FLUSH_SEPARATORS:
            return i + 1
    return limit


def _split_for_stream(text: str, limit: int) -> list[str]:
    """Split ``text`` into fragments of at most ``limit`` chars, word-safe.

    Concatenating the result reproduces ``text`` exactly. Used for the
    no-streaming-occurred path, where the whole answer is chunked at once.
    """
    if not text:
        return [text]
    fragments: list[str] = []
    remaining = text
    while remaining:
        cut = _flush_cut(remaining, limit) if len(remaining) > limit else len(remaining)
        fragments.append(remaining[:cut])
        remaining = remaining[cut:]
    return fragments


class BaseStreamingResponseWorkflow:
    """Publishes agent responses as StreamFragment chunks, incrementally.

    Implements the ``StreamingResponseWorkflow`` protocol. Subclass it and
    override the hook methods for agent-specific business logic; the wire
    mechanics (``_publish_fragment``, chunking, traceparent echo, usage) are
    inherited as-is.

    Args:
        agent_name: Used in ``StreamFragment.agent``.
        client: RabbitMQ client. Pass exactly one of ``client``/``publisher``;
            with ``client`` the workflow builds (and owns/closes) its own
            ``RabbitMQPublisher``.
        publisher: An already-built, externally-owned ``RabbitMQPublisher``
            (e.g. shared across many conversations for connection reuse).
            ``close()`` is a no-op when the publisher was injected this way.
        stream_name: Target RabbitMQ stream.
        conversation_id: Current conversation ID.
        message_id: Current message ID (echoed by the agent, matched back by
            the consumer — accepts ``int | str``).
        chunk_size: Maximum characters per fragment.
        traceparent: W3C ``traceparent`` from the inbound message, echoed on
            every fragment unless an active OpenTelemetry span overrides it.
        stream_node_names: If set, only ``TextChunkEvent``s from these graph
            nodes are streamed (multi-node agents that only want to expose
            one node's output).
        stream_key: If set, every fragment carries this key instead of
            defaulting to ``conversation_id`` — lets the consumer multiplex
            several independent streams onto the same conversation (e.g. a
            background job's progress bubble vs. the main reply).
        first_flush_prefix: One-shot text prepended to the very first
            published fragment (e.g. a "done" chip that replaces an
            already-open progress indicator the instant real content lands).
    """

    def __init__(
        self,
        agent_name: str,
        *,
        client: RabbitMQClient | None = None,
        publisher: RabbitMQPublisher | None = None,
        stream_name: str,
        conversation_id: str,
        message_id: str | int,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        traceparent: str | None = None,
        stream_node_names: set[str] | None = None,
        stream_key: str | None = None,
        first_flush_prefix: str = "",
    ) -> None:
        if (client is None) == (publisher is None):
            raise ValueError("Pass exactly one of `client` or `publisher`.")
        self._agent_name = agent_name
        self._owns_publisher = publisher is None
        if publisher is not None:
            self._publisher = publisher
        else:
            assert client is not None
            self._publisher = RabbitMQPublisher(client=client, exchange_name="", routing_key="")
        self._stream_name = stream_name
        self._conversation_id = conversation_id
        self._message_id = message_id
        self._chunk_size = chunk_size
        self._traceparent = traceparent
        self._stream_nodes = stream_node_names
        self._stream_key = stream_key
        self._first_flush_prefix = first_flush_prefix
        self._buffer = ""
        self._fragment_index = 0

    # -- Override points for subclasses ------------------------------------

    def _final_text(self, response: AgentResponse) -> str:
        """The authoritative answer text. Default: content["answer"] or
        content["content"] — override for a different output field/shape."""
        content: dict[str, Any] = response.content or {}
        return content.get("answer", "") or content.get("content", "")

    def _extra_state(self, response: AgentResponse) -> dict[str, Any]:
        """Extra keys to merge into the terminal fragment's ``state``
        (attachments, custom flags, ...). Default: none."""
        return {}

    def _empty_response_fallback(self, response: AgentResponse) -> str | None:
        """Message substituted for a turn that never streamed anything and
        resolves to empty ``_final_text`` — never publish pure silence.
        Override for a different message or locale; return ``None`` to
        opt out and keep publishing an empty completed fragment."""
        return EMPTY_RESPONSE_FALLBACK

    def _safe_flush_len(self, buffer: str) -> int:
        """How much of ``buffer`` ``on_stream_event`` may flush this round.

        Override to hold content back pending a later signal — e.g. an
        unclosed structured tag whose closer may arrive in a later chunk, or
        narration that must wait for a tool result before it's known whether
        to keep or discard it. Default: the whole buffer."""
        return len(buffer)

    def _should_buffer_text(self, event: TextChunkEvent, context: ExecutionContext) -> bool:
        """Whether to buffer this TextChunkEvent at all. Default: True.

        Override to drop text outright for turns whose visible reply must
        stay empty (e.g. a background job already owns the user-visible
        feedback for this turn)."""
        return True

    async def _on_tool_event(self, event: StreamEvent, context: ExecutionContext) -> None:
        """React to a non-text StreamEvent (tool calls, artifacts, progress
        chips, ...). No-op by default."""

    def _final_replace_state(self) -> dict[str, Any] | None:
        """State for an error/notice fragment: replace the buffer if
        streaming already happened, so deliberative text already streamed
        doesn't linger under the error/notice; append (None) otherwise."""
        if self._fragment_index > 0:
            return {"replace_content": True}
        return None

    # -- Wire mechanics -----------------------------------------------------

    async def _publish_fragment(
        self,
        content: str,
        status: Literal["streaming", "completed", "error"],
        state: dict[str, Any] | None = None,
    ) -> None:
        is_last = status in ("completed", "error")
        if self._first_flush_prefix:
            content = self._first_flush_prefix + content
            self._first_flush_prefix = ""
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
        payload = fragment.model_dump(mode="json")
        if self._stream_key:
            payload["stream_key"] = self._stream_key

        await self._publisher.publish_stream(
            payload=payload,
            stream_name=self._stream_name,
            options=PublishOptions(traceparent=active_traceparent(self._traceparent)),
        )
        self._fragment_index += 1

    # -- StreamingResponseWorkflow protocol ----------------------------------

    async def on_stream_event(self, event: StreamEvent, context: ExecutionContext) -> None:
        if not isinstance(event, TextChunkEvent):
            await self._on_tool_event(event, context)
            return
        if self._stream_nodes and event.node_name not in self._stream_nodes:
            return
        if not self._should_buffer_text(event, context):
            return

        self._buffer += event.content
        safe_len = min(self._safe_flush_len(self._buffer), len(self._buffer))
        while safe_len >= self._chunk_size:
            cut = _flush_cut(self._buffer[:safe_len], self._chunk_size)
            chunk = self._buffer[:cut]
            self._buffer = self._buffer[cut:]
            safe_len -= cut
            await self._publish_fragment(chunk, "streaming")

    async def send_response(self, response: AgentResponse, context: ExecutionContext) -> None:
        final_text = self._final_text(response)
        if not final_text and self._fragment_index == 0:
            fallback = self._empty_response_fallback(response)
            if fallback:
                logger.warning(
                    "Empty response substituted with fallback message",
                    conversation_id=self._conversation_id,
                    status=getattr(response.status, "value", str(response.status)),
                )
                final_text = fallback
        state = self._extra_state(response)
        if final_text:
            state["final_content"] = final_text
        usage = usage_state_from_list(getattr(response, "usage", None))
        if usage:
            state["usage"] = usage

        if self._fragment_index == 0:
            # Nothing streamed yet — chunk the whole answer now.
            fragments = _split_for_stream(final_text, self._chunk_size) if final_text else [""]
            for idx, fragment in enumerate(fragments):
                is_last = idx == len(fragments) - 1
                await self._publish_fragment(
                    fragment,
                    "completed" if is_last else "streaming",
                    state=state if is_last else None,
                )
        else:
            # Streaming happened — send the authoritative final text with
            # replace_content so the UI discards whatever was streamed
            # in favor of the confirmed final answer.
            self._buffer = ""
            if final_text:
                state["replace_content"] = True
                await self._publish_fragment(final_text, "completed", state=state)
            else:
                await self._publish_fragment("", "completed", state=state)

        logger.info(
            "Streamed response published",
            stream=self._stream_name,
            conversation_id=self._conversation_id,
            fragments_sent=self._fragment_index,
        )

    async def send_error(self, error_message: str) -> None:
        await self._publish_fragment(
            f"Error: {error_message}", "error", state=self._final_replace_state()
        )
        logger.error("Error response published", error=error_message)

    async def send_notice(self, message: str) -> None:
        """Publish *message* as a normal, completed reply — no "Error:"
        prefix, so a degraded-but-graceful message renders as an ordinary
        chat bubble rather than an error banner."""
        await self._publish_fragment(message, "completed", state=self._final_replace_state())

    async def close(self) -> None:
        """Close the publisher, if this workflow built its own."""
        if self._owns_publisher:
            await self._publisher.close()

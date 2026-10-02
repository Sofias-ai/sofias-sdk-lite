"""History provider protocol — a lightweight replacement for the internal
SQLite-backed memory stack, which is out of scope for the public SDK (v1).

``AgentRunner`` accepts an optional ``HistoryProvider`` so callers can
inject any backend (Redis, a database, a message log, ...);
``InMemoryHistory`` is the default, suitable for local development, tests,
and single-process deployments. No persistent backend ships here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """A single conversation turn stored in conversation history."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant", "system"]
    """Who produced this message."""

    content: str
    """The message text."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    """When the message was recorded (UTC)."""


@runtime_checkable
class HistoryProvider(Protocol):
    """Protocol defining the interface for conversation history backends.

    Unlike ``MemoryProvider`` (long-term summaries/facts/preferences) or
    ``ConversationStateProvider`` (ephemeral flow-scoped data), a
    ``HistoryProvider`` deals only with the ordered list of turns exchanged
    in a conversation.

    Example implementation::

        class RedisHistory:
            def __init__(self, redis_client):
                self._redis = redis_client

            async def get(self, conversation_id: str) -> list[Message]:
                raw = await self._redis.lrange(f"history:{conversation_id}", 0, -1)
                return [Message.model_validate_json(item) for item in raw]

            async def append(self, conversation_id: str, message: Message) -> None:
                await self._redis.rpush(
                    f"history:{conversation_id}", message.model_dump_json()
                )
    """

    async def get(self, conversation_id: str) -> list[Message]:
        """Retrieve the full message history for a conversation.

        Args:
            conversation_id: Unique identifier for the conversation.

        Returns:
            List of messages in chronological order. Empty list if the
            conversation has no recorded history.
        """
        ...

    async def append(self, conversation_id: str, message: Message) -> None:
        """Append a message to a conversation's history.

        Args:
            conversation_id: Unique identifier for the conversation.
            message: The message to record.
        """
        ...


@runtime_checkable
class SummarizingHistoryProvider(HistoryProvider, Protocol):
    """A ``HistoryProvider`` that can fold old turns into a summary.

    ``AgentRunner`` calls ``maybe_summarize`` *after* the response has been
    delivered, as a background task that never blocks the next message and is
    drained on graceful shutdown. Implementations decide whether a summary is
    due (e.g. by message count or token estimate) and how to store it; the
    runner only guarantees the call happens once per turn.

    Example implementation::

        class SummarizingRedisHistory(RedisHistory):
            async def maybe_summarize(self, conversation_id: str) -> None:
                messages = await self.get(conversation_id)
                if len(messages) < 40:
                    return
                summary = await self._llm_summarize(messages[:-10])
                await self._replace(conversation_id, [summary, *messages[-10:]])
    """

    async def maybe_summarize(self, conversation_id: str) -> None:
        """Summarise older turns if the conversation needs it.

        Called fire-and-forget after each delivered response. Must be safe to
        call when nothing needs summarising.
        """
        ...


class InMemoryHistory:
    """In-memory HistoryProvider for local development and tests.

    Stores messages in a dict of lists. Not suitable for production
    (history is lost on process restart).

    Example:
        ```python
        history = InMemoryHistory()
        await history.append("conv-1", Message(role="user", content="hi"))
        messages = await history.get("conv-1")  # [Message(role="user", ...)]
        ```
    """

    def __init__(self) -> None:
        self._store: dict[str, list[Message]] = {}

    async def get(self, conversation_id: str) -> list[Message]:
        """Return a copy of the recorded messages for a conversation."""
        return list(self._store.get(conversation_id, []))

    async def append(self, conversation_id: str, message: Message) -> None:
        """Record a new message for a conversation."""
        self._store.setdefault(conversation_id, []).append(message)


__all__ = ["Message", "HistoryProvider", "SummarizingHistoryProvider", "InMemoryHistory"]

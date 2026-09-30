"""Conversation state protocol and adapter for multi-turn flow persistence.

Provides the ``ConversationStateProvider`` protocol, the
``ConversationStateAdapter`` wrapper (adds error handling around a
provider), and ``InMemoryStateProvider`` for local development and tests.

Conceptually distinct from ``MemoryProvider`` (see
``sofias_sdk_lite.state.memory``):

- ``MemoryProvider``: long-lived conversation history (messages,
  summaries, facts) that grows over time and is never deleted.
- ``ConversationState``: ephemeral, flow-scoped data (e.g. validation
  questions, attempt counts) that is created and cleaned up within a
  multi-turn flow.

No concrete persistent backend (SQLite, Redis, etc.) ships here — bring
your own by implementing ``ConversationStateProvider``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from sofias_sdk_lite.observability._log import get_logger

logger = get_logger("state.conversation_state")


@runtime_checkable
class ConversationStateProvider(Protocol):
    """Protocol defining the interface for conversation state backends.

    Implementations can use any storage backend (SQLite, Redis, a
    database, etc.). State is scoped by ``(conversation_id, key)`` pairs.
    Each key stores a dict value. Keys are arbitrary strings chosen by the
    node (e.g. ``"validation:doc123"``).

    Conceptually distinct from ``MemoryProvider``:

    - ``MemoryProvider``: long-lived conversation history (messages,
      summaries, facts) that grows over time and is never deleted.
    - ``ConversationState``: ephemeral, flow-scoped data (e.g. validation
      questions, attempt counts) that is created and cleaned up within a
      multi-turn flow.

    Example implementation::

        class RedisStateProvider:
            def __init__(self, redis_client):
                self._redis = redis_client

            async def get(self, conversation_id: str, key: str) -> dict | None:
                data = await self._redis.get(f"state:{conversation_id}:{key}")
                return json.loads(data) if data else None

            async def set(self, conversation_id: str, key: str, value: dict) -> None:
                await self._redis.set(
                    f"state:{conversation_id}:{key}", json.dumps(value)
                )

            async def delete(self, conversation_id: str, key: str) -> None:
                await self._redis.delete(f"state:{conversation_id}:{key}")
    """

    async def get(
        self, conversation_id: str, key: str
    ) -> dict[str, Any] | None:
        """Retrieve state for a conversation and key.

        Args:
            conversation_id: Unique identifier for the conversation.
            key: State key (e.g. "validation:doc123").

        Returns:
            The stored dict, or None if no state exists for this key.
        """
        ...

    async def set(
        self, conversation_id: str, key: str, value: dict[str, Any]
    ) -> None:
        """Store state for a conversation and key.

        Args:
            conversation_id: Unique identifier for the conversation.
            key: State key (e.g. "validation:doc123").
            value: Dict to persist. Overwrites any existing value.
        """
        ...

    async def delete(self, conversation_id: str, key: str) -> None:
        """Delete state for a conversation and key.

        Args:
            conversation_id: Unique identifier for the conversation.
            key: State key to delete. No-op if key doesn't exist.
        """
        ...


class ConversationStateAdapter:
    """Adapter that wraps a ConversationStateProvider with error handling.

    Failures are logged but don't crash the agent by default
    (``raise_on_error=False``).

    Example:
        provider = InMemoryStateProvider()
        state = ConversationStateAdapter(provider)

        await state.set("conv-123", "validation:doc1", {"questions": [...]})
        data = await state.get("conv-123", "validation:doc1")
        await state.delete("conv-123", "validation:doc1")
    """

    def __init__(
        self,
        provider: ConversationStateProvider,
        *,
        raise_on_error: bool = False,
    ) -> None:
        """Initialize the conversation state adapter.

        Args:
            provider: The state provider implementation to delegate to.
            raise_on_error: If True, re-raise exceptions from provider.
                If False (default), log errors and return None/False.
        """
        self._provider = provider
        self._raise_on_error = raise_on_error

    @property
    def provider(self) -> ConversationStateProvider:
        """The underlying state provider."""
        return self._provider

    async def get(
        self, conversation_id: str, key: str
    ) -> dict[str, Any] | None:
        """Retrieve state with error handling.

        Args:
            conversation_id: Unique identifier for the conversation.
            key: State key.

        Returns:
            The stored dict, or None if not found or on error.
        """
        try:
            return await self._provider.get(conversation_id, key)
        except Exception as e:
            logger.error(
                "Failed to get conversation state",
                conversation_id=conversation_id,
                key=key,
                error=str(e),
                exc_info=True,
            )
            if self._raise_on_error:
                raise
            return None

    async def set(
        self, conversation_id: str, key: str, value: dict[str, Any]
    ) -> bool:
        """Store state with error handling.

        Args:
            conversation_id: Unique identifier for the conversation.
            key: State key.
            value: Dict to persist.

        Returns:
            True if stored successfully, False on error.
        """
        try:
            await self._provider.set(conversation_id, key, value)
            return True
        except Exception as e:
            logger.error(
                "Failed to set conversation state",
                conversation_id=conversation_id,
                key=key,
                error=str(e),
                exc_info=True,
            )
            if self._raise_on_error:
                raise
            return False

    async def delete(self, conversation_id: str, key: str) -> bool:
        """Delete state with error handling.

        Args:
            conversation_id: Unique identifier for the conversation.
            key: State key to delete.

        Returns:
            True if deleted successfully, False on error.
        """
        try:
            await self._provider.delete(conversation_id, key)
            return True
        except Exception as e:
            logger.error(
                "Failed to delete conversation state",
                conversation_id=conversation_id,
                key=key,
                error=str(e),
                exc_info=True,
            )
            if self._raise_on_error:
                raise
            return False


class InMemoryStateProvider:
    """In-memory state provider for local development and tests.

    Stores state in a nested dict. Not suitable for production
    (state is lost on process restart).

    Example:
        provider = InMemoryStateProvider()
        await provider.set("conv-1", "key-a", {"foo": "bar"})
        data = await provider.get("conv-1", "key-a")  # {"foo": "bar"}
        await provider.delete("conv-1", "key-a")
        data = await provider.get("conv-1", "key-a")  # None
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, dict[str, Any]]] = {}

    async def get(
        self, conversation_id: str, key: str
    ) -> dict[str, Any] | None:
        """Retrieve state from memory."""
        return self._store.get(conversation_id, {}).get(key)

    async def set(
        self, conversation_id: str, key: str, value: dict[str, Any]
    ) -> None:
        """Store state in memory."""
        if conversation_id not in self._store:
            self._store[conversation_id] = {}
        self._store[conversation_id][key] = value

    async def delete(self, conversation_id: str, key: str) -> None:
        """Delete state from memory."""
        conv = self._store.get(conversation_id)
        if conv is not None:
            conv.pop(key, None)


__all__ = [
    "ConversationStateProvider",
    "ConversationStateAdapter",
    "InMemoryStateProvider",
]

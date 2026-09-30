"""Memory provider protocol.

Defines the interface the SDK expects from an external long-lived memory
system (conversation history, summaries, facts, user preferences). This
module ships the protocol only — no concrete backend (Redis, Postgres, a
vector store, etc.) is vendored. Memory is out of scope for v1 of the
public SDK; the protocol ships so agents can be built against a stable
interface today and wired up to a real backend later.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MemoryProvider(Protocol):
    """Protocol defining the interface for memory providers.

    This protocol defines what the SDK expects from an external memory
    system. The concrete implementation should be provided by the agent
    that injects its memory library (Redis, PostgreSQL, vector store, etc.).

    The memory system handles two types of data:
    1. Conversation history: Recent messages in a conversation
    2. Long-term context: Summaries, facts, user preferences, etc.

    Example implementation:
        class RedisMemoryProvider:
            def __init__(self, redis_client):
                self._redis = redis_client

            async def get_conversation_history(
                self, conversation_id: str, max_messages: int | None = None
            ) -> list[dict]:
                key = f"conv:{conversation_id}"
                messages = await self._redis.lrange(key, 0, max_messages or -1)
                return [json.loads(m) for m in messages]

            async def store_message(
                self, conversation_id: str, message: dict
            ) -> None:
                key = f"conv:{conversation_id}"
                await self._redis.rpush(key, json.dumps(message))

            async def get_context(self, conversation_id: str) -> dict:
                key = f"ctx:{conversation_id}"
                data = await self._redis.get(key)
                return json.loads(data) if data else {}

            async def store_context(
                self, conversation_id: str, context: dict
            ) -> None:
                key = f"ctx:{conversation_id}"
                await self._redis.set(key, json.dumps(context))
    """

    async def get_conversation_history(
        self,
        conversation_id: str,
        max_messages: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve conversation history for a conversation.

        Args:
            conversation_id: Unique identifier for the conversation.
            max_messages: Maximum number of messages to return.
                If None, returns all available messages.

        Returns:
            List of message dictionaries, typically with keys like:
            - role: "user" | "assistant" | "system"
            - content: The message content
            - timestamp: When the message was sent
            - metadata: Optional additional data
        """
        ...

    async def store_message(
        self,
        conversation_id: str,
        message: dict[str, Any],
    ) -> None:
        """Store a message in the conversation history.

        Args:
            conversation_id: Unique identifier for the conversation.
            message: Message dictionary to store.
        """
        ...

    async def get_context(self, conversation_id: str) -> dict[str, Any]:
        """Retrieve long-term context for a conversation.

        Long-term context includes things like:
        - Conversation summaries
        - Extracted facts
        - User preferences
        - Entity references

        Args:
            conversation_id: Unique identifier for the conversation.

        Returns:
            Context dictionary. Empty dict if no context exists.
        """
        ...

    async def store_context(
        self,
        conversation_id: str,
        context: dict[str, Any],
    ) -> None:
        """Store long-term context for a conversation.

        Args:
            conversation_id: Unique identifier for the conversation.
            context: Context dictionary to store.
        """
        ...


__all__ = ["MemoryProvider"]

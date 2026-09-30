"""Conversation state, memory, and history protocols for sofias_sdk_lite.

Memory and history persistence are out of scope for v1: only protocols
and in-memory reference implementations ship here. No SQLite or other
concrete storage backend is vendored.
"""

from __future__ import annotations

from sofias_sdk_lite.state.conversation_state import (
    ConversationStateAdapter,
    ConversationStateProvider,
    InMemoryStateProvider,
)
from sofias_sdk_lite.state.history import (
    HistoryProvider,
    InMemoryHistory,
    Message,
    SummarizingHistoryProvider,
)
from sofias_sdk_lite.state.memory import MemoryProvider

__all__ = [
    "ConversationStateProvider",
    "ConversationStateAdapter",
    "InMemoryStateProvider",
    "MemoryProvider",
    "HistoryProvider",
    "SummarizingHistoryProvider",
    "InMemoryHistory",
    "Message",
]

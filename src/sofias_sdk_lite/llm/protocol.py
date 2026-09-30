"""The `LLMCallable` protocol and its request/response types.

This module defines the contract `LLMNode` drives. The SDK ships two
implementations (`sofias_sdk_lite.llm.GatewayLLM` for the Sofias gateway and
`sofias_sdk_lite.llm.OpenAICompatibleLLM` for any OpenAI-style endpoint) and a
factory (`sofias_sdk_lite.llm.create_llm`) that picks one from settings or the
environment. Implement the protocol yourself only when you need a client those
do not cover (a local model, a mocked LLM in tests, a different wire format).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from sofias_sdk_lite.llm.tokens import TokenUsage
from sofias_sdk_lite.llm.tools import ToolCall, ToolSpec

__all__ = [
    "LLMCallable",
    "LLMResponse",
    "LLMStreamEvent",
    "StreamableLLMCallable",
]


class LLMResponse(BaseModel):
    """Response from a single LLM invocation."""

    model_config = ConfigDict(frozen=True)

    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: TokenUsage | None = None
    raw_response: Any = None
    model_requested: str | None = None
    provider: str | None = None

    @property
    def is_final(self) -> bool:
        """Whether this response is a terminal answer (no tool calls to run)."""
        return not self.tool_calls

    @property
    def has_tool_calls(self) -> bool:
        """Whether the LLM requested one or more tool calls."""
        return bool(self.tool_calls)


class LLMStreamEvent(BaseModel):
    """A single streaming event from an LLM invocation."""

    model_config = ConfigDict(frozen=True)

    delta: str | None = None
    tool_call: ToolCall | None = None
    usage: TokenUsage | None = None
    model_requested: str | None = None
    provider: str | None = None


@runtime_checkable
class LLMCallable(Protocol):
    """Protocol defining how an LLM is invoked.

    Implement this against any LLM client (HTTP-based, local, mocked, etc.)
    to make it usable by the Sofias agent SDK.
    """

    async def invoke(
        self,
        prompt: str,
        tools: list[ToolSpec] | None = None,
        messages: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        response_format: dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> LLMResponse: ...


@runtime_checkable
class StreamableLLMCallable(LLMCallable, Protocol):
    """An `LLMCallable` that also supports streaming invocation."""

    # Deliberately ``def`` rather than ``async def``: implementations are async
    # *generators* (``async def`` with ``yield``), which return the iterator
    # directly, and callers write ``async for event in llm.stream_invoke(...)``.
    # An ``async def`` here would describe a coroutine returning an iterator, a
    # shape no real implementation has, and type-checkers would reject them all.
    def stream_invoke(
        self,
        prompt: str,
        tools: list[ToolSpec] | None = None,
        messages: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        response_format: dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> AsyncIterator[LLMStreamEvent]: ...

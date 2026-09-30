"""Streaming event types for the Agent SDK.

These events are yielded by Agent.stream_execute() and node.stream_execute()
to provide real-time visibility into agent execution.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "StreamEvent",
    "TextChunkEvent",
    "ToolCallStartEvent",
    "ToolCallResultEvent",
    "NodeStartEvent",
    "NodeCompleteEvent",
    "AgentCompleteEvent",
    "AggregationResponseEvent",
    "AgentErrorEvent",
]


class StreamEvent(BaseModel):
    """Base class for all streaming events.

    Subclasses override ``type`` with a ``Literal`` default so that each
    event carries a discriminator value. We intentionally omit the field
    here and let each concrete subclass declare it, which keeps type
    checkers happy while still allowing ``StreamEvent`` to be used as a
    generic base in type annotations.
    """

    model_config = ConfigDict(frozen=True)

    node_name: str | None = None
    timestamp: float = Field(default_factory=time.time)


class TextChunkEvent(StreamEvent):
    """A chunk of text content from the LLM."""

    type: Literal["text_chunk"] = "text_chunk"
    content: str


class ToolCallStartEvent(StreamEvent):
    """Emitted when a tool call begins."""

    type: Literal["tool_call_start"] = "tool_call_start"
    tool_name: str
    tool_input: dict[str, Any]
    tool_call_id: str | None = None


class ToolCallResultEvent(StreamEvent):
    """Emitted when a tool call completes."""

    type: Literal["tool_call_result"] = "tool_call_result"
    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None


class NodeStartEvent(StreamEvent):
    """Emitted when a node begins execution."""

    type: Literal["node_start"] = "node_start"
    node_type: str


class NodeCompleteEvent(StreamEvent):
    """Emitted when a node completes execution."""

    type: Literal["node_complete"] = "node_complete"
    node_type: str
    output: dict[str, Any]
    duration_ms: float


class AgentCompleteEvent(StreamEvent):
    """Emitted when the agent finishes execution successfully."""

    type: Literal["agent_complete"] = "agent_complete"
    content: dict[str, Any]
    execution_path: list[str]
    execution_time_ms: float
    usage: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Token usage grouped by (model_requested, provider).",
    )
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0


class AggregationResponseEvent(StreamEvent):
    """Emitted when the AggregatorNode receives a response for a pending correlation ID."""

    type: Literal["aggregation_response"] = "aggregation_response"
    correlation_id: str
    total_expected: int
    total_received: int
    is_resolved: bool


class AgentErrorEvent(StreamEvent):
    """Emitted when the agent encounters an error."""

    type: Literal["agent_error"] = "agent_error"
    error_type: str
    error_message: str
    execution_path: list[str]

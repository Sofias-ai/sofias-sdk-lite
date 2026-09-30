"""Typed execution context for agent graph execution.

This module defines ExecutionContext, the structured model that carries
execution identity, tracing metadata, and transition history. It is
accessible from any node via get_execution_context().
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from sofias_sdk_lite.llm import TokenUsage

__all__ = [
    "TokenAccumulator",
    "ExecutionContext",
    "get_execution_context",
    "set_execution_context",
    "usage_state_from_list",
]


@dataclass
class _UsageBucket:
    """Token counts for a single (model_requested, provider) pair."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0


class TokenAccumulator:
    """Mutable accumulator that sums tokens grouped by (model_requested, provider)."""

    def __init__(self) -> None:
        self._buckets: dict[tuple[str, str], _UsageBucket] = {}

    def _bucket(self, model_requested: str, provider: str) -> _UsageBucket:
        key = (model_requested, provider)
        if key not in self._buckets:
            self._buckets[key] = _UsageBucket()
        return self._buckets[key]

    # --- Scalar properties (backward compat: sum across all buckets) ---

    @property
    def prompt_tokens(self) -> int:
        return sum(b.prompt_tokens for b in self._buckets.values())

    @property
    def completion_tokens(self) -> int:
        return sum(b.completion_tokens for b in self._buckets.values())

    @property
    def cached_tokens(self) -> int:
        return sum(b.cached_tokens for b in self._buckets.values())

    @property
    def reasoning_tokens(self) -> int:
        return sum(b.reasoning_tokens for b in self._buckets.values())

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    # --- Mutators ---

    def add(
        self,
        usage: TokenUsage,
        model_requested: str = "",
        provider: str = "",
    ) -> None:
        """Sum tokens from an LLM response's TokenUsage, grouped by model/provider."""
        bucket = self._bucket(model_requested, provider)
        bucket.prompt_tokens += usage.prompt_tokens
        bucket.completion_tokens += usage.completion_tokens
        bucket.cached_tokens += usage.cached_tokens or 0
        bucket.reasoning_tokens += getattr(usage, "reasoning_tokens", 0) or 0

    def add_raw(
        self,
        prompt: int,
        completion: int,
        cached: int = 0,
        reasoning: int = 0,
        model_requested: str = "",
        provider: str = "",
    ) -> None:
        """Sum tokens from raw integer values (e.g. delegation extras)."""
        bucket = self._bucket(model_requested, provider)
        bucket.prompt_tokens += prompt
        bucket.completion_tokens += completion
        bucket.cached_tokens += cached
        bucket.reasoning_tokens += reasoning

    def merge(self, other: TokenAccumulator) -> None:
        """Combine another accumulator (e.g. fan-out branches)."""
        for (model, prov), other_bucket in other._buckets.items():
            bucket = self._bucket(model, prov)
            bucket.prompt_tokens += other_bucket.prompt_tokens
            bucket.completion_tokens += other_bucket.completion_tokens
            bucket.cached_tokens += other_bucket.cached_tokens
            bucket.reasoning_tokens += other_bucket.reasoning_tokens

    # --- Export ---

    def to_dict(self) -> dict[str, int]:
        """Export flat sum dict (backward compat)."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }

    def to_usage_list(self) -> list[dict[str, Any]]:
        """Export as list of dicts, one per (model_requested, provider) bucket."""
        result: list[dict[str, Any]] = []
        for (model, prov), bucket in self._buckets.items():
            result.append({
                "prompt_tokens": bucket.prompt_tokens,
                "completion_tokens": bucket.completion_tokens,
                "cached_tokens": bucket.cached_tokens,
                "reasoning_tokens": bucket.reasoning_tokens,
                "model_requested": model,
                "provider": prov,
            })
        return result


def _usage_entry(entry: Any, key: str, default: Any = 0) -> Any:
    """Read *key* off a usage entry that may be a dict or an object.

    ``AgentResponse.usage`` entries are typed as ``dict`` (built by
    ``TokenAccumulator.to_usage_list``), but callers are free to set the
    field directly with plain objects, so both shapes must work.
    """
    if isinstance(entry, dict):
        return cast(dict[str, Any], entry).get(key, default)
    return getattr(entry, key, default)


def _usage_entry_int(entry: Any, key: str) -> int:
    try:
        return int(_usage_entry(entry, key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def usage_state_from_list(usage: list[Any] | None) -> dict[str, Any] | None:
    """Collapse an ``AgentResponse.usage`` list into a single ``StreamFragment``
    ``state.usage`` object, or ``None`` when there is nothing to report.

    Every response workflow that publishes a terminal fragment should call
    this and merge the result into ``state``: it is the one place a billing
    consumer reads token usage from. Sums tokens across every LLM call in the
    turn (a turn can invoke several models: the main chat call plus e.g. an
    embedding call for RAG), and names the entry with the most completion
    tokens as ``model``, which picks the actual generation call over
    near-zero-completion side calls.
    """
    if not usage:
        return None

    primary = max(usage, key=lambda e: _usage_entry_int(e, "completion_tokens"))
    return {
        "prompt_tokens": sum(_usage_entry_int(e, "prompt_tokens") for e in usage),
        "completion_tokens": sum(_usage_entry_int(e, "completion_tokens") for e in usage),
        "model": _usage_entry(primary, "model_requested", ""),
    }


class ExecutionContext(BaseModel):
    """Structured execution context propagated via ContextVar.

    Created at the start of Agent.execute() and updated as the graph
    traverses nodes. Any node can read it via get_execution_context().

    Attributes:
        execution_id: Unique ID for this execution run (UUID4).
        agent_name: Name of the agent being executed.
        agent_version: Version of the agent being executed.
        trace_id: Distributed trace identifier from the incoming message.
        current_node: Name of the node currently being executed.
        transition_history: Ordered list of nodes visited so far.
        started_at: UTC timestamp when execution started.
        metadata: User-provided static context from AgentBuilder.with_context().
    """

    model_config = ConfigDict(validate_default=True, arbitrary_types_allowed=True)

    execution_id: str = Field(
        ...,
        description="Unique ID for this execution run (UUID4).",
    )

    agent_name: str = Field(
        ...,
        description="Name of the agent being executed.",
    )

    agent_version: str = Field(
        ...,
        description="Version of the agent being executed.",
    )

    trace_id: str | None = Field(
        default=None,
        description="Distributed trace identifier from the incoming message.",
    )

    current_node: str | None = Field(
        default=None,
        description="Name of the node currently being executed.",
    )

    transition_history: list[str] = Field(
        default_factory=list,
        description="Ordered list of nodes visited so far.",
    )

    started_at: datetime = Field(
        ...,
        description="UTC timestamp when execution started.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="User-provided static context from AgentBuilder.with_context().",
    )

    token_accumulator: TokenAccumulator = Field(
        default_factory=TokenAccumulator,
        exclude=True,
    )


# ContextVar for typed execution context shared across all nodes
_execution_context: ContextVar[ExecutionContext | None] = ContextVar(
    "execution_context", default=None
)


def get_execution_context() -> ExecutionContext | None:
    """Get the current execution context.

    Returns:
        The ExecutionContext for the current run, or None if not
        inside an Agent.execute() call.
    """
    return _execution_context.get()


def set_execution_context(ctx: ExecutionContext | None) -> None:
    """Set (or clear) the current execution context."""
    _execution_context.set(ctx)

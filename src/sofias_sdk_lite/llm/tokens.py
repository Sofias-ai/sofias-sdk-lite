"""Token usage accounting for LLM invocations."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["TokenUsage"]


class TokenUsage(BaseModel):
    """Token usage metrics reported by an LLM invocation."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None

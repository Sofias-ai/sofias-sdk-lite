"""Minimal tool-schema types referenced by the LLM protocol.

These describe the shape of a tool passed into an LLM invocation (`ToolSpec`)
and a tool invocation requested by the LLM in return (`ToolCall`). This is
not a tool execution framework: running a tool, retries, timeouts, and
result handling are a separate concern left to the caller (or a future
`sofias_sdk_lite.tools` module).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = ["ToolCall", "ToolSpec"]


class ToolSpec(BaseModel):
    """Specification for a tool made available to the LLM."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters_schema: dict[str, Any]


class ToolCall(BaseModel):
    """A tool call requested by the LLM."""

    model_config = ConfigDict(frozen=True)

    name: str
    input: dict[str, Any]
    id: str | None = None

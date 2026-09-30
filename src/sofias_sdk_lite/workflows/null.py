"""A no-op response workflow for local development and tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sofias_sdk_lite.agent.execution_context import ExecutionContext
    from sofias_sdk_lite.contracts.agent_contracts import AgentResponse

__all__ = ["NullWorkflow"]


class NullWorkflow:
    """A `ResponseWorkflow` that records calls instead of delivering anywhere.

    Useful for testing agents and runners without a real transport.
    """

    def __init__(self) -> None:
        self.responses: list["AgentResponse"] = []
        self.errors: list[str] = []
        self.closed = False

    async def send_response(self, response: "AgentResponse", context: "ExecutionContext") -> None:
        """Record the response instead of delivering it."""
        self.responses.append(response)

    async def send_error(self, error: str) -> None:
        """Record the error instead of delivering it."""
        self.errors.append(error)

    async def close(self) -> None:
        """Mark this workflow as closed."""
        self.closed = True

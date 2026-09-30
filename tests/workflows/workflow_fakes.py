"""Fakes shared by the response-workflow tests."""

from __future__ import annotations

from typing import Any

from sofias_sdk_lite import AgentResponse, ExecutionContext
from sofias_sdk_lite.rabbitmq.types import PublishOptions


class FakePublisher:
    """Records every ``publish_stream`` call instead of talking to a broker."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def publish_stream(
        self, payload: dict[str, Any], stream_name: str, options: PublishOptions | None = None
    ) -> None:
        self.calls.append({"payload": payload, "stream": stream_name, "options": options})

    async def close(self) -> None:
        self.closed = True

    @property
    def fragments(self) -> list[dict[str, Any]]:
        return [c["payload"] for c in self.calls]


def make_response(
    content: dict[str, Any] | None = None, usage: list[dict[str, Any]] | None = None
) -> AgentResponse:
    return AgentResponse(
        content=content or {},
        agent_name="agent",
        agent_version="0.0.1",
        usage=usage or [],
    )


def make_context() -> ExecutionContext:
    from datetime import datetime, timezone

    return ExecutionContext(
        execution_id="exec-1",
        agent_name="agent",
        agent_version="0.0.1",
        started_at=datetime.now(timezone.utc),
    )

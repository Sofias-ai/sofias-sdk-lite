"""Shared helpers for LLM adapter tests: an in-memory OpenAI-compatible server."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from sofias_sdk_lite.llm import openai_compatible

Handler = Callable[[httpx.Request, int], httpx.Response]


class FakeGateway:
    """Records requests and answers with scripted responses.

    ``script`` is a list of `httpx.Response` (or callables producing one) used
    in order; the last one repeats once exhausted.
    """

    def __init__(self, *script: httpx.Response | Handler) -> None:
        self.script = list(script)
        self.requests: list[httpx.Request] = []

    @property
    def payloads(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]

    def handler(self, request: httpx.Request) -> httpx.Response:
        index = len(self.requests)
        self.requests.append(request)
        step = self.script[min(index, len(self.script) - 1)]
        return step(request, index) if callable(step) else step

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def completion(
    content: str | None = "hello",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
    extra_fields: dict[str, Any] | None = None,
    status: int = 200,
) -> httpx.Response:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    body: dict[str, Any] = {"choices": [{"index": 0, "message": message}]}
    if usage is not None:
        body["usage"] = usage
    if extra_fields is not None:
        body["extra_fields"] = extra_fields
    return httpx.Response(status, json=body)


def sse(*chunks: dict[str, Any] | str, done: bool = True) -> httpx.Response:
    lines = [c if isinstance(c, str) else "data: " + json.dumps(c) for c in chunks]
    if done:
        lines.append("data: [DONE]")
    return httpx.Response(200, content="\n\n".join(lines).encode(), headers={"content-type": "text/event-stream"})


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openai_compatible, "_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setattr(openai_compatible, "_BACKOFF_MAX_SECONDS", 0.0)

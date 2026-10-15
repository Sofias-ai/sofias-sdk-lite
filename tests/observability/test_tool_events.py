"""One event line per tool call, with names and timings only."""

from __future__ import annotations

import json

import pytest

from sofias_sdk_lite.nodes.llm_node import LLMNode
from sofias_sdk_lite.observability.tool_events import EVENT_PREFIX, emit_tool_call
from tests.nodes.llm_fakes import FakeLLM, FakeRegistry, final, make_node, spec, tool_call


def events(output: str) -> list[dict]:
    return [
        json.loads(line.split(" ", 1)[1])
        for line in output.splitlines()
        if line.startswith(EVENT_PREFIX)
    ]


def test_splits_the_connector_from_the_tool(capsys: pytest.CaptureFixture[str]) -> None:
    emit_tool_call("notion__search", ok=True, duration_ms=412.4, node_name="answerer")

    (event,) = events(capsys.readouterr().out)
    assert event["tool"] == "search" and event["server"] == "notion"
    assert event["ok"] is True and event["duration_ms"] == 412 and event["node"] == "answerer"


def test_own_tools_have_no_server(capsys: pytest.CaptureFixture[str]) -> None:
    emit_tool_call("calculator", ok=False, duration_ms=3, error_type="failed")

    (event,) = events(capsys.readouterr().out)
    assert event["server"] is None and event["error_type"] == "failed"


def test_can_be_silenced(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOFIAS_EVENTS", "off")

    emit_tool_call("calculator", ok=True, duration_ms=1)

    assert events(capsys.readouterr().out) == []


async def test_the_llm_node_leaves_one_line_per_call_without_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    llm = FakeLLM(tool_call("read", call_id="a", path="/secret/user-file.csv"), final())
    node: LLMNode = make_node(llm, registry=FakeRegistry([spec()]))

    await node._run({"query": "hi"})

    out = capsys.readouterr().out
    (event,) = events(out)
    assert event["tool"] == "read"
    assert "user-file.csv" not in out

"""Verifies examples/02_support_agent.py builds and executes correctly.

Runs entirely in-process (NullWorkflow, no broker) per docs/guides/testing-your-agent.md.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sofias_sdk_lite import AgentMessage, NullWorkflow

_EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "examples" / "02_support_agent.py"
_spec = importlib.util.spec_from_file_location("support_agent_example", _EXAMPLE_PATH)
support_agent_example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(support_agent_example)

SupportAgentRunner = support_agent_example.SupportAgentRunner
SupportInput = support_agent_example.SupportInput
build_support_agent = support_agent_example.build_support_agent


@pytest.mark.asyncio
async def test_normal_message_routes_to_respond():
    agent = build_support_agent()
    response = await agent.execute(AgentMessage(content=SupportInput(message="How do I reset my password?")))

    assert response.status.value == "success"
    assert response.content["escalated"] is False
    assert "received" in response.content["reply"]
    assert response.execution_path == ["classify", "respond"]


@pytest.mark.asyncio
async def test_urgent_message_routes_to_escalate():
    agent = build_support_agent()
    response = await agent.execute(AgentMessage(content=SupportInput(message="Production is down, urgent!")))

    assert response.status.value == "success"
    assert response.content["escalated"] is True
    assert response.execution_path == ["classify", "escalate"]


@pytest.mark.asyncio
async def test_records_response_via_null_workflow():
    workflow = NullWorkflow()
    agent = build_support_agent(workflow)
    await agent.execute(AgentMessage(content=SupportInput(message="outage on checkout")))

    assert len(workflow.responses) == 1
    assert workflow.responses[0].content["escalated"] is True


def test_runner_build_agent_and_prepare_input_are_wired_correctly():
    runner = SupportAgentRunner.__new__(SupportAgentRunner)  # no RabbitMQ connection needed
    settings = SupportAgentRunner.settings_class()
    agent = runner.build_agent(settings, workflow=None)

    assert agent.name == "support_agent"
    assert agent.entry_node == "classify"

    class FakeTask:
        content = "hello"
        conversation_id = "conv-1"

    message = runner.prepare_input(FakeTask(), history=[], role="user")
    assert message.content.message == "hello"
    assert message.conversation_id == "conv-1"

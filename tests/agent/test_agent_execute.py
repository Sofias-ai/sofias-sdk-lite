"""End-to-end test of the core agent graph: build, execute, fan-out."""

from __future__ import annotations

import pytest

from sofias_sdk_lite import (
    AgentBuilder,
    AgentMessage,
    BaseAgentSettings,
    InputContract,
    NodeContract,
    OutputContract,
)


class GreetInput(InputContract):
    name: str


class GreetOutput(OutputContract):
    greeting: str


class HelloSettings(BaseAgentSettings):
    pass


def _greet(data: dict, context: dict | None = None) -> dict:
    return {"greeting": f"Hello, {data['name']}!"}


def _build_hello_agent():
    contract = NodeContract(input_schema=GreetInput, output_schema=GreetOutput)
    return (
        AgentBuilder("hello_agent", version="0.1.0")
        .with_settings_class(HelloSettings)
        .with_contract(input_schema=GreetInput, output_schema=GreetOutput)
        .add_function_node("greeter", contract, process_fn=_greet)
        .set_entry_node("greeter")
        .set_terminal("greeter")
        .build()
    )


@pytest.mark.asyncio
async def test_agent_executes_single_function_node():
    agent = _build_hello_agent()
    response = await agent.execute(AgentMessage(content=GreetInput(name="World")))

    assert response.status.value == "success"
    assert response.content == {"greeting": "Hello, World!"}
    assert response.execution_path == ["greeter"]


@pytest.mark.asyncio
async def test_agent_build_fails_without_settings_class():
    from sofias_sdk_lite.agent.agent_builder import AgentBuildError

    contract = NodeContract(input_schema=GreetInput, output_schema=GreetOutput)
    builder = (
        AgentBuilder("no_settings_agent", version="0.1.0")
        .with_contract(input_schema=GreetInput, output_schema=GreetOutput)
        .add_function_node("greeter", contract, process_fn=_greet)
        .set_entry_node("greeter")
        .set_terminal("greeter")
    )

    with pytest.raises(AgentBuildError):
        builder.build()

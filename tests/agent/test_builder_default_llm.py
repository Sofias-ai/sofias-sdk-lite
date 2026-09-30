"""`AgentBuilder.build()` resolves the LLM when `with_llm()` is not called."""

from __future__ import annotations

import pytest

from sofias_sdk_lite import (
    AgentBuildError,
    AgentBuilder,
    AgentMessage,
    BaseAgentSettings,
    LLMResponse,
    NodeContract,
)
from sofias_sdk_lite.llm import ENV_LLM_MODEL, GatewayLLM, default_llm
from sofias_sdk_lite.nodes.llm_node import LLMNode
from tests.nodes.llm_fakes import CONTRACT, FakeLLM, QueryInput, ResultOutput, make_config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (ENV_LLM_MODEL, "MODEL_NAME", "SOFIAS_LLM_API_KEY", "SOFIAS_LLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


def builder() -> AgentBuilder:
    return (
        AgentBuilder("llm_agent", version="0.1.0")
        .with_settings_class(BaseAgentSettings)
        .with_contract(input_schema=QueryInput, output_schema=ResultOutput)
        .add_llm_node("worker", make_config(), CONTRACT)
        .set_entry_node("worker")
        .set_terminal("worker")
    )


def worker_llm(agent) -> object:
    node = agent._nodes["worker"]
    assert isinstance(node, LLMNode)
    return node._llm


def test_build_fails_with_actionable_message_when_nothing_is_configured() -> None:
    with pytest.raises(AgentBuildError) as exc:
        builder().build()
    assert ENV_LLM_MODEL in str(exc.value)
    assert "with_llm()" in str(exc.value)


def test_explicit_with_llm_still_wins() -> None:
    fake = FakeLLM()
    with default_llm(FakeLLM()):
        agent = builder().with_llm(fake).build()
    assert worker_llm(agent) is fake


async def test_default_llm_scope_feeds_llm_nodes() -> None:
    fake = FakeLLM(LLMResponse(content='{"result": "from-default"}'))
    with default_llm(fake):
        agent = builder().build()
    assert worker_llm(agent) is fake

    response = await agent.execute(AgentMessage(content=QueryInput(query="q")))
    assert response.content == {"result": "from-default"}


def test_environment_builds_a_gateway_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_LLM_MODEL, "default")
    monkeypatch.setenv("SOFIAS_LLM_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("SOFIAS_LLM_API_KEY", "k")
    agent = builder().build()
    llm = worker_llm(agent)
    assert isinstance(llm, GatewayLLM)
    assert llm.model == "default"


def test_function_only_agents_never_touch_the_llm() -> None:
    contract = NodeContract(input_schema=QueryInput, output_schema=ResultOutput)
    agent = (
        AgentBuilder("fn_agent")
        .with_settings_class(BaseAgentSettings)
        .with_contract(input_schema=QueryInput, output_schema=ResultOutput)
        .add_function_node("f", contract, process_fn=lambda d, c: {"result": d["query"]})
        .set_entry_node("f")
        .set_terminal("f")
        .build()
    )
    assert "f" in agent._nodes


def test_planner_node_uses_resolved_default_llm() -> None:
    fake = FakeLLM()
    contract = NodeContract(input_schema=QueryInput, output_schema=ResultOutput)
    with default_llm(fake):
        agent = (
            AgentBuilder("planner_agent")
            .with_settings_class(BaseAgentSettings)
            .with_contract(input_schema=QueryInput, output_schema=ResultOutput)
            .add_function_node("f", contract, process_fn=lambda d, c: {"result": d["query"]})
            .add_planner_node("plan", contract)
            .set_entry_node("plan")
            .set_terminal("plan")
            .set_plan_only("f")
            .build()
        )
    assert agent._nodes["plan"]._llm is fake

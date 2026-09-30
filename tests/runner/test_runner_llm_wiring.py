"""`AgentRunner` builds the LLM from per-task settings and hands it to `build_agent`."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from sofias_sdk_lite import (
    AgentResponse,
    AgentRunner,
    BaseAgentSettings,
    RabbitMQConfig,
    RunnerConfig,
)
from sofias_sdk_lite.llm import ENV_LLM_MODEL, GatewayLLM, get_default_llm
from sofias_sdk_lite.rabbitmq.types import MessageContext
from tests.nodes.llm_fakes import FakeLLM


class NullWorkflow:
    async def send_response(self, response: Any, context: Any) -> None: ...

    async def send_error(self, error: str) -> None: ...

    async def close(self) -> None: ...


class FakeAgent:
    async def execute(self, message: Any) -> AgentResponse:
        return AgentResponse(content={"answer": "hi"}, agent_name="a", agent_version="1")


class Runner(AgentRunner):
    settings_class = BaseAgentSettings

    def __init__(self) -> None:
        super().__init__(RunnerConfig(queue="q", agent_name="agent", rabbitmq=RabbitMQConfig(host="localhost")))
        self.seen_llms: list[Any] = []
        self.seen_settings: list[BaseAgentSettings] = []

    def create_response_workflow(self, client, task, *, traceparent=None):
        return NullWorkflow()

    def build_agent(self, settings, workflow):
        self.seen_settings.append(settings)
        self.seen_llms.append(get_default_llm())
        return FakeAgent()

    def prepare_input(self, task, history, role):
        return MagicMock()


def ctx(agent_configuration: dict[str, Any] | None = None, conversation: str = "conv-1") -> MessageContext:
    return MessageContext(
        payload={
            "content": "hello",
            "agent": "agent",
            "tenant_identifier": "t",
            "conversation_id": conversation,
            "message_id": 1,
            "agent_configuration": agent_configuration or {},
        },
        headers={},
        traceparent=None,
        routing_key="q",
        exchange="",
        delivery_tag=1,
        correlation_id=None,
        reply_to=None,
        timestamp=None,
        _message=MagicMock(),
    )


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (ENV_LLM_MODEL, "MODEL_NAME"):
        monkeypatch.delenv(var, raising=False)


GATEWAY = {"model_name": "default", "router_url": "http://gw:8080/v1", "router_api_key": "k"}


async def test_settings_gateway_fields_become_the_default_llm() -> None:
    runner = Runner()
    await runner._handle_message(ctx(GATEWAY), client=MagicMock())

    llm = runner.seen_llms[0]
    assert isinstance(llm, GatewayLLM)
    assert llm.model == "default"
    assert str(llm._client.base_url) == "http://gw:8080/v1/"
    assert runner.seen_settings[0].router_api_key.get_secret_value() == "k"
    assert get_default_llm() is None  # scope closed after build


async def test_client_is_reused_across_tasks_with_the_same_gateway() -> None:
    runner = Runner()
    await runner._handle_message(ctx(GATEWAY), client=MagicMock())
    await runner._handle_message(ctx(GATEWAY, conversation="conv-2"), client=MagicMock())
    await runner._handle_message(ctx({**GATEWAY, "model_name": "other"}), client=MagicMock())

    assert runner.seen_llms[0] is runner.seen_llms[1]
    assert runner.seen_llms[2] is not runner.seen_llms[0]
    assert runner.seen_llms[2].model == "other"


async def test_no_model_in_settings_means_no_default_from_runner() -> None:
    runner = Runner()
    await runner._handle_message(ctx({}), client=MagicMock())
    assert runner.seen_llms == [None]


async def test_create_llm_hook_can_be_overridden() -> None:
    fake = FakeLLM()

    class FakeLLMRunner(Runner):
        def create_llm(self, settings):
            return fake

    runner = FakeLLMRunner()
    await runner._handle_message(ctx(GATEWAY), client=MagicMock())
    assert runner.seen_llms == [fake]

"""``AgentRunner`` forwards the inbound traceparent to the response workflow and
reports failures on the span and to the user."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from sofias_sdk_lite import (
    AgentResponse,
    AgentRunner,
    BaseAgentSettings,
    RabbitMQConfig,
    RunnerConfig,
)
from sofias_sdk_lite.rabbitmq.types import MessageContext
from sofias_sdk_lite.state.history import Message

TP = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


class RecordingWorkflow:
    def __init__(self, traceparent: str | None) -> None:
        self.traceparent = traceparent
        self.errors: list[str] = []
        self.closed = False

    async def send_response(self, response: Any, context: Any) -> None:
        return None

    async def send_error(self, error: str) -> None:
        self.errors.append(error)

    async def close(self) -> None:
        self.closed = True


class FakeAgent:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    async def execute(self, message: Any) -> AgentResponse:
        if self.fail:
            raise RuntimeError("graph exploded")
        return AgentResponse(content={"answer": "hi"}, agent_name="a", agent_version="1")


class Runner(AgentRunner):
    settings_class = BaseAgentSettings

    def __init__(self, *, fail: bool = False) -> None:
        super().__init__(
            RunnerConfig(queue="q", agent_name="agent", rabbitmq=RabbitMQConfig(host="localhost"))
        )
        self.fail = fail
        self.workflows: list[RecordingWorkflow] = []

    def create_response_workflow(self, client, task, *, traceparent=None):
        wf = RecordingWorkflow(traceparent)
        self.workflows.append(wf)
        return wf

    def build_agent(self, settings, workflow):
        return FakeAgent(fail=self.fail)

    def prepare_input(self, task, history, role):
        return MagicMock()


def _ctx(traceparent: str | None) -> MessageContext:
    return MessageContext(
        payload={
            "content": "hello",
            "agent": "agent",
            "tenant_identifier": "t",
            "conversation_id": "conv-1",
            "message_id": 7,
            "agent_configuration": {},
        },
        headers={},
        traceparent=traceparent,
        routing_key="q",
        exchange="",
        delivery_tag=1,
        correlation_id=None,
        reply_to=None,
        timestamp=None,
        _message=MagicMock(),
    )


class TestTraceparentForwarding:
    async def test_inbound_traceparent_reaches_the_workflow(self) -> None:
        runner = Runner()
        await runner._handle_message(_ctx(TP), client=MagicMock())
        assert runner.workflows[0].traceparent == TP
        assert runner.workflows[0].closed is True

    async def test_missing_traceparent_forwards_none(self) -> None:
        runner = Runner()
        await runner._handle_message(_ctx(None), client=MagicMock())
        assert runner.workflows[0].traceparent is None

    async def test_default_workflow_receives_traceparent(self) -> None:
        class DefaultRunner(AgentRunner):
            def build_agent(self, settings, workflow):
                return FakeAgent()

            def prepare_input(self, task, history, role):
                return MagicMock()

        runner = DefaultRunner(
            RunnerConfig(queue="q", agent_name="agent", rabbitmq=RabbitMQConfig(host="localhost"))
        )
        task = MagicMock(conversation_id="c", message_id=1)
        with patch("sofias_sdk_lite.runner.runner.ChatResponseWorkflow") as wf_cls:
            runner.create_response_workflow(MagicMock(), task, traceparent=TP)
        assert wf_cls.call_args.kwargs["traceparent"] == TP
        assert wf_cls.call_args.kwargs["message_id"] == 1

    async def test_history_records_both_turns(self) -> None:
        runner = Runner()
        await runner._handle_message(_ctx(TP), client=MagicMock())
        history = await runner._history.get("conv-1")
        assert [m.role for m in history] == ["user", "assistant"]
        assert history == [
            Message(role="user", content="hello", timestamp=history[0].timestamp),
            Message(role="assistant", content="hi", timestamp=history[1].timestamp),
        ]


class TestFailures:
    async def test_exception_is_recorded_on_span_and_reported_to_user(self) -> None:
        runner = Runner(fail=True)
        span = MagicMock()

        with patch("sofias_sdk_lite.runner.runner.consumer_span") as cs:
            cs.return_value.__enter__.return_value = span
            await runner._handle_message(_ctx(TP), client=MagicMock())

        span.record_error.assert_called_once()
        assert isinstance(span.record_error.call_args.args[0], RuntimeError)
        assert runner.workflows[0].errors == [
            "An internal error occurred while processing this message."
        ]
        assert runner.workflows[0].closed is True

    async def test_span_carries_agent_and_message_attributes(self) -> None:
        runner = Runner()
        with patch("sofias_sdk_lite.runner.runner.consumer_span") as cs:
            cs.return_value.__enter__.return_value = MagicMock()
            await runner._handle_message(_ctx(TP), client=MagicMock())

        kwargs = cs.call_args.kwargs
        assert cs.call_args.args == ("agent.handle",)
        assert kwargs["traceparent"] == TP
        assert kwargs["attributes"]["agent.name"] == "agent"
        assert kwargs["attributes"]["conversation.id"] == "conv-1"
        assert kwargs["attributes"]["agent.traceparent.incoming.present"] is True

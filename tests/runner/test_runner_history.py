"""History windowing and background summarization in ``AgentRunner``."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

from sofias_sdk_lite import (
    AgentResponse,
    AgentRunner,
    BaseAgentSettings,
    InMemoryHistory,
    RabbitMQConfig,
    RunnerConfig,
    SummarizingHistoryProvider,
)
from sofias_sdk_lite.rabbitmq.types import MessageContext
from sofias_sdk_lite.state.history import Message


class NullWorkflow:
    async def send_response(self, response: Any, context: Any) -> None: ...

    async def send_error(self, error: str) -> None: ...

    async def close(self) -> None: ...


class FakeAgent:
    async def execute(self, message: Any) -> AgentResponse:
        return AgentResponse(content={"answer": "hi"}, agent_name="a", agent_version="1")


class SummarizingHistory(InMemoryHistory):
    def __init__(self, *, fail: bool = False, delay: float = 0.0) -> None:
        super().__init__()
        self.fail = fail
        self.delay = delay
        self.summarized: list[str] = []

    async def maybe_summarize(self, conversation_id: str) -> None:
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("summarizer down")
        self.summarized.append(conversation_id)


class Runner(AgentRunner):
    settings_class = BaseAgentSettings

    def __init__(self, history=None) -> None:
        super().__init__(
            RunnerConfig(queue="q", agent_name="agent", rabbitmq=RabbitMQConfig(host="localhost")),
            history=history,
        )
        self.seen_history: list[Message] = []

    def create_response_workflow(self, client, task, *, traceparent=None):
        return NullWorkflow()

    def build_agent(self, settings, workflow):
        return FakeAgent()

    def prepare_input(self, task, history, role):
        self.seen_history = list(history)
        return MagicMock()


def _ctx() -> MessageContext:
    return MessageContext(
        payload={
            "content": "hello",
            "agent": "agent",
            "tenant_identifier": "t",
            "conversation_id": "conv-1",
            "message_id": 1,
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


def _msgs(n: int, size: int = 1) -> list[Message]:
    return [Message(role="user", content=f"{i}:" + "x" * size) for i in range(n)]


class TestWindowing:
    def test_defaults_match_the_main_sdk(self) -> None:
        assert AgentRunner.max_messages == 100
        assert AgentRunner.max_context_tokens == 65_536
        assert AgentRunner.auto_summarize is True

    def test_caps_by_message_count_keeping_the_most_recent(self) -> None:
        runner = Runner()
        runner.max_messages = 3
        window = runner._window_history(_msgs(10))
        assert [m.content[:2] for m in window] == ["7:", "8:", "9:"]

    def test_caps_by_estimated_tokens_keeping_the_most_recent(self) -> None:
        runner = Runner()
        runner.max_context_tokens = 100  # ≈ 320 chars
        window = runner._window_history(_msgs(10, size=100))
        assert len(window) == 3
        assert window[-1].content.startswith("9:")

    def test_latest_turn_always_survives(self) -> None:
        runner = Runner()
        runner.max_context_tokens = 1
        window = runner._window_history(_msgs(3, size=5000))
        assert len(window) == 1
        assert window[0].content.startswith("2:")

    def test_short_history_is_untouched(self) -> None:
        runner = Runner()
        msgs = _msgs(5)
        assert runner._window_history(msgs) == msgs

    async def test_windowed_history_is_what_prepare_input_receives(self) -> None:
        history = InMemoryHistory()
        for m in _msgs(10):
            await history.append("conv-1", m)
        runner = Runner(history)
        runner.max_messages = 2

        await runner._handle_message(_ctx(), client=MagicMock())

        # two windowed turns + the incoming user turn
        assert [m.content[:2] for m in runner.seen_history] == ["8:", "9:", "he"]


class TestBackgroundSummary:
    async def test_summarize_runs_after_response_without_blocking(self) -> None:
        history = SummarizingHistory(delay=0.01)
        runner = Runner(history)

        await runner._handle_message(_ctx(), client=MagicMock())

        assert history.summarized == []  # not awaited inline
        assert len(runner._background_tasks) == 1
        await runner._drain_background_tasks()
        assert history.summarized == ["conv-1"]
        assert runner._background_tasks == set()

    async def test_summary_failure_is_logged_not_raised(self) -> None:
        history = SummarizingHistory(fail=True)
        runner = Runner(history)

        await runner._handle_message(_ctx(), client=MagicMock())
        await runner._drain_background_tasks()  # must not raise

        assert history.summarized == []

    async def test_plain_history_provider_schedules_nothing(self) -> None:
        runner = Runner(InMemoryHistory())
        await runner._handle_message(_ctx(), client=MagicMock())
        assert runner._background_tasks == set()

    async def test_auto_summarize_false_opts_out(self) -> None:
        history = SummarizingHistory()
        runner = Runner(history)
        runner.auto_summarize = False

        await runner._handle_message(_ctx(), client=MagicMock())
        await runner._drain_background_tasks()

        assert history.summarized == []

    def test_protocol_is_runtime_checkable(self) -> None:
        assert isinstance(SummarizingHistory(), SummarizingHistoryProvider)
        assert not isinstance(InMemoryHistory(), SummarizingHistoryProvider)

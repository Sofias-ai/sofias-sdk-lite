"""``ChatResponseWorkflow``: chunking, usage on the terminal fragment, empty-response
fallback and traceparent echo."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sofias_sdk_lite.workflows.chat import EMPTY_RESPONSE_FALLBACK, ChatResponseWorkflow
from tests.workflows.workflow_fakes import FakePublisher, make_context, make_response


@pytest.fixture
def publisher() -> FakePublisher:
    return FakePublisher()


def _workflow(publisher: FakePublisher, **kwargs) -> ChatResponseWorkflow:
    with patch("sofias_sdk_lite.workflows.chat.RabbitMQPublisher", return_value=publisher):
        return ChatResponseWorkflow(
            agent_name="agent",
            client=MagicMock(),
            stream_name="agent.responses",
            conversation_id="conv-1",
            message_id=42,
            **kwargs,
        )


class TestChunking:
    async def test_chunks_answer_and_marks_only_the_last_completed(self, publisher) -> None:
        wf = _workflow(publisher, chunk_size=100)
        await wf.send_response(make_response({"answer": "a" * 250}), make_context())

        fragments = publisher.fragments
        assert len(fragments) == 3
        assert [f["status"] for f in fragments] == ["streaming", "streaming", "completed"]
        assert fragments[-1]["is_last"] is True
        assert fragments[-1]["fragment_index"] == 2

    async def test_exact_multiple_of_chunk_size_has_no_trailing_empty_fragment(
        self, publisher
    ) -> None:
        wf = _workflow(publisher, chunk_size=5)
        await wf.send_response(make_response({"answer": "a" * 10}), make_context())
        assert len(publisher.fragments) == 2

    async def test_content_key_is_a_fallback_for_answer(self, publisher) -> None:
        wf = _workflow(publisher)
        await wf.send_response(make_response({"content": "via content"}), make_context())
        assert publisher.fragments[0]["content"] == "via content"


class TestEmptyResponseFallback:
    async def test_empty_content_substitutes_the_fallback(self, publisher) -> None:
        wf = _workflow(publisher)
        await wf.send_response(make_response({}), make_context())

        fragments = publisher.fragments
        assert "".join(f["content"] for f in fragments) == EMPTY_RESPONSE_FALLBACK
        assert fragments[-1]["status"] == "completed"

    async def test_dict_without_answer_or_content_substitutes_the_fallback(self, publisher) -> None:
        wf = _workflow(publisher)
        await wf.send_response(make_response({"other": "x"}), make_context())
        assert "".join(f["content"] for f in publisher.fragments) == EMPTY_RESPONSE_FALLBACK

    async def test_non_empty_answer_is_untouched(self, publisher) -> None:
        wf = _workflow(publisher)
        await wf.send_response(make_response({"answer": "real"}), make_context())
        assert publisher.fragments[0]["content"] == "real"


class TestUsageState:
    async def test_usage_lands_on_the_terminal_fragment_only(self, publisher) -> None:
        wf = _workflow(publisher, chunk_size=100)
        await wf.send_response(
            make_response(
                {"answer": "a" * 250},
                usage=[{"prompt_tokens": 120, "completion_tokens": 30, "model_requested": "m"}],
            ),
            make_context(),
        )

        fragments = publisher.fragments
        assert "usage" not in fragments[0]["state"]
        assert "usage" not in fragments[1]["state"]
        assert fragments[2]["state"]["usage"] == {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "model": "m",
        }

    async def test_no_usage_leaves_state_empty(self, publisher) -> None:
        wf = _workflow(publisher)
        await wf.send_response(make_response({"answer": "ok"}, usage=[]), make_context())
        assert publisher.fragments[-1]["state"] == {}


class TestTraceparent:
    async def test_inbound_traceparent_is_echoed_on_every_fragment(self, publisher) -> None:
        tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        wf = _workflow(publisher, chunk_size=2, traceparent=tp)
        await wf.send_response(make_response({"answer": "abcd"}), make_context())

        assert len(publisher.calls) == 2
        assert all(c["options"].traceparent == tp for c in publisher.calls)

    async def test_no_traceparent_forwards_none(self, publisher) -> None:
        wf = _workflow(publisher)
        await wf.send_response(make_response({"answer": "x"}), make_context())
        assert publisher.calls[0]["options"].traceparent is None

    async def test_active_context_wins_over_inbound(self, publisher) -> None:
        wf = _workflow(publisher, traceparent="inbound")
        with patch("sofias_sdk_lite.workflows.chat.active_traceparent", return_value="active-span"):
            await wf.send_response(make_response({"answer": "x"}), make_context())
        assert publisher.calls[0]["options"].traceparent == "active-span"


class TestErrorAndClose:
    async def test_error_fragment(self, publisher) -> None:
        wf = _workflow(publisher)
        await wf.send_error("boom")
        fragment = publisher.fragments[0]
        assert fragment["status"] == "error"
        assert fragment["content"] == "Error: boom"
        assert fragment["is_last"] is True

    async def test_close_closes_publisher(self, publisher) -> None:
        wf = _workflow(publisher)
        await wf.close()
        assert publisher.closed is True

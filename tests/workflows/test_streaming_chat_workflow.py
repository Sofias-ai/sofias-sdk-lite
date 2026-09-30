"""``BaseStreamingResponseWorkflow``: word-safe incremental flushing, authoritative
final text, usage, fallback, notices, and hook overrides."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sofias_sdk_lite.agent.streaming import TextChunkEvent
from sofias_sdk_lite.agent.streaming import ToolCallResultEvent
from sofias_sdk_lite.workflows.chat import EMPTY_RESPONSE_FALLBACK
from sofias_sdk_lite.workflows.streaming_chat import (
    BaseStreamingResponseWorkflow,
    _flush_cut,
    _split_for_stream,
)
from tests.workflows.workflow_fakes import FakePublisher, make_context, make_response


@pytest.fixture
def publisher() -> FakePublisher:
    return FakePublisher()


def _workflow(publisher: FakePublisher, **kwargs) -> BaseStreamingResponseWorkflow:
    return BaseStreamingResponseWorkflow(
        "agent",
        publisher=publisher,  # type: ignore[arg-type]
        stream_name="s",
        conversation_id="c",
        message_id="m",
        **kwargs,
    )


class TestConstruction:
    def test_requires_exactly_one_of_client_or_publisher(self, publisher) -> None:
        with pytest.raises(ValueError):
            BaseStreamingResponseWorkflow("a", stream_name="s", conversation_id="c", message_id="m")
        with pytest.raises(ValueError):
            BaseStreamingResponseWorkflow(
                "a",
                client=MagicMock(),
                publisher=publisher,  # type: ignore[arg-type]
                stream_name="s",
                conversation_id="c",
                message_id="m",
            )

    def test_injected_publisher_is_used_directly(self, publisher) -> None:
        wf = _workflow(publisher)
        assert wf._publisher is publisher
        assert wf._owns_publisher is False


class TestOnStreamEvent:
    async def test_flushes_at_word_boundary_not_mid_word(self, publisher) -> None:
        wf = _workflow(publisher, chunk_size=10)
        await wf.on_stream_event(TextChunkEvent(content="hello world foo"), make_context())

        assert publisher.fragments[0]["content"] == "hello "
        assert wf._buffer == "world foo"

    async def test_below_chunk_size_stays_buffered(self, publisher) -> None:
        wf = _workflow(publisher, chunk_size=10)
        await wf.on_stream_event(TextChunkEvent(content="short"), make_context())
        assert publisher.calls == []
        assert wf._buffer == "short"

    async def test_node_filter_drops_other_nodes(self, publisher) -> None:
        wf = _workflow(publisher, chunk_size=10, stream_node_names={"assistant"})
        await wf.on_stream_event(
            TextChunkEvent(content="x" * 20, node_name="other"), make_context()
        )
        assert publisher.calls == []
        assert wf._buffer == ""

    async def test_node_filter_keeps_matching_node(self, publisher) -> None:
        wf = _workflow(publisher, chunk_size=10, stream_node_names={"assistant"})
        await wf.on_stream_event(
            TextChunkEvent(content="x" * 15, node_name="assistant"), make_context()
        )
        assert len(publisher.calls) == 1

    async def test_non_text_event_goes_to_the_tool_hook(self, publisher) -> None:
        wf = _workflow(publisher)
        wf._on_tool_event = AsyncMock()  # type: ignore[method-assign]
        event = ToolCallResultEvent(tool_name="t", success=True, output={})
        ctx = make_context()

        await wf.on_stream_event(event, ctx)

        wf._on_tool_event.assert_awaited_once_with(event, ctx)
        assert publisher.calls == []

    async def test_should_buffer_text_false_drops_the_chunk(self, publisher) -> None:
        wf = _workflow(publisher, chunk_size=10)
        wf._should_buffer_text = MagicMock(return_value=False)  # type: ignore[method-assign]
        await wf.on_stream_event(TextChunkEvent(content="x" * 20), make_context())
        assert wf._buffer == ""
        assert publisher.calls == []

    async def test_safe_flush_len_holds_content_back(self, publisher) -> None:
        wf = _workflow(publisher, chunk_size=10)
        wf._safe_flush_len = MagicMock(return_value=0)  # type: ignore[method-assign]
        await wf.on_stream_event(TextChunkEvent(content="x" * 20), make_context())
        assert wf._buffer == "x" * 20
        assert publisher.calls == []

    async def test_first_flush_prefix_is_prepended_once(self, publisher) -> None:
        wf = _workflow(publisher, chunk_size=5, first_flush_prefix="[done] ")
        await wf.on_stream_event(TextChunkEvent(content="aaaaa bbbbb "), make_context())
        assert publisher.fragments[0]["content"].startswith("[done] ")
        assert not publisher.fragments[1]["content"].startswith("[done] ")

    async def test_stream_key_is_added_to_payload(self, publisher) -> None:
        wf = _workflow(publisher, chunk_size=5, stream_key="job-7")
        await wf.on_stream_event(TextChunkEvent(content="aaaaa "), make_context())
        assert publisher.fragments[0]["stream_key"] == "job-7"


class TestSendResponse:
    async def test_no_prior_streaming_chunks_the_whole_answer(self, publisher) -> None:
        wf = _workflow(publisher, chunk_size=100)
        await wf.send_response(make_response({"answer": "a" * 250}), make_context())

        fragments = publisher.fragments
        assert len(fragments) == 3
        assert fragments[-1]["status"] == "completed"
        assert fragments[-1]["state"]["final_content"] == "a" * 250

    async def test_already_streamed_replaces_with_authoritative_text(self, publisher) -> None:
        wf = _workflow(publisher, chunk_size=100)
        await wf.on_stream_event(TextChunkEvent(content="a" * 150), make_context())
        assert wf._fragment_index == 1

        await wf.send_response(make_response({"answer": "a" * 150}), make_context())

        last = publisher.fragments[-1]
        assert last["state"]["replace_content"] is True
        assert last["content"] == "a" * 150
        assert wf._buffer == ""

    async def test_usage_is_attached_to_terminal_fragment(self, publisher) -> None:
        wf = _workflow(publisher)
        await wf.send_response(
            make_response(
                {"answer": "ok"},
                usage=[{"prompt_tokens": 10, "completion_tokens": 5, "model_requested": "m"}],
            ),
            make_context(),
        )
        assert publisher.fragments[-1]["state"]["usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "model": "m",
        }

    async def test_extra_state_hook_is_merged_in(self, publisher) -> None:
        wf = _workflow(publisher)
        wf._extra_state = MagicMock(return_value={"attachments": ["f.pdf"]})  # type: ignore[method-assign]
        await wf.send_response(make_response({"answer": "ok"}), make_context())
        assert publisher.fragments[-1]["state"]["attachments"] == ["f.pdf"]

    async def test_final_text_hook_overrides_extraction(self, publisher) -> None:
        class Custom(BaseStreamingResponseWorkflow):
            def _final_text(self, response):
                return response.content.get("response", "")

        wf = Custom(
            "agent",
            publisher=publisher,
            stream_name="s",
            conversation_id="c",
            message_id="m",  # type: ignore[arg-type]
        )
        await wf.send_response(make_response({"response": "custom field"}), make_context())
        assert publisher.fragments[0]["content"] == "custom field"

    async def test_traceparent_is_echoed(self, publisher) -> None:
        wf = _workflow(publisher, traceparent="00-abc-def-01")
        await wf.send_response(make_response({"answer": "ok"}), make_context())
        assert publisher.calls[0]["options"].traceparent == "00-abc-def-01"


class TestEmptyResponseFallback:
    async def test_empty_final_text_substitutes_the_fallback(self, publisher) -> None:
        wf = _workflow(publisher)
        await wf.send_response(make_response({"answer": ""}), make_context())

        fragments = publisher.fragments
        content = "".join(f["content"] for f in fragments)
        assert content == EMPTY_RESPONSE_FALLBACK
        assert fragments[-1]["state"]["final_content"] == content

    async def test_hook_can_override_the_message(self, publisher) -> None:
        class Custom(BaseStreamingResponseWorkflow):
            def _empty_response_fallback(self, response):
                return "mensaje personalizado"

        wf = Custom(
            "agent",
            publisher=publisher,
            stream_name="s",
            conversation_id="c",
            message_id="m",  # type: ignore[arg-type]
        )
        await wf.send_response(make_response({"answer": ""}), make_context())
        assert publisher.fragments[0]["content"] == "mensaje personalizado"

    async def test_hook_returning_none_opts_out(self, publisher) -> None:
        class NoFallback(BaseStreamingResponseWorkflow):
            def _empty_response_fallback(self, response):
                return None

        wf = NoFallback(
            "agent",
            publisher=publisher,
            stream_name="s",
            conversation_id="c",
            message_id="m",  # type: ignore[arg-type]
        )
        await wf.send_response(make_response({"answer": ""}), make_context())
        assert publisher.fragments[0]["content"] == ""

    async def test_already_streamed_empty_final_is_not_substituted(self, publisher) -> None:
        wf = _workflow(publisher, chunk_size=100)
        await wf.on_stream_event(TextChunkEvent(content="a" * 150), make_context())

        await wf.send_response(make_response({"answer": ""}), make_context())

        assert publisher.fragments[-1]["content"] == ""


class TestSendErrorAndNotice:
    async def test_error_before_streaming_appends(self, publisher) -> None:
        wf = _workflow(publisher, chunk_size=100)
        await wf.send_error("boom")
        fragment = publisher.fragments[0]
        assert fragment["status"] == "error"
        assert "Error: boom" in fragment["content"]
        assert fragment["state"] == {}

    async def test_error_after_streaming_replaces(self, publisher) -> None:
        wf = _workflow(publisher, chunk_size=100)
        await wf.on_stream_event(TextChunkEvent(content="a" * 150), make_context())
        await wf.send_error("boom")
        assert publisher.fragments[-1]["state"] == {"replace_content": True}

    async def test_notice_has_no_error_prefix_and_completes(self, publisher) -> None:
        wf = _workflow(publisher)
        await wf.send_notice("too big, try narrowing it")
        fragment = publisher.fragments[0]
        assert fragment["content"] == "too big, try narrowing it"
        assert fragment["status"] == "completed"


class TestClose:
    async def test_owned_publisher_is_closed(self, publisher) -> None:
        with patch(
            "sofias_sdk_lite.workflows.streaming_chat.RabbitMQPublisher", return_value=publisher
        ):
            wf = BaseStreamingResponseWorkflow(
                "agent", client=MagicMock(), stream_name="s", conversation_id="c", message_id="m"
            )
        await wf.close()
        assert publisher.closed is True

    async def test_injected_publisher_is_not_closed(self, publisher) -> None:
        wf = _workflow(publisher)
        await wf.close()
        assert publisher.closed is False


class TestHelpers:
    def test_flush_cut_prefers_last_separator_in_window(self) -> None:
        assert _flush_cut("hello world foo", 10) == 6

    def test_flush_cut_falls_back_to_limit_without_separator(self) -> None:
        assert _flush_cut("abcdefghij", 5) == 5

    def test_split_reconstructs_the_original_text(self) -> None:
        text = "word " * 40
        fragments = _split_for_stream(text, 20)
        assert "".join(fragments) == text
        assert all(len(f) <= 20 for f in fragments)

    def test_split_empty_text_yields_single_empty_fragment(self) -> None:
        assert _split_for_stream("", 20) == [""]

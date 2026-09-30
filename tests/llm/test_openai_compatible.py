"""`OpenAICompatibleLLM`: request shape, response parsing, retries, streaming."""

from __future__ import annotations

import httpx
import pytest

from sofias_sdk_lite.errors import EmptyLLMResponseError, LLMRequestError
from sofias_sdk_lite.llm import (
    LLMCallable,
    OpenAICompatibleLLM,
    StreamableLLMCallable,
    ToolSpec,
)
from tests.llm.conftest import FakeGateway, completion, sse

TOOL = ToolSpec(name="lookup", description="Look something up", parameters_schema={"type": "object"})


def make_llm(gateway: FakeGateway, **kwargs) -> OpenAICompatibleLLM:
    defaults = dict(api_key="sk-test", base_url="https://gw.example/v1", model="m1", max_retries=2)
    defaults.update(kwargs)
    return OpenAICompatibleLLM(transport=gateway.transport, **defaults)


class TestProtocolConformance:
    def test_satisfies_both_protocols_at_runtime(self) -> None:
        llm = make_llm(FakeGateway(completion()))
        assert isinstance(llm, LLMCallable)
        assert isinstance(llm, StreamableLLMCallable)

    def test_rejects_empty_model_or_base_url(self) -> None:
        with pytest.raises(ValueError):
            OpenAICompatibleLLM(api_key="k", base_url="https://x", model="")
        with pytest.raises(ValueError):
            OpenAICompatibleLLM(api_key="k", base_url="", model="m")


class TestRequestShape:
    async def test_posts_chat_completion_with_auth_and_messages(self) -> None:
        gw = FakeGateway(completion("ok"))
        llm = make_llm(gw, temperature=0.2, max_tokens=99)

        await llm.invoke("hi", system_prompt="be terse", messages=[{"role": "assistant", "content": "prev"}])

        request = gw.requests[0]
        assert request.method == "POST"
        assert str(request.url) == "https://gw.example/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer sk-test"
        payload = gw.payloads[0]
        assert payload["model"] == "m1"
        assert payload["temperature"] == 0.2
        assert payload["max_tokens"] == 99
        assert payload["messages"] == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "prev"},
        ]
        assert "stream" not in payload

    async def test_empty_api_key_sends_no_authorization_header(self) -> None:
        gw = FakeGateway(completion())
        await make_llm(gw, api_key="").invoke("hi")
        assert "Authorization" not in gw.requests[0].headers

    async def test_tools_response_format_and_parallel_flag(self) -> None:
        gw = FakeGateway(completion())
        llm = make_llm(gw)
        fmt = {"type": "json_schema", "json_schema": {"name": "out", "schema": {"type": "object"}}}

        await llm.invoke("hi", tools=[TOOL], response_format=fmt, parallel_tool_calls=False)

        payload = gw.payloads[0]
        assert payload["tools"] == [
            {
                "type": "function",
                "function": {"name": "lookup", "description": "Look something up", "parameters": {"type": "object"}},
            }
        ]
        assert payload["parallel_tool_calls"] is False
        assert payload["response_format"] == fmt

    async def test_compact_assistant_tool_calls_are_expanded(self) -> None:
        gw = FakeGateway(completion())
        history = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "name": "lookup", "arguments": {"q": 1}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "42"},
        ]
        await make_llm(gw).invoke("hi", messages=history)
        assistant = gw.payloads[0]["messages"][1]
        assert assistant["tool_calls"] == [
            {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": '{"q": 1}'}}
        ]
        assert gw.payloads[0]["messages"][2] == history[1]


class TestResponseParsing:
    async def test_parses_content_usage_and_provider_info(self) -> None:
        gw = FakeGateway(
            completion(
                "answer",
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "prompt_tokens_details": {"cached_tokens": 4},
                    "completion_tokens_details": {"reasoning_tokens": 2},
                },
                extra_fields={"model_requested": "real-model", "provider": "upstream"},
            )
        )
        response = await make_llm(gw, provider="litellm").invoke("hi")

        assert response.content == "answer"
        assert response.is_final
        assert response.usage is not None
        assert (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens) == (10, 5, 15)
        assert response.usage.cached_tokens == 4
        assert response.usage.reasoning_tokens == 2
        assert response.model_requested == "real-model"
        assert response.provider == "upstream"
        assert response.raw_response["choices"][0]["message"]["content"] == "answer"

    async def test_non_gateway_provider_ignores_extra_fields(self) -> None:
        gw = FakeGateway(completion("x", extra_fields={"model_requested": "other", "provider": "other"}))
        response = await make_llm(gw, provider="openai").invoke("hi")
        assert (response.model_requested, response.provider) == ("m1", "openai")

    async def test_parses_tool_calls(self) -> None:
        gw = FakeGateway(
            completion(
                None,
                tool_calls=[{"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": '{"q": "x"}'}}],
            )
        )
        response = await make_llm(gw).invoke("hi", tools=[TOOL])
        assert response.has_tool_calls
        assert response.tool_calls is not None
        assert response.tool_calls[0].name == "lookup"
        assert response.tool_calls[0].input == {"q": "x"}
        assert response.tool_calls[0].id == "c1"

    async def test_empty_response_raises(self) -> None:
        gw = FakeGateway(completion(None))
        with pytest.raises(EmptyLLMResponseError) as exc:
            await make_llm(gw).invoke("hi")
        assert exc.value.model == "m1"

    async def test_invalid_json_body_raises_request_error(self) -> None:
        gw = FakeGateway(httpx.Response(200, content=b"not json"))
        with pytest.raises(LLMRequestError):
            await make_llm(gw).invoke("hi")


class TestRetries:
    async def test_retries_retryable_status_then_succeeds(self) -> None:
        gw = FakeGateway(httpx.Response(429, headers={"Retry-After": "0"}), httpx.Response(503), completion("ok"))
        response = await make_llm(gw, max_retries=2).invoke("hi")
        assert response.content == "ok"
        assert len(gw.requests) == 3

    async def test_gives_up_after_max_retries_with_status(self) -> None:
        gw = FakeGateway(httpx.Response(500, text="boom"))
        with pytest.raises(LLMRequestError) as exc:
            await make_llm(gw, max_retries=1).invoke("hi")
        assert exc.value.status_code == 500
        assert exc.value.body == "boom"
        assert len(gw.requests) == 2

    async def test_non_retryable_status_fails_immediately(self) -> None:
        gw = FakeGateway(httpx.Response(400, text="bad request"))
        with pytest.raises(LLMRequestError) as exc:
            await make_llm(gw, max_retries=3).invoke("hi")
        assert exc.value.status_code == 400
        assert len(gw.requests) == 1

    async def test_transport_errors_are_retried_then_surfaced(self) -> None:
        def explode(request: httpx.Request, index: int) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        gw = FakeGateway(explode)
        with pytest.raises(LLMRequestError) as exc:
            await make_llm(gw, max_retries=1).invoke("hi")
        assert exc.value.status_code is None
        assert isinstance(exc.value.cause, httpx.ConnectError)
        assert len(gw.requests) == 2


class TestStreaming:
    async def test_yields_deltas_tool_calls_and_usage(self) -> None:
        gw = FakeGateway(
            sse(
                {"choices": [{"delta": {"content": "Hel"}}], "extra_fields": {"model_requested": "real", "provider": "up"}},
                {"choices": [{"delta": {"content": "lo"}}]},
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "lookup", "arguments": '{"q":'}}]}}]},
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ' "x"}'}}]}}]},
                {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}},
            )
        )
        llm = make_llm(gw, provider="sofias")

        events = [e async for e in llm.stream_invoke("hi", tools=[TOOL])]

        assert gw.payloads[0]["stream"] is True
        assert gw.payloads[0]["stream_options"] == {"include_usage": True}
        deltas = [e.delta for e in events if e.delta]
        assert "".join(deltas) == "Hello"
        tool_events = [e for e in events if e.tool_call]
        assert len(tool_events) == 1
        assert tool_events[0].tool_call is not None
        assert tool_events[0].tool_call.name == "lookup"
        assert tool_events[0].tool_call.input == {"q": "x"}
        assert tool_events[0].tool_call.id == "c1"
        assert events[-1].usage is not None and events[-1].usage.total_tokens == 3
        assert all(e.model_requested == "real" and e.provider == "up" for e in events)

    async def test_malformed_chunks_are_skipped(self) -> None:
        gw = FakeGateway(sse("data: {not json", {"choices": [{"delta": {"content": "ok"}}]}, ": comment", ""))
        events = [e async for e in make_llm(gw).stream_invoke("hi")]
        assert [e.delta for e in events] == ["ok"]

    async def test_empty_stream_raises(self) -> None:
        gw = FakeGateway(sse({"choices": [{"delta": {}}]}))
        with pytest.raises(EmptyLLMResponseError):
            async for _ in make_llm(gw).stream_invoke("hi"):
                pass

    async def test_stream_retries_transient_failures(self) -> None:
        gw = FakeGateway(httpx.Response(502), sse({"choices": [{"delta": {"content": "ok"}}]}))
        events = [e async for e in make_llm(gw, max_retries=1).stream_invoke("hi")]
        assert [e.delta for e in events] == ["ok"]
        assert len(gw.requests) == 2

    async def test_unparseable_tool_arguments_become_empty_input(self) -> None:
        gw = FakeGateway(
            sse({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c", "function": {"name": "lookup", "arguments": "{oops"}}]}}]})
        )
        events = [e async for e in make_llm(gw).stream_invoke("hi")]
        assert events[0].tool_call is not None and events[0].tool_call.input == {}


class TestLifecycle:
    async def test_async_context_manager_closes_client(self) -> None:
        async with make_llm(FakeGateway(completion())) as llm:
            assert (await llm.invoke("hi")).content == "hello"
        assert llm._client.is_closed

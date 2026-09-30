"""OpenAI-compatible chat-completions client implementing `StreamableLLMCallable`.

Any gateway or provider that speaks the OpenAI ``/chat/completions`` wire
format (the Sofias gateway, LiteLLM, OpenRouter, vLLM, OpenAI itself, ...) works with
this adapter. It is built on ``httpx`` only, so the SDK stays free of any
vendor SDK.

Retries: transport errors and HTTP 408/409/429/5xx are retried with
exponential backoff (honouring ``Retry-After`` when present) up to
``max_retries`` times. Anything else is surfaced as `LLMRequestError`.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import AsyncIterator
from typing import Any, ClassVar, cast

import httpx

from sofias_sdk_lite.errors.exceptions import EmptyLLMResponseError, LLMRequestError
from sofias_sdk_lite.llm.protocol import LLMResponse, LLMStreamEvent
from sofias_sdk_lite.llm.tokens import TokenUsage
from sofias_sdk_lite.llm.tools import ToolCall, ToolSpec
from sofias_sdk_lite.observability._log import get_logger

logger = get_logger("llm.openai_compatible")

__all__ = ["OpenAICompatibleLLM"]

_CHAT_COMPLETIONS_PATH = "/chat/completions"
_RETRYABLE_STATUS: frozenset[int] = frozenset({408, 409, 429, 500, 502, 503, 504})
_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_MAX_SECONDS = 8.0
_BODY_PREVIEW_CHARS = 500
_SSE_DATA_PREFIX = "data:"
_SSE_DONE = "[DONE]"


def _as_dict(value: Any) -> dict[str, Any]:
    """Narrow a JSON value to a dict (empty dict when it is anything else)."""
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    """Narrow a JSON value to a list (empty list when it is anything else)."""
    return cast(list[Any], value) if isinstance(value, list) else []


class OpenAICompatibleLLM:
    """`StreamableLLMCallable` over the OpenAI chat-completions HTTP API.

    Args:
        api_key: Bearer token. Empty string sends no ``Authorization`` header
            (for gateways that do not authenticate on an internal network).
        base_url: API root, e.g. ``https://api.openai.com/v1``.
        model: Model identifier (or gateway alias) sent on every request.
        provider: Label reported in `LLMResponse.provider` and logs. Gateways
            listed in ``GATEWAY_PROVIDERS`` have the real upstream provider
            and model read back from the response's ``extra_fields``.
        temperature: Sampling temperature.
        max_retries: Retries on transient failures (transport errors,
            408/409/429/5xx). ``0`` disables retrying.
        timeout: Per-request timeout in seconds.
        max_tokens: Output budget sent on every request; ``None`` omits it.
        default_headers: Extra headers sent on every request.
        transport: Optional ``httpx`` transport (tests inject a
            ``MockTransport`` here).
    """

    GATEWAY_PROVIDERS: ClassVar[frozenset[str]] = frozenset({"sofias", "litellm", "openrouter"})

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        provider: str = "openai",
        temperature: float = 0.3,
        max_retries: int = 3,
        timeout: float = 120.0,
        max_tokens: int | None = None,
        default_headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not model:
            raise ValueError("model must be a non-empty string")
        if not base_url:
            raise ValueError("base_url must be a non-empty string")
        self._model = model
        self._provider = provider
        self._temperature = temperature
        self._max_retries = max(0, max_retries)
        self._max_tokens = max_tokens

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if default_headers:
            headers.update(default_headers)

        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    # ------------------------------------------------------------------ props

    @property
    def model(self) -> str:
        """Model identifier sent on every request."""
        return self._model

    @property
    def provider(self) -> str:
        """Provider label configured for this client."""
        return self._provider

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> OpenAICompatibleLLM:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------ subclassing

    def _augment_request_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Hook for subclasses to inject provider-specific request fields in place."""
        return None

    # --------------------------------------------------------------- building

    @staticmethod
    def _convert_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_schema,
                },
            }
            for tool in tools
        ]

    @staticmethod
    def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Expand the SDK's compact assistant ``tool_calls`` into OpenAI's shape."""
        converted: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") != "assistant" or "tool_calls" not in msg:
                converted.append(msg)
                continue
            openai_tool_calls: list[dict[str, Any]] = []
            for raw in _as_list(msg["tool_calls"]):
                tc = _as_dict(raw)
                if "function" in tc:  # already in OpenAI shape
                    openai_tool_calls.append(tc)
                    continue
                arguments: Any = tc.get("arguments", tc.get("input", {}))
                openai_tool_calls.append(
                    {
                        "id": tc.get("id") or f"call_{len(openai_tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": (
                                json.dumps(arguments) if isinstance(arguments, dict) else arguments
                            ),
                        },
                    }
                )
            converted.append(
                {"role": "assistant", "content": msg.get("content"), "tool_calls": openai_tool_calls}
            )
        return converted

    def _build_payload(
        self,
        prompt: str,
        tools: list[ToolSpec] | None,
        messages: list[dict[str, Any]] | None,
        system_prompt: str | None,
        response_format: dict[str, Any] | None,
        parallel_tool_calls: bool | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        chat_messages: list[dict[str, Any]] = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        chat_messages.append({"role": "user", "content": prompt})
        if messages:
            chat_messages.extend(self._convert_messages(messages))

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": chat_messages,
            "temperature": self._temperature,
        }
        if self._max_tokens is not None:
            payload["max_tokens"] = self._max_tokens
        if tools:
            payload["tools"] = self._convert_tools(tools)
            if parallel_tool_calls is not None:
                payload["parallel_tool_calls"] = parallel_tool_calls
        if response_format:
            payload["response_format"] = response_format
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}

        self._augment_request_kwargs(payload)
        return payload

    # ---------------------------------------------------------------- parsing

    @staticmethod
    def _parse_usage(usage: Any) -> TokenUsage | None:
        if not isinstance(usage, dict):
            return None
        usage_dict = _as_dict(usage)
        prompt_details = _as_dict(usage_dict.get("prompt_tokens_details"))
        completion_details = _as_dict(usage_dict.get("completion_tokens_details"))
        return TokenUsage(
            prompt_tokens=int(usage_dict.get("prompt_tokens") or 0),
            completion_tokens=int(usage_dict.get("completion_tokens") or 0),
            total_tokens=int(usage_dict.get("total_tokens") or 0),
            cached_tokens=prompt_details.get("cached_tokens"),
            reasoning_tokens=completion_details.get("reasoning_tokens"),
        )

    def _parse_provider_info(self, data: dict[str, Any]) -> tuple[str, str]:
        """Return ``(model_requested, provider)``, reading gateway metadata when present."""
        if self._provider in self.GATEWAY_PROVIDERS:
            extra = _as_dict(data.get("extra_fields"))
            if extra:
                return (
                    str(extra.get("model_requested") or self._model),
                    str(extra.get("provider") or self._provider),
                )
        return self._model, self._provider

    @staticmethod
    def _parse_tool_calls(message: dict[str, Any]) -> list[ToolCall] | None:
        raw_calls = _as_list(message.get("tool_calls"))
        if not raw_calls:
            return None
        tool_calls: list[ToolCall] = []
        for raw in raw_calls:
            tc = _as_dict(raw)
            function = _as_dict(tc.get("function"))
            arguments: Any = function.get("arguments") or "{}"
            parsed: dict[str, Any] = (
                _as_dict(json.loads(arguments)) if isinstance(arguments, str) else _as_dict(arguments)
            )
            tool_calls.append(
                ToolCall(name=str(function.get("name") or ""), input=parsed, id=tc.get("id"))
            )
        return tool_calls

    # --------------------------------------------------------------- transport

    def _request_error(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: str | None = None,
        cause: Exception | None = None,
    ) -> LLMRequestError:
        return LLMRequestError(
            message,
            provider=self._provider,
            model=self._model,
            status_code=status_code,
            body=body[:_BODY_PREVIEW_CHARS] if body else None,
            cause=cause,
        )

    @staticmethod
    def _backoff_seconds(attempt: int, response: httpx.Response | None) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return min(float(retry_after), _BACKOFF_MAX_SECONDS)
                except ValueError:
                    pass
        exponential = _BACKOFF_BASE_SECONDS * (2**attempt)
        return min(exponential, _BACKOFF_MAX_SECONDS) * (0.5 + random.random() / 2)

    async def _send(self, payload: dict[str, Any], *, stream: bool) -> httpx.Response:
        """POST the payload, retrying transient failures. Returns a 2xx response."""
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            request = self._client.build_request("POST", _CHAT_COMPLETIONS_PATH, json=payload)
            try:
                response = await self._client.send(request, stream=stream)
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
                await asyncio.sleep(self._backoff_seconds(attempt, None))
                continue

            if response.is_success:
                return response

            body = (await response.aread()).decode("utf-8", errors="replace")
            await response.aclose()
            if response.status_code in _RETRYABLE_STATUS and attempt < self._max_retries:
                logger.warning(
                    "LLM request failed, retrying",
                    provider=self._provider,
                    model=self._model,
                    status_code=response.status_code,
                    attempt=attempt + 1,
                    max_retries=self._max_retries,
                )
                await asyncio.sleep(self._backoff_seconds(attempt, response))
                continue
            raise self._request_error(
                f"LLM request failed with HTTP {response.status_code}",
                status_code=response.status_code,
                body=body,
            )

        raise self._request_error(
            f"LLM request failed after {self._max_retries + 1} attempt(s): {last_error}",
            cause=last_error,
        )

    # ------------------------------------------------------------------ invoke

    async def invoke(
        self,
        prompt: str,
        tools: list[ToolSpec] | None = None,
        messages: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        response_format: dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        **_extra: Any,
    ) -> LLMResponse:
        """Run one chat completion and return the parsed response."""
        payload = self._build_payload(
            prompt, tools, messages, system_prompt, response_format, parallel_tool_calls, stream=False
        )
        started = time.perf_counter()
        response = await self._send(payload, stream=False)
        try:
            data: dict[str, Any] = _as_dict(response.json())
        except ValueError as exc:
            raise self._request_error(
                "LLM response was not valid JSON", body=response.text, cause=exc
            ) from exc

        choices = _as_list(data.get("choices"))
        message = _as_dict(_as_dict(choices[0]).get("message")) if choices else {}
        content: str | None = message.get("content")
        tool_calls = self._parse_tool_calls(message)
        usage = self._parse_usage(data.get("usage"))
        model_requested, provider = self._parse_provider_info(data)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)

        if not content and not tool_calls:
            logger.warning(
                "LLM returned empty response",
                provider=self._provider,
                model=self._model,
                duration_ms=duration_ms,
            )
            raise EmptyLLMResponseError(provider=self._provider, model=self._model)

        logger.info(
            "LLM call completed",
            provider=provider,
            model=model_requested,
            gateway=self._provider,
            duration_ms=duration_ms,
            total_tokens=usage.total_tokens if usage else None,
            tool_calls_count=len(tool_calls) if tool_calls else 0,
        )
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            raw_response=data,
            model_requested=model_requested,
            provider=provider,
        )

    # ----------------------------------------------------------------- stream

    async def stream_invoke(
        self,
        prompt: str,
        tools: list[ToolSpec] | None = None,
        messages: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        response_format: dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        **_extra: Any,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Run one chat completion as a server-sent-events stream.

        Yields text deltas as they arrive; tool calls are accumulated and
        yielded once complete, followed by a final usage event when the
        gateway reports one.
        """
        payload = self._build_payload(
            prompt, tools, messages, system_prompt, response_format, parallel_tool_calls, stream=True
        )
        started = time.perf_counter()
        response = await self._send(payload, stream=True)

        pending_tool_calls: dict[int, dict[str, Any]] = {}
        usage: TokenUsage | None = None
        has_content = False
        model_requested, provider = self._model, self._provider
        provider_info_parsed = False

        try:
            async for chunk in self._iter_sse(response):
                if not provider_info_parsed:
                    model_requested, provider = self._parse_provider_info(chunk)
                    provider_info_parsed = True

                if chunk.get("usage"):
                    usage = self._parse_usage(chunk["usage"])

                choices = _as_list(chunk.get("choices"))
                if not choices:
                    continue
                delta = _as_dict(_as_dict(choices[0]).get("delta"))

                text: str | None = delta.get("content")
                if text:
                    has_content = True
                    yield LLMStreamEvent(
                        delta=text, model_requested=model_requested, provider=provider
                    )

                for tc_delta in _as_list(delta.get("tool_calls")):
                    self._accumulate_tool_call(pending_tool_calls, _as_dict(tc_delta))
        finally:
            await response.aclose()

        if not has_content and not pending_tool_calls:
            logger.warning(
                "LLM stream returned empty response", provider=self._provider, model=self._model
            )
            raise EmptyLLMResponseError(provider=self._provider, model=self._model)

        for index in sorted(pending_tool_calls):
            tc = pending_tool_calls[index]
            arguments: dict[str, Any] = {}
            if tc["arguments"]:
                try:
                    arguments = _as_dict(json.loads(tc["arguments"]))
                except json.JSONDecodeError:
                    arguments = {}
            yield LLMStreamEvent(
                tool_call=ToolCall(name=tc["name"], input=arguments, id=tc["id"] or None),
                model_requested=model_requested,
                provider=provider,
            )

        if usage is not None:
            yield LLMStreamEvent(usage=usage, model_requested=model_requested, provider=provider)

        logger.info(
            "LLM stream call completed",
            provider=provider,
            model=model_requested,
            gateway=self._provider,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            total_tokens=usage.total_tokens if usage else None,
            tool_calls_count=len(pending_tool_calls),
        )

    @staticmethod
    def _accumulate_tool_call(pending: dict[int, dict[str, Any]], tc_delta: dict[str, Any]) -> None:
        index = int(tc_delta.get("index") or 0)
        function = _as_dict(tc_delta.get("function"))
        entry = pending.setdefault(index, {"id": "", "name": "", "arguments": ""})
        if tc_delta.get("id"):
            entry["id"] = str(tc_delta["id"])
        if function.get("name"):
            entry["name"] = str(function["name"])
        if function.get("arguments"):
            entry["arguments"] += str(function["arguments"])

    @staticmethod
    async def _iter_sse(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
        """Yield each ``data:`` JSON object from an SSE body, stopping at ``[DONE]``."""
        async for raw_line in response.aiter_lines():
            line = raw_line.strip()
            if not line or not line.startswith(_SSE_DATA_PREFIX):
                continue
            data = line[len(_SSE_DATA_PREFIX) :].strip()
            if data == _SSE_DONE:
                return
            try:
                parsed: Any = json.loads(data)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed SSE chunk", preview=data[:80])
                continue
            if isinstance(parsed, dict):
                yield _as_dict(parsed)

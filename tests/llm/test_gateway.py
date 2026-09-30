"""`GatewayLLM`: gateway defaults layered on the OpenAI-compatible client."""

from __future__ import annotations

import pytest

from sofias_sdk_lite.errors.exceptions import LLMConfigurationError
from sofias_sdk_lite.llm import (
    DEFAULT_GATEWAY_MAX_TOKENS,
    PROVIDER_GATEWAY,
    GatewayLLM,
    normalize_gateway_base_url,
)
from tests.llm.conftest import FakeGateway, completion

GATEWAY_URL = "https://gateway.example/v1"


def make(gateway: FakeGateway, **kwargs) -> GatewayLLM:
    defaults = dict(api_key="k", model="default", base_url=GATEWAY_URL)
    defaults.update(kwargs)
    return GatewayLLM(transport=gateway.transport, **defaults)


def test_reports_the_sofias_provider_label() -> None:
    llm = make(FakeGateway(completion()))
    assert llm.provider == PROVIDER_GATEWAY == "sofias"
    assert str(llm._client.base_url) == GATEWAY_URL + "/"


def test_there_is_no_default_address() -> None:
    with pytest.raises(TypeError):
        GatewayLLM(api_key="k", model="default")  # type: ignore[call-arg]


@pytest.mark.parametrize("empty", ["", "   ", "/"])
def test_empty_base_url_is_a_configuration_error(empty: str) -> None:
    with pytest.raises(LLMConfigurationError) as exc:
        make(FakeGateway(completion()), base_url=empty)
    assert "SOFIAS_LLM_BASE_URL" in str(exc.value)


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("https://gateway.example", "https://gateway.example/v1"),
        ("https://gateway.example/", "https://gateway.example/v1"),
        ("http://gateway:8080", "http://gateway:8080/v1"),
        ("http://gateway:8080/v1", "http://gateway:8080/v1"),
        ("http://gateway:8080/v1/", "http://gateway:8080/v1"),
        ("https://gw.example/openai", "https://gw.example/openai"),
    ],
)
def test_normalize_gateway_base_url(given: str, expected: str) -> None:
    assert normalize_gateway_base_url(given) == expected


def test_normalize_rejects_an_empty_url() -> None:
    with pytest.raises(LLMConfigurationError):
        normalize_gateway_base_url("")


async def test_bare_host_gets_v1_appended() -> None:
    gw = FakeGateway(completion())
    await make(gw, base_url="https://gateway.example").invoke("hi")
    assert str(gw.requests[0].url) == "https://gateway.example/v1/chat/completions"


async def test_injects_default_max_tokens_and_fallback_reasoning_effort() -> None:
    gw = FakeGateway(completion())
    await make(gw).invoke("hi")
    payload = gw.payloads[0]
    assert payload["max_tokens"] == DEFAULT_GATEWAY_MAX_TOKENS
    assert payload["reasoning_effort"] == "low"
    assert payload["temperature"] == 0.1


@pytest.mark.parametrize(
    ("model", "effort"),
    [("openai/gpt-oss-120b", "low"), ("deepseek-v3.2", "low"), ("Qwen3-235B", "none"), ("glm-5.3-flash", "low"), ("mystery", "low")],
)
async def test_reasoning_effort_follows_model_family(model: str, effort: str) -> None:
    gw = FakeGateway(completion())
    await make(gw, model=model).invoke("hi")
    assert gw.payloads[0]["reasoning_effort"] == effort


async def test_explicit_reasoning_effort_wins() -> None:
    gw = FakeGateway(completion())
    await make(gw, model="qwen", reasoning_effort="medium").invoke("hi")
    assert gw.payloads[0]["reasoning_effort"] == "medium"


async def test_empty_reasoning_effort_disables_injection() -> None:
    gw = FakeGateway(completion())
    await make(gw, reasoning_effort="").invoke("hi")
    assert "reasoning_effort" not in gw.payloads[0]


async def test_explicit_max_tokens_wins_over_default() -> None:
    gw = FakeGateway(completion())
    await make(gw, max_tokens=512).invoke("hi")
    assert gw.payloads[0]["max_tokens"] == 512


async def test_reads_upstream_provider_from_extra_fields() -> None:
    gw = FakeGateway(completion("x", extra_fields={"model_requested": "model-x", "provider": "upstream-a"}))
    response = await make(gw).invoke("hi")
    assert (response.model_requested, response.provider) == ("model-x", "upstream-a")

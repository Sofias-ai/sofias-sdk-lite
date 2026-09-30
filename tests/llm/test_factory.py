"""`create_llm` resolution (explicit > settings > env) and the default-LLM context."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from sofias_sdk_lite import BaseAgentSettings
from sofias_sdk_lite.errors import LLMConfigurationError
from sofias_sdk_lite.llm import (
    ENV_LLM_API_KEY,
    ENV_LLM_BASE_URL,
    ENV_LLM_MODEL,
    ENV_LLM_PROVIDER,
    PROVIDER_GATEWAY,
    GatewayLLM,
    LLMGatewayConfig,
    OpenAICompatibleLLM,
    create_llm,
    default_llm,
    gateway_config_from_env,
    gateway_config_from_settings,
    get_default_llm,
    resolve_default_llm,
)

GATEWAY_URL = "https://gateway.example/v1"

ENV_VARS = (
    ENV_LLM_MODEL, ENV_LLM_BASE_URL, ENV_LLM_API_KEY, ENV_LLM_PROVIDER,
    "MODEL_NAME", "ROUTER_URL", "ROUTER_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestFromEnv:
    def test_none_when_no_model(self) -> None:
        assert gateway_config_from_env({}) is None
        assert gateway_config_from_env({ENV_LLM_BASE_URL: "http://x"}) is None

    def test_reads_sofias_variables(self) -> None:
        cfg = gateway_config_from_env(
            {ENV_LLM_MODEL: "m", ENV_LLM_BASE_URL: "http://gw/v1", ENV_LLM_API_KEY: "k", ENV_LLM_PROVIDER: "openai"}
        )
        assert cfg == LLMGatewayConfig(model="m", base_url="http://gw/v1", api_key=SecretStr("k"), provider="openai")

    def test_platform_injected_variables_build_a_gateway_client(self) -> None:
        cfg = gateway_config_from_env(
            {ENV_LLM_MODEL: "default", ENV_LLM_BASE_URL: "https://gateway.example", ENV_LLM_API_KEY: "k"}
        )
        assert cfg is not None
        assert (cfg.model, cfg.base_url, cfg.api_key.get_secret_value()) == ("default", "https://gateway.example", "k")
        llm = create_llm(cfg)
        assert isinstance(llm, GatewayLLM)
        assert str(llm._client.base_url) == "https://gateway.example/v1/"

    def test_sofias_variables_win_over_platform_names(self) -> None:
        cfg = gateway_config_from_env(
            {ENV_LLM_MODEL: "a", "MODEL_NAME": "c", "ROUTER_API_KEY": "k-router"}
        )
        assert cfg is not None
        assert cfg.model == "a"
        assert cfg.api_key.get_secret_value() == "k-router"  # falls through per variable

    def test_platform_names_are_accepted_as_fallback(self) -> None:
        cfg = gateway_config_from_env({"MODEL_NAME": "m", "ROUTER_URL": "http://gw/v1", "ROUTER_API_KEY": "k"})
        assert cfg is not None
        assert (cfg.model, cfg.base_url, cfg.api_key.get_secret_value(), cfg.provider) == ("m", "http://gw/v1", "k", PROVIDER_GATEWAY)

    def test_model_without_url_has_no_default_address(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_LLM_MODEL, "default")
        cfg = gateway_config_from_env()
        assert cfg is not None
        assert cfg.base_url == ""
        assert cfg.provider == PROVIDER_GATEWAY
        with pytest.raises(LLMConfigurationError) as exc:
            create_llm(cfg)
        assert exc.value.missing == ["base_url"]


class TestFromSettings:
    def test_none_without_model_name(self) -> None:
        assert gateway_config_from_settings(BaseAgentSettings()) is None
        assert not BaseAgentSettings().has_llm_gateway

    def test_reads_base_agent_settings(self) -> None:
        settings = BaseAgentSettings(model_name="m", router_url="http://gw/v1", router_api_key="secret")
        assert settings.has_llm_gateway
        cfg = gateway_config_from_settings(settings)
        assert cfg is not None
        assert (cfg.model, cfg.base_url, cfg.api_key.get_secret_value()) == ("m", "http://gw/v1", "secret")
        assert "secret" not in repr(settings)

    def test_duck_typed_settings_work(self) -> None:
        class Legacy:
            model_name = "m"
            router_api_key = "plain"

        cfg = gateway_config_from_settings(Legacy())
        assert cfg is not None
        assert cfg.api_key.get_secret_value() == "plain"
        assert cfg.base_url == ""

    def test_router_provider_selects_adapter(self) -> None:
        settings = BaseAgentSettings(model_name="m", router_provider="openai")
        cfg = gateway_config_from_settings(settings)
        assert cfg is not None and cfg.provider == "openai"


class TestCreateLLM:
    def test_explicit_config_builds_the_gateway_client_by_default(self) -> None:
        llm = create_llm(LLMGatewayConfig(model="default", base_url=GATEWAY_URL, api_key=SecretStr("k")))
        assert isinstance(llm, GatewayLLM)
        assert llm.model == "default"

    def test_keyword_overrides_apply_on_top_of_config(self) -> None:
        cfg = LLMGatewayConfig(model="a", provider=PROVIDER_GATEWAY)
        llm = create_llm(cfg, model="b", provider="openai", base_url="http://o/v1")
        assert isinstance(llm, OpenAICompatibleLLM) and not isinstance(llm, GatewayLLM)
        assert llm.model == "b"
        assert llm.provider == "openai"

    def test_keywords_alone_are_enough(self) -> None:
        llm = create_llm(model="m", base_url="http://gw/v1", api_key="k", provider="litellm")
        assert llm.provider == "litellm"

    def test_settings_source(self) -> None:
        llm = create_llm(BaseAgentSettings(model_name="m", router_url=GATEWAY_URL, router_api_key="k"))
        assert isinstance(llm, GatewayLLM) and llm.model == "m"

    def test_env_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_LLM_MODEL, "env-model")
        monkeypatch.setenv(ENV_LLM_BASE_URL, GATEWAY_URL)
        monkeypatch.setenv(ENV_LLM_API_KEY, "k")
        assert create_llm().model == "env-model"

    def test_settings_without_model_fall_through_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_LLM_MODEL, "env-model")
        monkeypatch.setenv(ENV_LLM_BASE_URL, GATEWAY_URL)
        assert create_llm(BaseAgentSettings()).model == "env-model"

    def test_missing_everything_raises_actionable_error(self) -> None:
        with pytest.raises(LLMConfigurationError) as exc:
            create_llm()
        assert ENV_LLM_MODEL in str(exc.value)
        assert exc.value.missing == ["model"]

    def test_missing_url_raises_actionable_error(self) -> None:
        with pytest.raises(LLMConfigurationError) as exc:
            create_llm(model="m")
        assert ENV_LLM_BASE_URL in str(exc.value)
        assert exc.value.missing == ["base_url"]

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(LLMConfigurationError) as exc:
            create_llm(model="m", base_url=GATEWAY_URL, provider="carrier-pigeon")
        assert "carrier-pigeon" in str(exc.value)

    def test_temperature_max_tokens_and_reasoning_effort_reach_the_adapter(self) -> None:
        llm = create_llm(model="m", base_url=GATEWAY_URL, temperature=0.9, max_tokens=123, reasoning_effort="high")
        assert isinstance(llm, GatewayLLM)
        assert llm._temperature == 0.9
        assert llm._max_tokens == 123
        assert llm._reasoning_effort == "high"


class TestDefaultLLMContext:
    def test_no_default_outside_scope(self) -> None:
        assert get_default_llm() is None
        assert resolve_default_llm() is None

    def test_scope_installs_and_restores(self) -> None:
        marker = object()
        with default_llm(marker):  # type: ignore[arg-type]
            assert get_default_llm() is marker
            assert resolve_default_llm() is marker
        assert get_default_llm() is None

    def test_none_leaves_outer_scope_untouched(self) -> None:
        outer = object()
        with default_llm(outer), default_llm(None):  # type: ignore[arg-type]
            assert get_default_llm() is outer

    def test_resolve_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_LLM_MODEL, "env-model")
        monkeypatch.setenv(ENV_LLM_BASE_URL, GATEWAY_URL)
        llm = resolve_default_llm()
        assert isinstance(llm, GatewayLLM) and llm.model == "env-model"

    def test_resolve_with_model_but_no_url_is_a_configuration_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_LLM_MODEL, "env-model")
        with pytest.raises(LLMConfigurationError):
            resolve_default_llm()

    def test_context_default_beats_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_LLM_MODEL, "env-model")
        marker = object()
        with default_llm(marker):  # type: ignore[arg-type]
            assert resolve_default_llm() is marker

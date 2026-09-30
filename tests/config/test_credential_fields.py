"""``BaseAgentSettings.CREDENTIAL_FIELDS`` keeps secrets out of prompt templates."""

from __future__ import annotations

from pydantic import SecretStr

from sofias_sdk_lite import BaseAgentSettings
from sofias_sdk_lite.config.runtime_config import SettingsResolver
from sofias_sdk_lite.llm import ToolSpec
from sofias_sdk_lite.nodes.llm_node import LLMNode
from tests.nodes.llm_fakes import CONTRACT, FakeLLM, make_config


class SecretSettings(BaseAgentSettings):
    CREDENTIAL_FIELDS = frozenset({"api_key"})

    model_name: str = "gpt-4o-mini"
    api_key: SecretStr = SecretStr("sk-live-123")


def test_base_class_declares_gateway_key_as_credential() -> None:
    assert BaseAgentSettings.CREDENTIAL_FIELDS == frozenset({"router_api_key"})


def test_secretstr_never_leaks_in_repr_or_dump() -> None:
    settings = SecretSettings()
    assert "sk-live-123" not in repr(settings)
    assert "sk-live-123" not in str(settings.model_dump())
    assert settings.api_key.get_secret_value() == "sk-live-123"


def test_credential_fields_are_excluded_from_runtime_vars() -> None:
    resolver: SettingsResolver[SecretSettings] = SettingsResolver(SecretSettings)
    resolver.set_current(SecretSettings())
    node = LLMNode(
        config=make_config(),
        contract=CONTRACT,
        llm=FakeLLM(),
        settings_resolver=resolver,
    )

    runtime_vars = node._get_runtime_vars()

    assert "api_key" not in runtime_vars
    assert runtime_vars["model_name"] == "gpt-4o-mini"


def test_overhead_estimate_helper_sees_tool_specs() -> None:
    node = LLMNode(config=make_config(system_prompt=None), contract=CONTRACT, llm=FakeLLM())
    specs = [ToolSpec(name="a", description="b" * 32, parameters_schema={})]
    assert node._estimate_fixed_overhead_tokens("", specs) > 0

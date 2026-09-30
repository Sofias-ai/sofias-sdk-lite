"""Tests for ConfigSource implementations."""

from __future__ import annotations

import json

import pytest

from sofias_sdk_lite.config import EnvConfigSource, StaticConfigSource
from sofias_sdk_lite.errors import ConfigSourceError


@pytest.mark.asyncio
async def test_static_config_source_returns_configured_dict():
    source = StaticConfigSource({"model_name": "gpt-4o-mini"})
    result = await source.fetch("my_agent")
    assert result == {"model_name": "gpt-4o-mini"}


@pytest.mark.asyncio
async def test_static_config_source_returns_a_copy_not_a_reference():
    original = {"key": "value"}
    source = StaticConfigSource(original)
    result = await source.fetch("my_agent")
    result["key"] = "mutated"
    assert original["key"] == "value"


@pytest.mark.asyncio
async def test_env_config_source_reads_json_from_env_var(monkeypatch):
    monkeypatch.setenv("SOFIAS_AGENT_CONFIG", json.dumps({"api_key": "sk-test"}))
    source = EnvConfigSource()
    result = await source.fetch("my_agent")
    assert result == {"api_key": "sk-test"}


@pytest.mark.asyncio
async def test_env_config_source_raises_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("SOFIAS_AGENT_CONFIG", raising=False)
    source = EnvConfigSource()
    with pytest.raises(ConfigSourceError):
        await source.fetch("my_agent")


@pytest.mark.asyncio
async def test_env_config_source_raises_on_invalid_json(monkeypatch):
    monkeypatch.setenv("SOFIAS_AGENT_CONFIG", "{not valid json")
    source = EnvConfigSource()
    with pytest.raises(ConfigSourceError):
        await source.fetch("my_agent")

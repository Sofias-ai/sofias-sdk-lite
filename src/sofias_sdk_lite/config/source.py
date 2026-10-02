"""Protocol-based runtime configuration sources.

The internal Sofias platform resolves agent runtime configuration by
calling a private backend API. That backend is not part of this public
SDK. Instead, this module defines `ConfigSource` — a small protocol that
any application can implement to supply runtime configuration however it
sees fit (a database, a feature-flag service, a config file, environment
variables, a static dict, or indeed an internal backend API of your own).

Two ready-to-use implementations are provided for the common cases:

- `StaticConfigSource`: always returns the same dict. Useful for local
  development, tests, and simple deployments with one fixed configuration.
- `EnvConfigSource`: reads a JSON blob from a single environment variable.
  Useful for container-based deployments that inject configuration via
  the environment.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol, runtime_checkable

from sofias_sdk_lite.errors.exceptions import ConfigSourceError


@runtime_checkable
class ConfigSource(Protocol):
    """Supplies runtime configuration for an agent, keyed by agent name and conversation.

    Implementations may fetch configuration from anywhere: a database, a
    remote service, a local file, environment variables, or an in-memory
    dict. The only contract is the `fetch` coroutine's signature and
    return type.

    Example:
        ```python
        class DatabaseConfigSource:
            async def fetch(
                self, agent_name: str, conversation_id: str | None = None
            ) -> dict[str, Any]:
                return await my_db.get_agent_config(agent_name)
        ```
    """

    async def fetch(self, agent_name: str, conversation_id: str | None = None) -> dict[str, Any]:
        """Fetch the runtime configuration payload for an agent.

        Args:
            agent_name: The name/identity of the agent requesting configuration.
            conversation_id: Optional conversation identifier, for sources
                that vary configuration per-conversation (e.g. A/B tests,
                per-tenant overrides).

        Returns:
            A raw configuration dictionary, typically passed to a
            `sofias_sdk_lite.config.runtime_config.SettingsResolver` for
            validation into an `AgentSettings` subclass.
        """
        ...


class StaticConfigSource:
    """A `ConfigSource` that always returns the same static dict.

    Useful for local development, tests, and any deployment where a single
    fixed configuration applies to every agent and conversation.

    Example:
        ```python
        source = StaticConfigSource({"model_name": "gpt-4o-mini", "api_key": "..."})
        config = await source.fetch("my_agent")
        ```
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with the static configuration to always return.

        Args:
            config: The configuration dictionary to return from every
                `fetch` call. Stored by reference; a shallow copy is
                returned on each call so callers cannot mutate the
                stored configuration through the returned dict.
        """
        self._config = config

    async def fetch(self, agent_name: str, conversation_id: str | None = None) -> dict[str, Any]:
        """Return a copy of the static configuration.

        Args:
            agent_name: Unused; accepted to satisfy the `ConfigSource` protocol.
            conversation_id: Unused; accepted to satisfy the `ConfigSource` protocol.

        Returns:
            A shallow copy of the configured dict.
        """
        return dict(self._config)


class EnvConfigSource:
    """A `ConfigSource` that reads a JSON blob from a single environment variable.

    Useful for container-based deployments that inject the full
    configuration payload as one environment variable containing a JSON
    object, e.g.:

        export SOFIAS_AGENT_CONFIG='{"model_name": "gpt-4o-mini", "api_key": "sk-..."}'

    Example:
        ```python
        source = EnvConfigSource()  # reads SOFIAS_AGENT_CONFIG
        config = await source.fetch("my_agent")
        ```
    """

    def __init__(self, env_var: str = "SOFIAS_AGENT_CONFIG") -> None:
        """Initialize with the name of the environment variable to read.

        Args:
            env_var: Name of the environment variable holding a JSON object.
                Defaults to `"SOFIAS_AGENT_CONFIG"`.
        """
        self._env_var = env_var

    async def fetch(self, agent_name: str, conversation_id: str | None = None) -> dict[str, Any]:
        """Read and parse the configured environment variable.

        Args:
            agent_name: Unused for lookup, but included in error context if
                the environment variable is missing or invalid.
            conversation_id: Unused; accepted to satisfy the `ConfigSource` protocol.

        Returns:
            The parsed JSON object as a dict.

        Raises:
            ConfigSourceError: If the environment variable is unset, empty,
                not valid JSON, or does not decode to a JSON object (dict).
        """
        raw = os.environ.get(self._env_var)
        if raw is None or raw.strip() == "":
            raise ConfigSourceError(
                f"Environment variable '{self._env_var}' is not set",
                source="EnvConfigSource",
                agent_name=agent_name,
            )

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ConfigSourceError(
                f"Environment variable '{self._env_var}' does not contain valid JSON",
                source="EnvConfigSource",
                agent_name=agent_name,
                cause=e,
            ) from e

        if not isinstance(parsed, dict):
            raise ConfigSourceError(
                f"Environment variable '{self._env_var}' must decode to a JSON object, "
                f"got {type(parsed).__name__}",
                source="EnvConfigSource",
                agent_name=agent_name,
            )

        return parsed


__all__ = [
    "ConfigSource",
    "StaticConfigSource",
    "EnvConfigSource",
]

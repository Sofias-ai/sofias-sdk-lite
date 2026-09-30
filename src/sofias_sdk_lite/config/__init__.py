"""Configuration module for the Sofias Agents SDK."""

from sofias_sdk_lite.config.agent_settings import BaseAgentSettings
from sofias_sdk_lite.config.defaults import (
    DEFAULT_ITERATION_TIMEOUT_SECONDS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TOOL_LOOP_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_RETRY_BACKOFF_MULTIPLIER,
    DEFAULT_RETRY_DELAY_SECONDS,
    DEFAULT_RETRY_MAX_DELAY_SECONDS,
    DEFAULT_STRICT_VALIDATION,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOOL_LOOP_TOKEN_RESERVE,
    DEFAULT_TOP_P,
    DEFAULT_VERBOSE_LOGGING,
)
from sofias_sdk_lite.config.runtime_config import (
    AgentSettings,
    RuntimeConfigError,
    SettingsResolver,
)
from sofias_sdk_lite.config.sdk_config import (
    LLMConfig,
    RetryConfig,
    SDKConfig,
    ToolLoopConfig,
)
from sofias_sdk_lite.config.source import (
    ConfigSource,
    EnvConfigSource,
    StaticConfigSource,
)

__all__ = [
    # Defaults
    "DEFAULT_ITERATION_TIMEOUT_SECONDS",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MAX_TOOL_LOOP_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_RETRY_BACKOFF_MULTIPLIER",
    "DEFAULT_RETRY_DELAY_SECONDS",
    "DEFAULT_RETRY_MAX_DELAY_SECONDS",
    "DEFAULT_STRICT_VALIDATION",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOOL_LOOP_TOKEN_RESERVE",
    "DEFAULT_TOP_P",
    "DEFAULT_VERBOSE_LOGGING",
    # SDK Config
    "LLMConfig",
    "RetryConfig",
    "SDKConfig",
    "ToolLoopConfig",
    # Runtime Config
    "AgentSettings",
    "BaseAgentSettings",
    "RuntimeConfigError",
    "SettingsResolver",
    # Config Source
    "ConfigSource",
    "EnvConfigSource",
    "StaticConfigSource",
]

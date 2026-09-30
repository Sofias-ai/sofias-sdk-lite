"""Global SDK configuration.

This module defines the static configuration that applies to the entire SDK.
Developers instantiate this once when setting up their agent.
"""

from pydantic import BaseModel, ConfigDict

from sofias_sdk_lite.config.defaults import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_RETRY_BACKOFF_MULTIPLIER,
    DEFAULT_RETRY_DELAY_SECONDS,
    DEFAULT_RETRY_MAX_DELAY_SECONDS,
    DEFAULT_STRICT_VALIDATION,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_VERBOSE_LOGGING,
)


class LLMConfig(BaseModel):
    """LLM configuration settings."""

    model_config = ConfigDict(frozen=True)

    model: str = DEFAULT_MODEL
    """LLM model identifier."""

    temperature: float = DEFAULT_TEMPERATURE
    """Temperature for LLM calls (0.0 to 2.0)."""

    max_tokens: int = DEFAULT_MAX_TOKENS
    """Maximum tokens in LLM response."""

    top_p: float = DEFAULT_TOP_P
    """Top-p (nucleus sampling) value."""


class RetryConfig(BaseModel):
    """Retry behavior configuration."""

    model_config = ConfigDict(frozen=True)

    max_retries: int = DEFAULT_MAX_RETRIES
    """Maximum number of retry attempts."""

    delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS
    """Initial delay between retries in seconds."""

    backoff_multiplier: float = DEFAULT_RETRY_BACKOFF_MULTIPLIER
    """Multiplier for exponential backoff."""

    max_delay_seconds: float = DEFAULT_RETRY_MAX_DELAY_SECONDS
    """Maximum delay between retries in seconds."""


class ToolLoopConfig(BaseModel):
    """Tool loop configuration settings."""

    model_config = ConfigDict(frozen=True)

    max_iterations: int = DEFAULT_MAX_ITERATIONS
    """Maximum iterations for the tool loop."""


class SDKConfig(BaseModel):
    """Global SDK configuration.

    This is the top-level configuration that developers instantiate once
    when setting up their agent. All values have sensible defaults.

    Example:
        config = SDKConfig(
            llm=LLMConfig(model="gpt-4o-mini"),
            strict_validation=True,
        )
    """

    model_config = ConfigDict(frozen=True)

    llm: LLMConfig = LLMConfig()
    """Default LLM settings for all nodes (can be overridden per-node)."""

    retry: RetryConfig = RetryConfig()
    """Default retry settings for error handling."""

    tool_loop: ToolLoopConfig = ToolLoopConfig()
    """Default tool loop settings."""

    strict_validation: bool = DEFAULT_STRICT_VALIDATION
    """Enable strict validation mode globally."""

    verbose_logging: bool = DEFAULT_VERBOSE_LOGGING
    """Enable verbose logging output."""


__all__ = [
    "LLMConfig",
    "RetryConfig",
    "ToolLoopConfig",
    "SDKConfig",
]

"""Default values for the Sofias Agents SDK.

This module contains all SDK-level default constants.
Single source of truth for default configuration values.
"""

from typing import Final

# =============================================================================
# LLM Defaults
# =============================================================================

DEFAULT_MODEL: Final[str] = "gpt-4o"
"""Default LLM model to use when not specified."""

DEFAULT_TEMPERATURE: Final[float] = 0.7
"""Default temperature for LLM calls."""

DEFAULT_MAX_TOKENS: Final[int] = 4096
"""Default maximum tokens for LLM responses."""

DEFAULT_TOP_P: Final[float] = 1.0
"""Default top_p (nucleus sampling) value."""

# =============================================================================
# Tool Loop Defaults
# =============================================================================

DEFAULT_MAX_ITERATIONS: Final[int] = 10
"""Default maximum iterations for a node's tool loop."""

DEFAULT_MAX_TOOL_LOOP_TOKENS: Final[int] = 128_000
"""Token budget for the WHOLE request of a tool-loop iteration.

The budget means the model's context window, not "what is left for the
history". The fixed prefix (system prompt + user prompt + tool schemas) is
*measured* on every iteration and counted against this number, together with
the accumulated tool-loop history. An agent running a model with a larger
window should raise it via ``NodeToolLoopConfig.max_tool_loop_tokens``.
"""

DEFAULT_TOOL_LOOP_TOKEN_RESERVE: Final[int] = 16_384
"""Tokens reserved for the final LLM response within the tool-loop budget."""

DEFAULT_ITERATION_TIMEOUT_SECONDS: Final[float] = 60.0
"""Default timeout in seconds for a single tool loop iteration."""

# =============================================================================
# Retry Defaults
# =============================================================================

DEFAULT_MAX_RETRIES: Final[int] = 3
"""Default number of retries for recoverable errors."""

DEFAULT_RETRY_DELAY_SECONDS: Final[float] = 1.0
"""Default initial delay between retries in seconds."""

DEFAULT_RETRY_BACKOFF_MULTIPLIER: Final[float] = 2.0
"""Default multiplier for exponential backoff between retries."""

DEFAULT_RETRY_MAX_DELAY_SECONDS: Final[float] = 30.0
"""Default maximum delay between retries in seconds."""

# =============================================================================
# Validation Defaults
# =============================================================================

DEFAULT_STRICT_VALIDATION: Final[bool] = True
"""Default strict validation mode (extra fields forbidden, strict types)."""

# =============================================================================
# Logging Defaults
# =============================================================================

DEFAULT_VERBOSE_LOGGING: Final[bool] = False
"""Default verbose logging mode."""

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TOP_P",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_TOOL_LOOP_TOKENS",
    "DEFAULT_TOOL_LOOP_TOKEN_RESERVE",
    "DEFAULT_ITERATION_TIMEOUT_SECONDS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_DELAY_SECONDS",
    "DEFAULT_RETRY_BACKOFF_MULTIPLIER",
    "DEFAULT_RETRY_MAX_DELAY_SECONDS",
    "DEFAULT_STRICT_VALIDATION",
    "DEFAULT_VERBOSE_LOGGING",
]

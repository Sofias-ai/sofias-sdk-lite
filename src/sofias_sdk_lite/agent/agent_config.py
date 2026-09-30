"""Agent configuration for complete agent definitions.

This module defines the configuration structure for an entire agent,
including its identity, contracts, SDK config reference, and settings class.
"""

from pydantic import BaseModel, ConfigDict

from sofias_sdk_lite.config.runtime_config import AgentSettings
from sofias_sdk_lite.config.sdk_config import SDKConfig
from sofias_sdk_lite.contracts import AgentContract

__all__ = [
    "AgentLLMConfig",
    "AgentConfig",
]


class AgentLLMConfig(BaseModel):
    """LLM configuration at the agent level.

    All fields are optional. When None, values are resolved from
    the SDK-level config, then defaults. These settings apply to
    all nodes that don't define their own LLM config.
    """

    model_config = ConfigDict(frozen=True)

    model: str | None = None
    """LLM model identifier. None means inherit from SDK config."""

    temperature: float | None = None
    """Temperature for LLM calls. None means inherit from SDK config."""

    max_tokens: int | None = None
    """Maximum tokens in LLM response. None means inherit from SDK config."""

    top_p: float | None = None
    """Top-p (nucleus sampling) value. None means inherit from SDK config."""


class AgentConfig(BaseModel):
    """Configuration for a complete agent.

    This is the top-level configuration that defines an agent's identity,
    contracts, and default settings. It serves as the middle layer in the
    three-level configuration resolution: SDK > Agent > Node.

    Example:
        config = AgentConfig(
            name="inbox-assistant",
            version="1.0.0",
            description="Processes and classifies incoming emails",
            contract=AgentContract(EmailInput, ProcessedEmailOutput),
            sdk_config=SDKConfig(),
            settings_class=InboxAssistantSettings,
            llm=AgentLLMConfig(model="gpt-4o"),
        )
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # Identity
    name: str
    """Unique name identifying this agent."""

    version: str
    """Semantic version of the agent (e.g., '1.0.0')."""

    description: str | None = None
    """Human-readable description of what this agent does."""

    # Contracts
    contract: AgentContract
    """Contract defining the agent's input/output schemas."""

    # SDK configuration reference
    sdk_config: SDKConfig
    """Reference to the global SDK configuration."""

    # Runtime settings class (the type, not instance)
    settings_class: type[AgentSettings]
    """The AgentSettings subclass this agent uses for runtime configuration."""

    # Agent-level LLM defaults (optional, inherits from SDK if not set)
    llm: AgentLLMConfig | None = None
    """LLM settings for this agent. None means use SDK defaults."""

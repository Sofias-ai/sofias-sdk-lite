"""Configuration for `AgentRunner`."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from sofias_sdk_lite.rabbitmq.config import RabbitMQConfig

__all__ = ["RunnerConfig"]


class RunnerConfig(BaseModel):
    """Configuration for running an agent against RabbitMQ.

    Every field that determines wire-level behavior (queue name, response
    stream, role mapping) is explicit here — the runner has no built-in
    naming convention linking an agent's identity to its queue or stream.
    """

    model_config = ConfigDict(frozen=True)

    queue: str
    """The RabbitMQ queue this runner consumes tasks from."""

    agent_name: str
    """The agent's identity, independent of the queue it consumes from.

    Used for `StreamFragment.agent` and passed to `ConfigSource.fetch()`.
    """

    rabbitmq: RabbitMQConfig
    """Connection settings for the RabbitMQ broker."""

    response_stream: str = "agent.responses"
    """Stream name for `ChatResponseWorkflow` to publish to.

    This is a neutral default, not a required naming convention — override
    it to match whatever your deployment uses.
    """

    role_map: dict[str, str] = Field(default_factory=dict)
    """Optional mapping from an incoming `user_role` value to your agent's
    own role vocabulary. Unmapped roles pass through unchanged."""

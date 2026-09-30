"""Canonical wire-format models for messages exchanged over RabbitMQ.

These models are the source of truth for messages exchanged between the
platform (publisher) and agents (consumer), and between agents during
delegation.

Field names, types, and aliases are kept **byte-identical** to the
internal Sofias platform's wire format so that agents built with this SDK
interoperate with the existing platform without any translation layer.
Do not rename fields when modifying this module, even where an English
name might read more naturally — the wire contract is the product.

- ``AgentTaskMessage``: inbound message that a conversational agent receives.
- ``ApiTaskMessage``: inbound message for API-triggered agents (no
  conversational context).
- ``AgentTaskResponse``: response that an agent callback returns.
- ``StreamFragment``: a single fragment published to a RabbitMQ stream
  during streaming responses.
- ``ChatRequest`` / ``ChatResponse``: legacy chat-over-RabbitMQ shapes
  (``ChatRequest`` is deprecated in favor of ``AgentTaskMessage``).
- ``DelegationRequest`` / ``DelegationResponse``: standard contract for
  inter-agent delegation.
- ``DelegationTarget`` / ``DelegationStep`` / ``DelegationPlan``: static and
  dynamic (DAG) delegation planning models.
- ``MCPServerConfig``: MCP server connection details.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

import nh3

from sofias_sdk_lite.contracts.strict_contract import StrictContract
from sofias_sdk_lite.observability._log import get_logger

logger = get_logger("messaging.models")

_HAS_HTML = re.compile(r"<[a-zA-Z!/]")

__all__ = [
    "MCPServerConfig",
    "AgentTaskMessage",
    "ApiTaskMessage",
    "AgentTaskResponse",
    "StreamFragment",
    "TextContent",
    "FileContent",
    "FileMessage",
    "UserMessage",
    "ChatRequest",
    "AttachmentResponse",
    "ChatResponse",
    "StatusResponse",
    "DelegationStatus",
    "DelegationRequest",
    "DelegationResponse",
    "DelegationTarget",
    "DelegationStep",
    "DelegationPlan",
]


class MCPServerConfig(BaseModel, populate_by_name=True):
    """Configuration for an MCP server attached to an agent."""

    name: str
    address: str
    port: int
    conf: dict[str, Any] = Field(default={}, alias="configuration")


class _BaseTaskMessage(BaseModel):
    """Shared fields for all task message variants.

    Not exported — use ``AgentTaskMessage`` (conversational) or
    ``ApiTaskMessage`` (API-triggered, no conversational context).
    """

    # --- Content ---
    content: str
    messages: list[dict[str, Any]] = []

    # --- Routing ---
    agent: str

    # --- Agent configuration ---
    agent_configuration: dict[str, Any] = {}
    mcp_servers: list[MCPServerConfig] = []

    # --- Tenant / user context ---
    tenant_identifier: str
    tenant_domain: str = ""
    user_email: str = ""
    user_role: str | None = None
    external_user_id: str | None = None
    channel_type: str = ""

    # --- Agent ecosystem ---
    available_agents: dict[str, Any] | list[str] = {}
    agent_internal_capabilities: str | None = None

    # --- Delegation (optional) ---
    reply_to: str | None = None
    correlation_id: str | None = None


class AgentTaskMessage(_BaseTaskMessage):
    """Canonical schema for the message a conversational agent receives via RabbitMQ.

    Used by both the publisher (platform -> agent) and the delegation
    transport (agent -> agent). The receiver does not need to distinguish
    the origin.
    """

    # --- Message identity (required for conversational agents) ---
    conversation_id: str
    message_id: int | str


class ApiTaskMessage(_BaseTaskMessage):
    """Schema for API-triggered agents that operate without conversational context.

    Same shape as ``AgentTaskMessage`` but ``conversation_id`` and
    ``message_id`` are optional. When absent the agent processes the
    request and delivers results via its own channel (e.g. HTTP callback)
    rather than publishing to a RabbitMQ response stream.
    """

    # --- Message identity (optional for API agents) ---
    conversation_id: str | None = None
    message_id: int | str | None = None


class AgentTaskResponse(BaseModel):
    """Canonical schema for the response returned by an agent callback."""

    response: str
    state: dict[str, Any] = {}


class StreamFragment(BaseModel):
    """A single fragment published to a RabbitMQ stream during streaming.

    Every agent that streams responses to the frontend must publish
    messages conforming to this schema so the frontend can consume
    them uniformly regardless of the source agent.
    """

    conversation_id: str
    message_id: int | str
    content: str
    agent: str

    @field_validator("content", mode="before")
    @classmethod
    def _sanitize_html(cls, v: str) -> str:
        if not v or not _HAS_HTML.search(v):
            return v
        return nh3.clean(v)

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    state: dict[str, Any] = {}
    status: Literal["streaming", "completed", "error"] = "streaming"
    complete: bool = False
    fragment_index: int = 0
    is_last: bool = False


class TextContent(BaseModel):
    """Text content within a message."""

    role: Literal["user", "system", "assistant"]
    content: str


class FileContent(BaseModel):
    """Attached file content."""

    type: Literal["file"]
    url: str
    mime_type: str = Field(..., alias="mime_type")
    filename: str


class FileMessage(BaseModel):
    """Message carrying an attached file."""

    role: Literal["user"]
    content: FileContent


class UserMessage(BaseModel):
    """User message that may contain text or a file."""

    role: Literal["user", "system", "assistant"]
    content: Union[str, FileContent]


class ChatRequest(BaseModel):
    """Inbound chat request from RabbitMQ.

    .. deprecated::
        Use ``AgentTaskMessage`` instead. This model is no longer
        validated by the consumer and will be removed in a future
        release.
    """

    history: Optional[List[UserMessage]] = None
    context: Optional[dict] = None
    messages: Optional[List[UserMessage]] = None


class AttachmentResponse(BaseModel):
    """Attached file response."""

    s3_key: str
    filename: str
    content_type: str
    byte_size: int


class ChatResponse(BaseModel):
    """Chat response to publish on RabbitMQ."""

    conversation_id: Union[str, int]
    message_id: Optional[Union[str, int]] = None
    response: str
    status: Literal["completed", "error", "pending", "streaming"]
    session_id: Union[str, int]
    state: dict = Field(default_factory=dict)
    attachments: Optional[List[AttachmentResponse]] = None


class StatusResponse(BaseModel):
    """Service status response."""

    status: str
    timestamp: str
    version: str
    message: str


class DelegationStatus(str, Enum):
    """Status of a delegation response.

    Attributes:
        SUCCESS: The delegation completed successfully.
        ERROR: The delegation failed with an error.
    """

    SUCCESS = "success"
    ERROR = "error"


class DelegationRequest(StrictContract):
    """Standard request contract for inter-agent delegation.

    This is what an agent sends when delegating work to another agent.
    The target agent receives this as its input.

    Inherits strict validation from ``StrictContract``, making it usable
    as a node's input contract.

    Attributes:
        input: The main instruction or query for the target agent.
        extras: Optional additional data specific to the use case.
            Not validated by the SDK; the receiving agent validates
            what it needs from this field.
    """

    input: str = Field(
        ...,
        description="The instruction, query, or request for the target agent.",
    )

    extras: dict[str, Any] | None = Field(
        default=None,
        description="Optional additional data for the target agent. "
        "Content is not validated by the SDK.",
    )


class DelegationResponse(StrictContract):
    """Standard response contract for inter-agent delegation.

    This is what an agent returns after completing delegated work.
    The calling agent receives this as the delegation result.

    Inherits strict validation from ``StrictContract``, making it usable
    as a node's output contract.

    Attributes:
        status: Whether the delegation succeeded or failed.
        result: The response content when status is SUCCESS.
        error_message: Error description when status is ERROR.
        extras: Optional additional data the agent wants to return
            (sources, scores, metadata). Not validated by the SDK.
    """

    status: DelegationStatus = Field(
        ...,
        description="Whether the delegation completed successfully or failed.",
    )

    result: str | None = Field(
        default=None,
        description="The response content when the delegation succeeded.",
    )

    error_message: str | None = Field(
        default=None,
        description="Error description when the delegation failed.",
    )

    extras: dict[str, Any] | None = Field(
        default=None,
        description="Optional additional response data (sources, scores, metadata). "
        "Content is not validated by the SDK.",
    )


class DelegationTarget(BaseModel):
    """Configuration for a target agent in delegation.

    Defines where to send delegation requests and how to handle
    the communication with a specific target agent.

    Example:
        target = DelegationTarget(
            agent_name="summarizer",
            routing_key="agents.summarizer.tasks",
            input_mapping="transform_for_summarizer",
            output_contract=SummaryOutput,
            timeout_seconds=120,
        )

    .. note::
        Unlike the internal platform, this SDK does **not** assume any
        queue-naming convention. ``routing_key`` must be set explicitly —
        see ``get_routing_key()``.
    """

    model_config = {"frozen": True}

    agent_name: str
    """Name of the target agent to delegate to."""

    routing_key: str | None = None
    """RabbitMQ routing key for the target agent's queue.
    Must be set explicitly — the SDK does not derive this from
    ``agent_name`` via any naming convention."""

    input_mapping: str | None = None
    """Name of a registered mapper function to transform input before delegation.
    If None, the input is sent as-is."""

    output_contract: type[StrictContract] | None = None
    """Expected output contract from the target agent.
    If defined, the response is validated against this schema.
    Accepts OutputContract, DelegationResponse, or any StrictContract subclass.
    If None, any dict response is accepted."""

    timeout_seconds: int = 300
    """Timeout in seconds for waiting for this target's response."""

    def get_routing_key(self) -> str:
        """Return the explicit routing key for this target.

        Raises:
            ValueError: If ``routing_key`` was not set explicitly. The
                public SDK does not assume any ``agent.<name>``-style
                queue-naming convention; callers must supply the routing
                key that matches their own deployment topology.
        """
        if self.routing_key is None:
            raise ValueError(
                f"DelegationTarget for agent '{self.agent_name}' has no "
                "explicit routing_key. Set it when constructing the target "
                "— this SDK does not assume a queue-naming convention."
            )
        return self.routing_key


class DelegationStep(BaseModel):
    """A single step in a delegation plan for DAG execution.

    Represents one delegation task within a plan, including its dependencies
    on other steps. Used when dynamic targets are enabled.

    Example:
        step = DelegationStep(
            id="step_1",
            agent_name="web-search",
            input={"query": "Bitcoin price"},
            depends_on=[],
        )
    """

    model_config = {"frozen": True}

    id: str
    """Unique identifier for this step within the plan."""

    agent_name: str
    """Name of the target agent to delegate to."""

    input: dict[str, Any]
    """Payload to send to the target agent."""

    depends_on: list[str] = []
    """List of step IDs that must complete before this step can run."""

    routing_key: str | None = None
    """Override for the agent's routing key. Must be set explicitly —
    the SDK does not derive this from ``agent_name`` via any naming
    convention."""

    def get_routing_key(self) -> str:
        """Return the explicit routing key for this step.

        Raises:
            ValueError: If ``routing_key`` was not set explicitly. See
                ``DelegationTarget.get_routing_key`` for rationale.
        """
        if self.routing_key is None:
            raise ValueError(
                f"DelegationStep '{self.id}' (agent '{self.agent_name}') has "
                "no explicit routing_key. Set it when constructing the step "
                "— this SDK does not assume a queue-naming convention."
            )
        return self.routing_key


class DelegationPlan(BaseModel):
    """A complete delegation plan for DAG execution.

    Contains a list of steps with their dependencies, forming a directed
    acyclic graph (DAG). This is the expected input format for a
    delegation node running in DAG mode with dynamic targets.

    Example:
        plan = DelegationPlan(
            steps=[
                DelegationStep(id="s1", agent_name="search", input={"q": "A"}),
                DelegationStep(id="s2", agent_name="search", input={"q": "B"}),
                DelegationStep(id="s3", agent_name="synthesizer", input={}, depends_on=["s1", "s2"]),
            ]
        )
    """

    model_config = {"frozen": True}

    steps: list[DelegationStep]
    """List of steps in the delegation plan."""

    @field_validator("steps")
    @classmethod
    def validate_no_duplicate_ids(cls, steps: list[DelegationStep]) -> list[DelegationStep]:
        """Ensure all step IDs are unique."""
        ids = [step.id for step in steps]
        duplicates = [id_ for id_ in ids if ids.count(id_) > 1]
        if duplicates:
            raise ValueError(f"Duplicate step IDs found: {set(duplicates)}")
        return steps

    @model_validator(mode="after")
    def validate_dependencies(self) -> "DelegationPlan":
        """Validate that all dependencies reference existing steps and there are no cycles."""
        step_ids = {step.id for step in self.steps}

        # Check all depends_on reference existing steps
        for step in self.steps:
            for dep_id in step.depends_on:
                if dep_id not in step_ids:
                    raise ValueError(
                        f"Step '{step.id}' depends on '{dep_id}' which does not exist"
                    )

        # Check for cycles using DFS
        self._check_for_cycles(step_ids)

        return self

    def _check_for_cycles(self, step_ids: set[str]) -> None:
        """Detect cycles in the dependency graph using DFS."""
        # Build adjacency list
        graph: dict[str, list[str]] = {step.id: list(step.depends_on) for step in self.steps}

        visited: set[str] = set()
        rec_stack: set[str] = set()

        def dfs(node: str) -> bool:
            """Return True if a cycle is detected."""
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for step_id in step_ids:
            if step_id not in visited:
                if dfs(step_id):
                    raise ValueError("Cycle detected in step dependencies")

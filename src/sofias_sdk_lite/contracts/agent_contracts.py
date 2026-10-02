"""Typed envelopes for agent execution input and output.

This module defines AgentMessage and AgentResponse, the formal contracts
for Agent.execute(). They replace the previous dict-based interface with
strongly-typed envelopes that carry domain data plus transport metadata.

- AgentMessage[T] is generic over the agent's InputContract subclass.
- AgentResponse wraps the validated output with execution metadata.

Example:
    ```python
    class MyInput(InputContract):
        query: str

    message = AgentMessage[MyInput](
        content=MyInput(query="Hello"),
        trace_id="trace-abc",
        runtime_config={"model": "gpt-4o"},
    )

    response = await agent.execute(message)
    assert response.status == ResponseStatus.SUCCESS
    print(response.content)           # validated output dict
    print(response.execution_path)    # ["classifier", "responder"]
    print(response.execution_time_ms) # 1234.5
    ```
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import Field

from sofias_sdk_lite.contracts.node_contracts import InputContract
from sofias_sdk_lite.contracts.strict_contract import StrictContract

__all__ = [
    "ResponseStatus",
    "AgentMessage",
    "AgentResponse",
]

T = TypeVar("T", bound=InputContract)


class ResponseStatus(str, Enum):
    """Status of an agent execution response.

    Attributes:
        SUCCESS: The execution completed successfully.
        ERROR: The execution encountered a controlled business error.
    """

    SUCCESS = "success"
    ERROR = "error"


class AgentMessage(StrictContract, Generic[T]):
    """Typed input envelope for agent execution.

    Generic over the agent's InputContract subclass, so each agent
    can declare its expected input type:

        AgentMessage[RealEstateInput]
        AgentMessage[SupportTicketInput]

    The ``content`` field is validated against the concrete InputContract,
    while the envelope fields carry transport-level metadata.

    Attributes:
        content: Domain input data, validated against the agent's InputContract.
        conversation_id: Conversation thread identifier.
        trace_id: Distributed trace identifier for observability.
        runtime_config: Runtime configuration overrides (model, temperature, etc.).
        metadata: Additional transport-level metadata.
    """

    content: T = Field(
        ...,
        description="Domain input data, validated against the agent's InputContract.",
    )

    conversation_id: str | None = Field(
        default=None,
        description="Conversation thread identifier.",
    )

    trace_id: str | None = Field(
        default=None,
        description="Distributed trace identifier for observability.",
    )

    runtime_config: dict[str, Any] | None = Field(
        default=None,
        description="Runtime configuration overrides (model, temperature, etc.).",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional transport-level metadata.",
    )


class AgentResponse(StrictContract):
    """Typed output envelope from agent execution.

    Wraps the domain output (validated against AgentContract.output_schema)
    with execution metadata. Returned by Agent.execute().

    For controlled business errors, ``status`` is ERROR and ``error_message``
    describes the problem. Infrastructure failures (graph compilation,
    unexpected node explosions) still raise exceptions.

    Attributes:
        content: Domain output data, validated against the agent's OutputContract.
        status: Whether the execution succeeded or hit a business error.
        agent_name: Name of the agent that produced this response.
        agent_version: Version of the agent that produced this response.
        execution_path: Ordered list of nodes executed.
        error_message: Error description when status is ERROR.
        execution_time_ms: Wall-clock execution time in milliseconds.
        metadata: Response metadata (run_id, trace_id, timing, etc.).
    """

    content: dict[str, Any] = Field(
        ...,
        description="Domain output data, validated against the agent's OutputContract.",
    )

    status: ResponseStatus = Field(
        default=ResponseStatus.SUCCESS,
        description="Whether the execution succeeded or hit a business error.",
    )

    agent_name: str = Field(
        ...,
        description="Name of the agent that produced this response.",
    )

    agent_version: str = Field(
        ...,
        description="Version of the agent that produced this response.",
    )

    execution_path: list[str] = Field(
        default_factory=list,
        description="Ordered list of nodes executed.",
    )

    error_message: str | None = Field(
        default=None,
        description="Error description when status is ERROR.",
    )

    execution_time_ms: float | None = Field(
        default=None,
        description="Wall-clock execution time in milliseconds.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Response metadata (run_id, trace_id, timing, etc.).",
    )

    usage: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Token usage grouped by (model_requested, provider).",
    )

    prompt_tokens: int = Field(default=0, deprecated="Use 'usage' list instead.")
    completion_tokens: int = Field(default=0, deprecated="Use 'usage' list instead.")
    total_tokens: int = Field(default=0, deprecated="Use 'usage' list instead.")
    cached_tokens: int = Field(default=0, deprecated="Use 'usage' list instead.")
    reasoning_tokens: int = Field(default=0, deprecated="Use 'usage' list instead.")

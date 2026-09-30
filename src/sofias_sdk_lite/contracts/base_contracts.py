"""Base contracts for inter-agent delegation protocol.

This module defines the standard request/response contracts used when agents
communicate with each other via delegation (RabbitMQ). These contracts provide
a common protocol so that any delegating node knows exactly what to send and
what to expect back, without needing to know the internals of the target agent.

Example:
    # Agent A delegates to Agent B
    request = DelegationRequest(
        input="Summarize this document",
        extras={"document_id": "doc-123", "max_length": 500},
    )

    # Agent B responds
    response = DelegationResponse(
        status=DelegationStatus.SUCCESS,
        result="The document discusses...",
        extras={"word_count": 150, "confidence": 0.95},
    )
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

from sofias_sdk_lite.contracts.strict_contract import StrictContract

__all__ = [
    "DelegationStatus",
    "DelegationRequest",
    "DelegationResponse",
]


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

    Inherits strict validation from StrictContract, making it usable
    as a node's input_contract.

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

    Inherits strict validation from StrictContract, making it usable
    as a node's output_contract.

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

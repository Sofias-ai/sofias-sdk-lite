"""Configuration for AggregatorNode.

This module defines the configuration for nodes that collect responses
from previously dispatched delegations using configurable resolution policies.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

__all__ = ["AggregatorNodeConfig"]


class AggregatorNodeConfig(BaseModel):
    """Configuration for an AggregatorNode that collects delegation responses.

    AggregatorNode implements a fan-in pattern: it waits for responses
    from previously dispatched correlation IDs and resolves according
    to a configurable policy (all, any, majority).

    Example:
        config = AggregatorNodeConfig(
            name="collect_results",
            description="Collects responses from all sub-agents",
            resolution_policy="all",
            timeout_seconds=120.0,
            on_timeout="partial_result",
        )
    """

    model_config = ConfigDict(frozen=True)

    name: str
    """Unique name identifying this node within the agent."""

    description: str | None = None
    """Human-readable description of what this node does."""

    resolution_policy: Literal["all", "any", "majority"] = "all"
    """Policy for deciding when aggregation is resolved:
    - 'all': Wait until all expected responses are received.
    - 'any': Resolve as soon as at least one response arrives.
    - 'majority': Resolve when more than floor(expected * majority_threshold) responses arrive."""

    majority_threshold: float = 0.5
    """Threshold for the 'majority' policy. Only used when resolution_policy='majority'.
    Must be between 0 (exclusive) and 1 (exclusive)."""

    timeout_seconds: float = 300.0
    """Maximum time in seconds to wait for responses before applying on_timeout policy."""

    on_timeout: Literal["partial_result", "fail", "retry_missing"] = "fail"
    """Behavior when timeout is reached before the resolution policy is satisfied:
    - 'fail': Raise AggregationTimeoutError.
    - 'partial_result': Return whatever responses have been collected so far.
    - 'retry_missing': Call the retry_handler for pending IDs and keep waiting."""

    max_retries: int = 1
    """Maximum number of retry rounds when on_timeout='retry_missing'. Ignored for other policies."""

    retry_timeout_seconds: float | None = None
    """Timeout for each retry round when on_timeout='retry_missing'.
    If None, each retry uses the remaining time from the original timeout_seconds.
    If set, each retry gets its own independent timeout window."""

    poll_interval_seconds: float = 2.0
    """Interval in seconds between polling attempts for responses."""

    pending_ids_field: str = "pending_ids"
    """Field name in input_data that contains the list of correlation IDs to wait for."""

    @field_validator("majority_threshold")
    @classmethod
    def validate_majority_threshold(cls, v: float) -> float:
        """Ensure majority_threshold is between 0 and 1 (exclusive)."""
        if not (0 < v < 1):
            raise ValueError(f"majority_threshold must be between 0 and 1 (exclusive), got {v}")
        return v

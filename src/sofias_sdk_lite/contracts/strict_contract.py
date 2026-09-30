"""Base strict contract for all SDK contracts.

This module defines the base class that provides strict validation
configuration for all contracts in the SDK.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = [
    "StrictContract",
]


class StrictContract(BaseModel):
    """Base class for all strictly-validated contracts in the SDK.

    This class provides the common validation configuration used by both:
    - Node contracts (InputContract, OutputContract) for intra-graph data flow
    - Delegation contracts (DelegationRequest, DelegationResponse) for inter-agent protocol

    Configuration:
        - extra="ignore": Silently drop any fields not defined in the schema
          (not "forbid" — unknown fields do not raise; they are discarded)
        - validate_default=True: Validate default values
        - strict=True: Use strict type coercion (no automatic conversions)
    """

    model_config = ConfigDict(
        extra="ignore",
        validate_default=True,
        strict=True,
    )

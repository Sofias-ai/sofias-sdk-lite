"""Contract definitions for the Agent SDK.

This module provides:
- StrictContract: Base class for all strictly-validated contracts
- Node contracts (InputContract, OutputContract) for intra-graph data flow
- Agent contracts (AgentMessage, AgentResponse) for agent execution envelope
- Delegation contracts (DelegationRequest, DelegationResponse) for inter-agent protocol
"""

from sofias_sdk_lite.contracts.agent_contracts import (
    AgentMessage,
    AgentResponse,
    ResponseStatus,
)
from sofias_sdk_lite.contracts.base_contracts import (
    DelegationRequest,
    DelegationResponse,
    DelegationStatus,
)
from sofias_sdk_lite.contracts.node_contracts import (
    AgentContract,
    InputContract,
    InputT,
    NodeContract,
    OutputContract,
    OutputT,
)
from sofias_sdk_lite.contracts.strict_contract import StrictContract

__all__ = [
    # Base contract
    "StrictContract",
    # Node/Agent contracts
    "InputContract",
    "OutputContract",
    "NodeContract",
    "AgentContract",
    "InputT",
    "OutputT",
    # Agent execution envelope
    "AgentMessage",
    "AgentResponse",
    "ResponseStatus",
    # Delegation contracts
    "DelegationRequest",
    "DelegationResponse",
    "DelegationStatus",
]

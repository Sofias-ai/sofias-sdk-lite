"""Node module for the Sofias Agents SDK.

Provides BaseNode and all concrete node types (LLMNode, FunctionNode,
DelegationNode, AggregatorNode, PlannerNode), their configuration models,
and the DAG planning/execution support (ExecutionPlan, PlanExecutor).
"""

from sofias_sdk_lite.nodes.aggregator_config import AggregatorNodeConfig
from sofias_sdk_lite.nodes.aggregator_node import AggregatorNode, RetryHandler
from sofias_sdk_lite.nodes.base_node import BaseNode, NodeType
from sofias_sdk_lite.nodes.delegation_config import (
    DelegationNodeConfig,
    DelegationPlan,
    DelegationStep,
    DelegationTarget,
)
from sofias_sdk_lite.nodes.delegation_node import DelegationNode, InputMapper
from sofias_sdk_lite.nodes.delegation_transport import (
    DelegationTransport,
    NullDelegationTransport,
)
from sofias_sdk_lite.nodes.function_node import FunctionNode, ProcessFn
from sofias_sdk_lite.nodes.llm_node import (
    LLMCallable,
    LLMNode,
    LLMResponse,
    OnChunkHook,
    OnLLMResponseHook,
    OnToolCallHook,
    OnToolResultHook,
    ShouldContinueHook,
    ToolCall,
    ToolResult,
)
from sofias_sdk_lite.nodes.llm_node_config import (
    FunctionNodeConfig,
    LLMNodeConfig,
    LLMSettings,
    NodePromptConfig,
    NodeRetryConfig,
    NodeToolLoopConfig,
)
from sofias_sdk_lite.nodes.plan_executor import PlanExecutor
from sofias_sdk_lite.nodes.planner_node import PlannerNode
from sofias_sdk_lite.nodes.planning_models import (
    ExecutionPlan,
    NodeDescriptor,
    PlanResult,
    PlanStep,
)
from sofias_sdk_lite.nodes.prompt_assembler import PromptAssembler, ToolSpec

__all__ = [
    # Base Node
    "BaseNode",
    "NodeType",
    # LLM Node
    "LLMNode",
    "LLMCallable",
    "LLMResponse",
    "ToolCall",
    "ToolResult",
    # Hook Types
    "OnChunkHook",
    "OnToolCallHook",
    "OnToolResultHook",
    "OnLLMResponseHook",
    "ShouldContinueHook",
    # Node Config
    "LLMNodeConfig",
    "FunctionNodeConfig",
    "LLMSettings",
    "NodePromptConfig",
    "NodeRetryConfig",
    "NodeToolLoopConfig",
    # Prompt Assembler
    "PromptAssembler",
    "ToolSpec",
    # Function Node
    "FunctionNode",
    "ProcessFn",
    # Aggregator Node
    "AggregatorNode",
    "AggregatorNodeConfig",
    "RetryHandler",
    # Delegation Node
    "DelegationNode",
    "DelegationNodeConfig",
    "DelegationPlan",
    "DelegationStep",
    "DelegationTarget",
    "InputMapper",
    # Delegation Transport
    "DelegationTransport",
    "NullDelegationTransport",
    # Planner Node
    "PlannerNode",
    # Planning models + executor
    "ExecutionPlan",
    "NodeDescriptor",
    "PlanResult",
    "PlanStep",
    "PlanExecutor",
]

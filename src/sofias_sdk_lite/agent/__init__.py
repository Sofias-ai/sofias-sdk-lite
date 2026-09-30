"""Agent module: the Agent executor, builder, and supporting runtime pieces."""

from sofias_sdk_lite.agent.agent import Agent
from sofias_sdk_lite.agent.agent_builder import AgentBuildError, AgentBuilder
from sofias_sdk_lite.agent.agent_config import AgentConfig, AgentLLMConfig
from sofias_sdk_lite.agent.execution_context import (
    ExecutionContext,
    TokenAccumulator,
    get_execution_context,
    set_execution_context,
    usage_state_from_list,
)
from sofias_sdk_lite.agent.middleware import (
    AgentMiddleware,
    LoggingMiddleware,
    TimingMiddleware,
)
from sofias_sdk_lite.agent.response_workflow import (
    BaseDelegationResponseWorkflow,
    PublishFn,
    ResponseWorkflow,
    StreamingResponseWorkflow,
)
from sofias_sdk_lite.agent.streaming import (
    AgentCompleteEvent,
    AgentErrorEvent,
    AggregationResponseEvent,
    NodeCompleteEvent,
    NodeStartEvent,
    StreamEvent,
    TextChunkEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)

__all__ = [
    # Agent
    "Agent",
    "get_execution_context",
    "set_execution_context",
    "ExecutionContext",
    "TokenAccumulator",
    "usage_state_from_list",
    # Builder
    "AgentBuilder",
    "AgentBuildError",
    # Config
    "AgentConfig",
    "AgentLLMConfig",
    # Middleware
    "AgentMiddleware",
    "LoggingMiddleware",
    "TimingMiddleware",
    # Response workflow
    "ResponseWorkflow",
    "StreamingResponseWorkflow",
    "BaseDelegationResponseWorkflow",
    "PublishFn",
    # Streaming events
    "StreamEvent",
    "TextChunkEvent",
    "ToolCallStartEvent",
    "ToolCallResultEvent",
    "NodeStartEvent",
    "NodeCompleteEvent",
    "AgentCompleteEvent",
    "AggregationResponseEvent",
    "AgentErrorEvent",
]

"""Sofias Agents SDK — build agents on the Sofias agent graph and run them over RabbitMQ.

Quickstart::

    from sofias_sdk_lite import AgentBuilder, InputContract, NodeContract, OutputContract

    class GreetInput(InputContract):
        name: str

    class GreetOutput(OutputContract):
        greeting: str

    agent = (
        AgentBuilder("hello_agent", version="0.1.0")
        .with_settings_class(MySettings)
        .with_contract(input_schema=GreetInput, output_schema=GreetOutput)
        .add_function_node(
            "greeter",
            NodeContract(input_schema=GreetInput, output_schema=GreetOutput),
            process_fn=lambda data, ctx: {"greeting": f"Hello, {data['name']}!"},
        )
        .set_entry_node("greeter")
        .set_terminal("greeter")
        .build()
    )

See the docs site for the full guide: agent graph concepts, running an agent
against RabbitMQ, streaming, delegation, and testing.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from sofias_sdk_lite.agent import (
    Agent,
    AgentBuildError,
    AgentBuilder,
    AgentConfig,
    AgentLLMConfig,
    AgentMiddleware,
    BaseDelegationResponseWorkflow,
    ExecutionContext,
    LoggingMiddleware,
    PublishFn,
    ResponseWorkflow,
    StreamingResponseWorkflow,
    get_execution_context,
    set_execution_context,
    usage_state_from_list,
)
from sofias_sdk_lite.config import (
    AgentSettings,
    BaseAgentSettings,
    ConfigSource,
    EnvConfigSource,
    RuntimeConfigError,
    SDKConfig,
    SettingsResolver,
    StaticConfigSource,
)
from sofias_sdk_lite.contracts import (
    AgentContract,
    AgentMessage,
    AgentResponse,
    DelegationRequest,
    DelegationResponse,
    DelegationStatus,
    InputContract,
    NodeContract,
    OutputContract,
    ResponseStatus,
    StrictContract,
)
from sofias_sdk_lite.errors import (
    AgentSDKError,
    CircuitBreakerConfig,
    ConfigSourceError,
    EmptyLLMResponseError,
    LLMConfigurationError,
    LLMRequestError,
    ContractValidationError,
    ErrorHandler,
    ErrorHandlerConfig,
    GraphExecutionError,
    InputValidationError,
    NodeExecutionError,
    OutputValidationError,
    RetryPolicy,
    RoutingError,
    ToolExecutionError,
)
from sofias_sdk_lite.llm import (
    GatewayLLM,
    LLMCallable,
    LLMGatewayConfig,
    LLMResponse,
    OpenAICompatibleLLM,
    StreamableLLMCallable,
    create_llm,
    default_llm,
    TokenUsage,
    ToolCall,
    ToolSpec,
)
from sofias_sdk_lite.messaging import (
    AgentTaskMessage,
    ApiTaskMessage,
    ChatRequest,
    ChatResponse,
    FileContent,
    FileMessage,
    StreamFragment,
)
from sofias_sdk_lite.nodes import (
    AggregatorNode,
    AggregatorNodeConfig,
    BaseNode,
    DelegationNode,
    DelegationNodeConfig,
    DelegationTransport,
    FunctionNode,
    LLMNode,
    LLMNodeConfig,
    NodeType,
    NullDelegationTransport,
    PlannerNode,
)
from sofias_sdk_lite.rabbitmq import (
    AckPolicy,
    RabbitMQClient,
    RabbitMQConfig,
    RabbitMQConsumer,
    RabbitMQDelegationTransport,
    RabbitMQPublisher,
    RabbitMQRPCClient,
)
from sofias_sdk_lite.routing import (
    CompositeStrategy,
    ConditionalStrategy,
    ConfidenceThresholdStrategy,
    FanOutStrategy,
    FieldPresenceStrategy,
    FieldValueStrategy,
    Router,
    RoutingStrategy,
    StaticRoute,
)
from sofias_sdk_lite.runner import AgentRunner, RunnerConfig
from sofias_sdk_lite.state import (
    ConversationStateProvider,
    HistoryProvider,
    InMemoryHistory,
    InMemoryStateProvider,
    MemoryProvider,
    Message,
    SummarizingHistoryProvider,
)
from sofias_sdk_lite.workflows import (
    BaseStreamingResponseWorkflow,
    ChatResponseWorkflow,
    NullWorkflow,
)

try:
    # Derived from the installed distribution so it never drifts from
    # pyproject.toml (CI rewrites the version there on tagged releases).
    __version__ = version("sofias-sdk-lite")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    # Agent
    "Agent",
    "AgentBuilder",
    "AgentBuildError",
    "AgentConfig",
    "AgentLLMConfig",
    "AgentMiddleware",
    "LoggingMiddleware",
    "ExecutionContext",
    "get_execution_context",
    "set_execution_context",
    "usage_state_from_list",
    "ResponseWorkflow",
    "StreamingResponseWorkflow",
    "BaseDelegationResponseWorkflow",
    "PublishFn",
    # Config
    "AgentSettings",
    "BaseAgentSettings",
    "SDKConfig",
    "SettingsResolver",
    "RuntimeConfigError",
    "ConfigSource",
    "StaticConfigSource",
    "EnvConfigSource",
    # Contracts
    "StrictContract",
    "InputContract",
    "OutputContract",
    "NodeContract",
    "AgentContract",
    "AgentMessage",
    "AgentResponse",
    "ResponseStatus",
    "DelegationRequest",
    "DelegationResponse",
    "DelegationStatus",
    # Errors
    "AgentSDKError",
    "ContractValidationError",
    "InputValidationError",
    "OutputValidationError",
    "NodeExecutionError",
    "RoutingError",
    "ToolExecutionError",
    "GraphExecutionError",
    "ConfigSourceError",
    "EmptyLLMResponseError",
    "LLMConfigurationError",
    "LLMRequestError",
    "ErrorHandler",
    "ErrorHandlerConfig",
    "RetryPolicy",
    "CircuitBreakerConfig",
    # LLM protocol
    "LLMCallable",
    "StreamableLLMCallable",
    "LLMResponse",
    # LLM clients + factory
    "GatewayLLM",
    "OpenAICompatibleLLM",
    "LLMGatewayConfig",
    "create_llm",
    "default_llm",
    "TokenUsage",
    "ToolCall",
    "ToolSpec",
    # Messaging
    "AgentTaskMessage",
    "ApiTaskMessage",
    "ChatRequest",
    "ChatResponse",
    "StreamFragment",
    "FileMessage",
    "FileContent",
    # Nodes
    "BaseNode",
    "NodeType",
    "LLMNode",
    "LLMNodeConfig",
    "FunctionNode",
    "AggregatorNode",
    "AggregatorNodeConfig",
    "DelegationNode",
    "DelegationNodeConfig",
    "DelegationTransport",
    "NullDelegationTransport",
    "PlannerNode",
    # Routing
    "Router",
    "RoutingStrategy",
    "StaticRoute",
    "FieldValueStrategy",
    "FieldPresenceStrategy",
    "ConditionalStrategy",
    "ConfidenceThresholdStrategy",
    "CompositeStrategy",
    "FanOutStrategy",
    # RabbitMQ
    "RabbitMQConfig",
    "RabbitMQClient",
    "RabbitMQConsumer",
    "RabbitMQPublisher",
    "RabbitMQRPCClient",
    "RabbitMQDelegationTransport",
    "AckPolicy",
    # Runner
    "AgentRunner",
    "RunnerConfig",
    # Workflows
    "ChatResponseWorkflow",
    "BaseStreamingResponseWorkflow",
    "NullWorkflow",
    # State
    "ConversationStateProvider",
    "InMemoryStateProvider",
    "MemoryProvider",
    "HistoryProvider",
    "SummarizingHistoryProvider",
    "InMemoryHistory",
    "Message",
]

"""Error handling module for the Sofias Agents SDK."""

from sofias_sdk_lite.errors.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitBreakerStore,
    CircuitState,
    InMemoryCircuitBreakerStore,
)
from sofias_sdk_lite.errors.error_handler import (
    BackoffStrategy,
    ErrorHandler,
    ErrorHandlerConfig,
    ExecutionErrorContext,
    FallbackConfig,
    RetryPolicy,
)
from sofias_sdk_lite.errors.exceptions import (
    AgentSDKError,
    AggregationTimeoutError,
    ConfigSourceError,
    ContractValidationError,
    DelegationError,
    DelegationPlanError,
    DelegationStepError,
    DelegationTimeoutError,
    EmptyLLMResponseError,
    GraphExecutionError,
    InputValidationError,
    LLMConfigurationError,
    LLMRequestError,
    MaxIterationsError,
    MCPConnectionError,
    MCPToolExecutionError,
    NodeExecutionError,
    OutputValidationError,
    ParallelExecutionError,
    PlanExecutionError,
    RoutingError,
    ToolExecutionError,
)

__all__ = [
    # Exceptions
    "AgentSDKError",
    "AggregationTimeoutError",
    "ConfigSourceError",
    "ContractValidationError",
    "DelegationError",
    "DelegationPlanError",
    "DelegationStepError",
    "DelegationTimeoutError",
    "EmptyLLMResponseError",
    "GraphExecutionError",
    "InputValidationError",
    "LLMConfigurationError",
    "LLMRequestError",
    "MaxIterationsError",
    "MCPConnectionError",
    "MCPToolExecutionError",
    "NodeExecutionError",
    "OutputValidationError",
    "ParallelExecutionError",
    "PlanExecutionError",
    "RoutingError",
    "ToolExecutionError",
    # Error Handler
    "ErrorHandler",
    "ErrorHandlerConfig",
    "ExecutionErrorContext",
    # Retry
    "RetryPolicy",
    "BackoffStrategy",
    # Fallback
    "FallbackConfig",
    # Circuit Breaker
    "CircuitBreakerConfig",
    "CircuitBreakerStore",
    "CircuitBreakerState",
    "CircuitState",
    "InMemoryCircuitBreakerStore",
]

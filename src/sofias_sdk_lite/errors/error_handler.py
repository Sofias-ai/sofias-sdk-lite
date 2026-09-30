"""Error handling for graph execution.

Provides the ErrorHandler class which manages retry logic, fallback
routing, and circuit breaker patterns for node execution in the agent graph.
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field

from sofias_sdk_lite.errors.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitBreakerStore,
    CircuitState,
    InMemoryCircuitBreakerStore,
)
from sofias_sdk_lite.errors.exceptions import (
    GraphExecutionError,
    NodeExecutionError,
    OutputValidationError,
    ToolExecutionError,
)
from sofias_sdk_lite.observability._log import get_logger

logger = get_logger("errors.error_handler")

__all__ = [
    "BackoffStrategy",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "CircuitBreakerStore",
    "CircuitState",
    "ErrorHandler",
    "ErrorHandlerConfig",
    "ExecutionErrorContext",
    "FallbackConfig",
    "InMemoryCircuitBreakerStore",
    "RetryPolicy",
]


class BackoffStrategy(Enum):
    """Backoff strategy for retries."""

    FIXED = "fixed"
    """Use the same delay for all retries."""

    EXPONENTIAL = "exponential"
    """Double the delay after each retry."""

    LINEAR = "linear"
    """Add the base delay after each retry."""


class RetryPolicy(BaseModel):
    """Configuration for retry behavior."""

    model_config = ConfigDict(frozen=True)

    max_retries: int = 3
    """Maximum number of retry attempts."""

    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    """Strategy for calculating delay between retries."""

    delay: float = 1.0
    """Base delay in seconds between retries."""

    max_delay: float = 30.0
    """Maximum delay in seconds (caps exponential/linear growth)."""

    retryable_exceptions: tuple[type[Exception], ...] = Field(
        default=(NodeExecutionError, ToolExecutionError, OutputValidationError)
    )
    """Exception types that should trigger retries.

    OutputValidationError is retried by default because LLM providers may
    return error payloads (e.g. content-filter messages) inside a
    successful HTTP response; the node parses them as valid JSON but they
    fail output-contract validation. Retrying gives the LLM another chance
    to produce a conforming response.
    """


class FallbackConfig(BaseModel):
    """Configuration for fallback routing."""

    model_config = ConfigDict(frozen=True)

    fallbacks: dict[str, str] = Field(default_factory=dict)
    """Mapping of node name to fallback node name."""


class ErrorHandlerConfig(BaseModel):
    """Configuration for the error handler.

    Provides global defaults that can be overridden per-node.
    """

    model_config = ConfigDict(frozen=True)

    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    """Global retry policy (default for all nodes)."""

    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    """Global circuit breaker config (default for all nodes)."""

    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    """Fallback configuration for all nodes."""

    node_retry_overrides: dict[str, RetryPolicy] = Field(default_factory=dict)
    """Per-node retry policy overrides."""

    node_circuit_breaker_overrides: dict[str, CircuitBreakerConfig] = Field(default_factory=dict)
    """Per-node circuit breaker config overrides."""


class ExecutionErrorContext(BaseModel):
    """Context information about an execution failure.

    Passed to fallback nodes so they know what failed and why.
    """

    model_config = ConfigDict(frozen=True)

    failed_node: str
    exception_type: str
    exception_message: str
    retry_count: int
    circuit_state: CircuitState
    error_chain: list[dict[str, Any]] = Field(default_factory=list)
    """Ordered list of all errors encountered during retry attempts."""


NodeExecuteFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
FallbackExecuteFn = Callable[[dict[str, Any], ExecutionErrorContext], Awaitable[dict[str, Any]]]


class ErrorHandler:
    """Manages error handling for graph node execution.

    Wraps node execution with:
    1. Circuit breaker: prevents repeatedly calling a failing node
    2. Retry: attempts to recover from transient failures
    3. Fallback: routes to alternative nodes when recovery fails

    The handler does NOT execute nodes directly. It wraps the execution
    callable provided by the agent.

    Example:
        handler = ErrorHandler(config=config, store=store)
        result = await handler.handle_node_execution(
            node_name="processor",
            execute_fn=node.execute,
            input_data={"query": "hello"},
        )
    """

    def __init__(
        self,
        config: ErrorHandlerConfig | None = None,
        store: CircuitBreakerStore | None = None,
    ) -> None:
        self._config = config or ErrorHandlerConfig()
        self._store = store or InMemoryCircuitBreakerStore(self._config.circuit_breaker)
        self._fallback_executors: dict[str, FallbackExecuteFn] = {}

    def register_fallback_executor(self, node_name: str, executor: FallbackExecuteFn) -> None:
        """Register a fallback executor for a node.

        The fallback executor receives the original input plus error context.
        """
        self._fallback_executors[node_name] = executor

    def _get_retry_policy(self, node_name: str) -> RetryPolicy:
        return self._config.node_retry_overrides.get(node_name, self._config.retry)

    def _get_circuit_breaker_config(self, node_name: str) -> CircuitBreakerConfig:
        return self._config.node_circuit_breaker_overrides.get(
            node_name, self._config.circuit_breaker
        )

    def _get_fallback_node(self, node_name: str) -> str | None:
        return self._config.fallback.fallbacks.get(node_name)

    def _calculate_delay(self, retry_policy: RetryPolicy, attempt: int) -> float:
        base_delay = retry_policy.delay
        if retry_policy.backoff_strategy == BackoffStrategy.FIXED:
            delay = base_delay
        elif retry_policy.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = base_delay * (2**attempt)
        elif retry_policy.backoff_strategy == BackoffStrategy.LINEAR:
            delay = base_delay * (attempt + 1)
        else:
            delay = base_delay
        return min(delay, retry_policy.max_delay)

    def _is_retryable(self, exception: Exception, retry_policy: RetryPolicy) -> bool:
        return isinstance(exception, retry_policy.retryable_exceptions)

    async def _execute_with_retry(
        self,
        node_name: str,
        execute_fn: NodeExecuteFn,
        input_data: dict[str, Any],
        retry_policy: RetryPolicy,
    ) -> tuple[dict[str, Any] | None, Exception | None, int, list[dict[str, Any]]]:
        last_exception: Exception | None = None
        retry_count = 0
        error_chain: list[dict[str, Any]] = []

        for attempt in range(retry_policy.max_retries + 1):
            try:
                result = await execute_fn(input_data)
                if attempt > 0:
                    logger.info(
                        "Retry succeeded",
                        node_name=node_name,
                        attempt=attempt + 1,
                        total_attempts=retry_policy.max_retries + 1,
                    )
                return result, None, retry_count, error_chain
            except Exception as e:
                last_exception = e
                retry_count = attempt
                error_chain.append(
                    {
                        "attempt": attempt + 1,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    }
                )

                if hasattr(e, "validation_errors"):
                    logger.debug(
                        "Output validation detail",
                        node_name=node_name,
                        attempt=attempt + 1,
                        schema_name=f"{getattr(e, 'schema_name', 'unknown')}",
                        validation_errors=f"{getattr(e, 'validation_errors', [])}",
                        raw_data_preview=f"{str(getattr(e, 'data', ''))[:3000]}",
                    )

                if not self._is_retryable(e, retry_policy):
                    logger.debug(
                        "Exception not retryable", node_name=node_name, error_type=type(e).__name__
                    )
                    break

                if attempt >= retry_policy.max_retries:
                    logger.warning(
                        "Max retries exhausted",
                        node_name=node_name,
                        max_retries=retry_policy.max_retries,
                        error_type=type(e).__name__,
                        error=str(e),
                    )
                    break

                delay = self._calculate_delay(retry_policy, attempt)
                logger.info(
                    "Retrying node execution",
                    node_name=node_name,
                    attempt=attempt + 1,
                    max_retries=retry_policy.max_retries,
                    delay_seconds=delay,
                    backoff_strategy=retry_policy.backoff_strategy.value,
                    error_type=type(e).__name__,
                )
                await asyncio.sleep(delay)

        return None, last_exception, retry_count, error_chain

    async def _execute_fallback(
        self,
        node_name: str,
        input_data: dict[str, Any],
        error_context: ExecutionErrorContext,
    ) -> dict[str, Any]:
        fallback_node = self._get_fallback_node(node_name)
        if fallback_node is None:
            logger.error(
                "No fallback configured for failed node",
                failed_node=node_name,
                error_type=error_context.exception_type,
            )
            raise GraphExecutionError(
                f"Node '{node_name}' failed and no fallback configured", cause=None
            )

        if fallback_node not in self._fallback_executors:
            logger.error(
                "Fallback executor not registered",
                failed_node=node_name,
                fallback_node=fallback_node,
            )
            raise GraphExecutionError(
                f"Fallback node '{fallback_node}' for '{node_name}' has no executor registered",
                cause=None,
            )

        logger.info(
            "Executing fallback",
            failed_node=node_name,
            fallback_node=fallback_node,
            error_type=error_context.exception_type,
            retry_count=error_context.retry_count,
        )

        executor = self._fallback_executors[fallback_node]
        result = await executor(input_data, error_context)

        logger.info(
            "Fallback completed successfully", failed_node=node_name, fallback_node=fallback_node
        )
        return result

    async def handle_node_execution(
        self,
        node_name: str,
        execute_fn: NodeExecuteFn,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a node with full error handling.

        Flow:
        1. Check circuit breaker state (OPEN -> fallback directly)
        2. Execute with retry logic
        3. On success: record success, return result
        4. On failure after retries: record failure, try fallback, or raise
        """
        circuit_state = await self._store.get_state(node_name)

        if circuit_state.state == CircuitState.OPEN:
            logger.warning(
                "Circuit breaker open, routing to fallback",
                node_name=node_name,
                failure_count=circuit_state.failure_count,
            )
            error_context = ExecutionErrorContext(
                failed_node=node_name,
                exception_type="CircuitOpen",
                exception_message=f"Circuit breaker is open for node '{node_name}'",
                retry_count=0,
                circuit_state=CircuitState.OPEN,
            )
            return await self._execute_fallback(node_name, input_data, error_context)

        if circuit_state.state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker half-open, testing recovery", node_name=node_name)

        retry_policy = self._get_retry_policy(node_name)
        result, exception, retry_count, error_chain = await self._execute_with_retry(
            node_name, execute_fn, input_data, retry_policy
        )

        if result is not None:
            await self._store.record_success(node_name)
            if circuit_state.state == CircuitState.HALF_OPEN:
                logger.info("Circuit breaker recovered, closing circuit", node_name=node_name)
            return result

        await self._store.record_failure(node_name)
        updated_circuit_state = await self._store.get_state(node_name)

        if (
            circuit_state.state != CircuitState.OPEN
            and updated_circuit_state.state == CircuitState.OPEN
        ):
            logger.warning(
                "Circuit breaker tripped",
                node_name=node_name,
                failure_count=updated_circuit_state.failure_count,
                threshold=self._get_circuit_breaker_config(node_name).failure_threshold,
            )

        error_context = ExecutionErrorContext(
            failed_node=node_name,
            exception_type=type(exception).__name__ if exception else "Unknown",
            exception_message=str(exception) if exception else "Unknown error",
            retry_count=retry_count,
            circuit_state=updated_circuit_state.state,
            error_chain=error_chain,
        )

        fallback_node = self._get_fallback_node(node_name)
        if fallback_node is not None:
            return await self._execute_fallback(node_name, input_data, error_context)

        logger.error(
            "Node execution failed with no fallback",
            node_name=node_name,
            retry_count=retry_count,
            error_type=error_context.exception_type,
            error=error_context.exception_message,
        )
        raise GraphExecutionError(
            f"Node '{node_name}' failed after {retry_count + 1} attempts with no fallback",
            cause=exception,
        )

    @property
    def config(self) -> ErrorHandlerConfig:
        return self._config

    @property
    def store(self) -> CircuitBreakerStore:
        return self._store

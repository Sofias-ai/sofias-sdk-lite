"""Exception hierarchy for the Sofias Agents SDK."""

from typing import Any


class AgentSDKError(Exception):
    """Base exception for all Agent SDK errors."""

    def __init__(self, message: str, **context: Any) -> None:
        self.message = message
        self.context = context
        super().__init__(message)

    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} [{context_str}]"
        return self.message


class ContractValidationError(AgentSDKError):
    """Error when data fails validation against a contract schema."""

    def __init__(
        self,
        message: str,
        *,
        schema_name: str,
        data: dict[str, Any],
        validation_errors: list[Any],
    ) -> None:
        self.schema_name = schema_name
        self.data = data
        self.validation_errors = validation_errors
        super().__init__(
            message,
            schema_name=schema_name,
            validation_errors=validation_errors,
        )


class InputValidationError(ContractValidationError):
    """Error when input data fails validation against the input contract."""

    def __init__(
        self,
        *,
        schema_name: str,
        data: dict[str, Any],
        validation_errors: list[Any],
    ) -> None:
        super().__init__(
            f"Input validation failed for schema '{schema_name}'",
            schema_name=schema_name,
            data=data,
            validation_errors=validation_errors,
        )


class OutputValidationError(ContractValidationError):
    """Error when output data fails validation against the output contract."""

    def __init__(
        self,
        *,
        schema_name: str,
        data: dict[str, Any],
        validation_errors: list[Any],
    ) -> None:
        super().__init__(
            f"Output validation failed for schema '{schema_name}'",
            schema_name=schema_name,
            data=data,
            validation_errors=validation_errors,
        )


class NodeExecutionError(AgentSDKError):
    """Error during the execution of a node."""

    def __init__(self, message: str, *, node_name: str, cause: Exception | None = None) -> None:
        self.node_name = node_name
        self.cause = cause
        super().__init__(message, node_name=node_name)
        if cause:
            self.__cause__ = cause


class RoutingError(AgentSDKError):
    """Error in routing logic between nodes."""

    def __init__(
        self, message: str, *, source_node: str, available_targets: list[str] | None = None
    ) -> None:
        self.source_node = source_node
        self.available_targets = available_targets or []
        super().__init__(
            message, source_node=source_node, available_targets=self.available_targets
        )


class ToolExecutionError(AgentSDKError):
    """Error during tool execution within a node's tool loop."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        node_name: str,
        cause: Exception | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.node_name = node_name
        self.cause = cause
        super().__init__(message, tool_name=tool_name, node_name=node_name)
        if cause:
            self.__cause__ = cause


class MaxIterationsError(AgentSDKError):
    """Error when a node's tool loop exceeds the maximum allowed iterations."""

    def __init__(self, *, node_name: str, max_iterations: int, iterations_completed: int) -> None:
        self.node_name = node_name
        self.max_iterations = max_iterations
        self.iterations_completed = iterations_completed
        super().__init__(
            f"Node '{node_name}' exceeded maximum iterations ({max_iterations})",
            node_name=node_name,
            max_iterations=max_iterations,
            iterations_completed=iterations_completed,
        )


class GraphExecutionError(AgentSDKError):
    """Error during the execution of the complete graph."""

    def __init__(
        self, message: str, *, graph_name: str | None = None, cause: Exception | None = None
    ) -> None:
        self.graph_name = graph_name
        self.cause = cause
        super().__init__(message, graph_name=graph_name)
        if cause:
            self.__cause__ = cause


class PlanExecutionError(GraphExecutionError):
    """Error during execution of a plan produced by PlannerNode.

    Raised when a step in the plan fails and the failure policy
    causes the entire plan to be cancelled.
    """

    def __init__(
        self,
        message: str,
        *,
        failed_step_id: str,
        failed_node_name: str,
        completed_steps: list[str],
        skipped_steps: list[str],
        cause: Exception | None = None,
    ) -> None:
        self.failed_step_id = failed_step_id
        self.failed_node_name = failed_node_name
        self.completed_steps = completed_steps
        self.skipped_steps = skipped_steps
        super().__init__(message, graph_name=None, cause=cause)
        self.context["failed_step_id"] = failed_step_id
        self.context["failed_node_name"] = failed_node_name
        self.context["completed_steps"] = completed_steps
        self.context["skipped_steps"] = skipped_steps


class ParallelExecutionError(GraphExecutionError):
    """Error during fan-out parallel execution of nodes.

    Raised when a parallel branch fails and the error policy
    is "fail_all", cancelling all remaining branches.
    """

    def __init__(
        self,
        message: str,
        *,
        failed_nodes: dict[str, str],
        completed_nodes: list[str],
        cause: Exception | None = None,
    ) -> None:
        self.failed_nodes = failed_nodes
        self.completed_nodes = completed_nodes
        super().__init__(message, graph_name=None, cause=cause)
        self.context["failed_nodes"] = failed_nodes
        self.context["completed_nodes"] = completed_nodes


class AggregationTimeoutError(AgentSDKError):
    """Error when aggregation does not resolve within the configured timeout.

    Raised by AggregatorNode when the resolution policy (all, any, majority)
    is not satisfied before the timeout expires and on_timeout is 'fail'.
    """

    def __init__(
        self,
        message: str,
        *,
        timeout_seconds: float,
        total_expected: int,
        total_received: int,
        pending_ids: list[str],
        received_ids: list[str],
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.total_expected = total_expected
        self.total_received = total_received
        self.pending_ids = pending_ids
        self.received_ids = received_ids
        super().__init__(
            message,
            timeout_seconds=timeout_seconds,
            total_expected=total_expected,
            total_received=total_received,
            pending_ids=pending_ids,
            received_ids=received_ids,
        )


class DelegationError(AgentSDKError):
    """Error during delegation to another agent.

    Raised when a DelegationNode fails to delegate work to a target agent.
    This can happen due to publishing errors, invalid responses, or
    response validation failures.
    """

    def __init__(
        self,
        message: str,
        *,
        target_agent: str,
        correlation_id: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.target_agent = target_agent
        self.correlation_id = correlation_id
        self.cause = cause
        super().__init__(
            message,
            target_agent=target_agent,
            correlation_id=correlation_id,
        )
        if cause:
            self.__cause__ = cause


class DelegationTimeoutError(DelegationError):
    """Error when a delegation target does not respond within the timeout.

    For parallel delegation, includes information about which agents
    did not respond in the pending_agents field.
    """

    def __init__(
        self,
        message: str,
        *,
        target_agent: str,
        correlation_id: str | None = None,
        timeout_seconds: int,
        pending_agents: list[str] | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.pending_agents = pending_agents or []
        super().__init__(
            message,
            target_agent=target_agent,
            correlation_id=correlation_id,
            cause=cause,
        )
        self.context["timeout_seconds"] = timeout_seconds
        if self.pending_agents:
            self.context["pending_agents"] = self.pending_agents


class DelegationStepError(DelegationError):
    """Error when an individual step in a DAG delegation fails.

    Contains information about the failed step and the original error
    that caused the failure.
    """

    def __init__(
        self,
        message: str,
        *,
        step_id: str,
        target_agent: str,
        original_error: Exception | None = None,
        correlation_id: str | None = None,
    ) -> None:
        self.step_id = step_id
        self.original_error = original_error
        super().__init__(
            message,
            target_agent=target_agent,
            correlation_id=correlation_id,
            cause=original_error,
        )
        self.context["step_id"] = step_id


class DelegationPlanError(DelegationError):
    """Error when a complete delegation plan fails due to fail_plan policy.

    Raised when a step fails and the failure policy is 'fail_plan',
    causing the entire plan to be cancelled. Contains information about
    which steps completed, which failed, and which were skipped.
    """

    def __init__(
        self,
        message: str,
        *,
        failed_step_id: str,
        target_agent: str,
        completed_steps: list[str],
        skipped_steps: list[str],
        correlation_id: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.failed_step_id = failed_step_id
        self.completed_steps = completed_steps
        self.skipped_steps = skipped_steps
        super().__init__(
            message,
            target_agent=target_agent,
            correlation_id=correlation_id,
            cause=cause,
        )
        self.context["failed_step_id"] = failed_step_id
        self.context["completed_steps"] = completed_steps
        self.context["skipped_steps"] = skipped_steps


class MCPConnectionError(AgentSDKError):
    """Error when connecting to an MCP server fails."""

    def __init__(
        self,
        message: str,
        *,
        server_name: str,
        url: str,
        cause: Exception | None = None,
    ) -> None:
        self.server_name = server_name
        self.url = url
        self.cause = cause
        super().__init__(message, server_name=server_name, url=url)
        if cause:
            self.__cause__ = cause


class MCPToolExecutionError(AgentSDKError):
    """Error when executing a tool via MCP fails."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        server_name: str,
        cause: Exception | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.server_name = server_name
        self.cause = cause
        super().__init__(message, tool_name=tool_name, server_name=server_name)
        if cause:
            self.__cause__ = cause


class ConfigSourceError(AgentSDKError):
    """Error when a `ConfigSource` fails to supply runtime configuration.

    Raised when a configured source cannot produce a valid configuration
    payload for an agent — for example, an environment variable that is
    unset or does not contain valid JSON.
    """

    def __init__(
        self,
        message: str,
        *,
        source: str,
        agent_name: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.source = source
        self.agent_name = agent_name
        self.cause = cause
        super().__init__(message, source=source, agent_name=agent_name)
        if cause:
            self.__cause__ = cause


class LLMConfigurationError(AgentSDKError):
    """Error when no usable LLM gateway configuration can be resolved.

    Raised by `sofias_sdk_lite.llm.create_llm` when neither the explicit
    arguments, the agent settings, nor the environment provide enough
    information (at minimum a model identifier) to build an LLM client.
    """

    def __init__(self, message: str, *, missing: list[str] | None = None) -> None:
        self.missing = missing or []
        super().__init__(message, missing=self.missing)


class LLMRequestError(AgentSDKError):
    """Error when an LLM gateway request fails after exhausting retries.

    Carries the HTTP status (``None`` for transport-level failures) and a
    truncated response body so callers can log or branch on it.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str,
        status_code: int | None = None,
        body: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.body = body
        self.cause = cause
        super().__init__(
            message,
            provider=provider,
            model=model,
            status_code=status_code,
        )


class EmptyLLMResponseError(AgentSDKError):
    """Error when the LLM returns neither text content nor tool calls.

    `LLMNode` treats this like any other invocation failure and retries the
    call, so a transient empty completion does not end the turn.
    """

    def __init__(self, *, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        super().__init__(
            "LLM returned empty response (no content, no tool_calls)",
            provider=provider,
            model=model,
        )

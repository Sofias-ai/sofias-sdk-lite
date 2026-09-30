"""Agent implementation for graph orchestration.

This module provides the Agent class, which is the central orchestrator
for executing agent graphs. The agent manages node execution, routing,
error handling, and context propagation.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sofias_sdk_lite.config.runtime_config import AgentSettings, SettingsResolver
from sofias_sdk_lite.agent.execution_context import (
    ExecutionContext,
    get_execution_context,
    set_execution_context,
)
from sofias_sdk_lite.contracts import AgentContract
from sofias_sdk_lite.contracts.agent_contracts import (
    AgentMessage,
    AgentResponse,
    ResponseStatus,
)
from sofias_sdk_lite.errors.exceptions import (
    AgentSDKError,
    GraphExecutionError,
    ParallelExecutionError,
    PlanExecutionError,
    RoutingError,
)
from sofias_sdk_lite.nodes.base_node import NodeType
from sofias_sdk_lite.nodes.planning_models import ExecutionPlan
from sofias_sdk_lite.observability._log import get_logger

if TYPE_CHECKING:
    from sofias_sdk_lite.agent.response_workflow import (
        ResponseWorkflow,
        StreamingResponseWorkflow,
    )
    from sofias_sdk_lite.errors.error_handler import ErrorHandler, ExecutionErrorContext
    from sofias_sdk_lite.nodes.base_node import BaseNode as Node
    from sofias_sdk_lite.nodes.plan_executor import PlanExecutor
    from sofias_sdk_lite.routing.router import Router
    from sofias_sdk_lite.agent.streaming import StreamEvent

__all__ = ["Agent"]

logger = get_logger(__name__)


class Agent:
    """The complete agent graph ready for execution.

    The Agent is a pure orchestrator. It does not contain business logic.
    Its responsibilities are:
    1. Receive input and resolve runtime configuration
    2. Traverse the graph executing nodes according to routing decisions
    3. Manage errors through the error handler
    4. Produce validated output

    The agent does NOT execute LLM calls or tools directly. It delegates
    to Node instances which handle their own execution.

    Example:
        # Agent is typically built via AgentBuilder
        agent = (
            AgentBuilder("my_agent", version="1.0.0")
            .with_settings_class(MySettings)
            .with_contract(input_schema=MyInput, output_schema=MyOutput)
            .add_node("start", node_config, contract)
            .set_entry_node("start")
            .set_terminal("start")
            .build()
        )

        # Execute the agent
        response = await agent.execute(
            AgentMessage(
                content=MyInput(query="Hello"),
                runtime_config={"model": "gpt-4o"},
            )
        )
        assert response.status == ResponseStatus.SUCCESS
    """

    def __init__(
        self,
        *,
        name: str,
        version: str,
        description: str | None,
        nodes: dict[str, Node],
        router: Router,
        error_handler: ErrorHandler,
        contract: AgentContract,
        entry_node: str,
        settings_resolver: SettingsResolver[AgentSettings],
        context: dict[str, Any] | None = None,
        response_workflow: ResponseWorkflow | None = None,
        plan_executor: PlanExecutor | None = None,
        middleware: list[Any] | None = None,
    ) -> None:
        """Initialize the agent.

        This constructor is typically called by AgentBuilder.build(),
        not directly by users.

        Args:
            name: Unique name identifying this agent.
            version: Semantic version of the agent.
            description: Human-readable description.
            nodes: Dictionary mapping node names to Node instances.
            router: Router managing graph edges and routing decisions.
            error_handler: ErrorHandler for retry, fallback, circuit breaker.
            contract: Contract defining agent input/output schemas.
            entry_node: Name of the node where execution starts.
            settings_resolver: Resolver for runtime settings.
            context: Read-only context shared with all nodes.
            response_workflow: Optional workflow for delivering the response.
            plan_executor: Optional executor for plans produced by PlannerNodes.
            middleware: Optional list of AgentMiddleware for before/after node hooks.
        """
        self._name = name
        self._version = version
        self._description = description
        self._nodes = nodes
        self._router = router
        self._error_handler = error_handler
        self._contract = contract
        self._entry_node = entry_node
        self._settings_resolver = settings_resolver
        self._context = context or {}
        self._response_workflow = response_workflow
        self._plan_executor = plan_executor
        self._middleware = middleware or []

    @property
    def name(self) -> str:
        """The agent's name."""
        return self._name

    @property
    def version(self) -> str:
        """The agent's version."""
        return self._version

    @property
    def description(self) -> str | None:
        """The agent's description."""
        return self._description

    @property
    def entry_node(self) -> str:
        """The name of the entry node."""
        return self._entry_node

    @property
    def context(self) -> dict[str, Any]:
        """The read-only context shared with all nodes."""
        return dict(self._context)

    async def execute(self, message: AgentMessage) -> AgentResponse:
        """Execute the agent graph with the given input message.

        This is the main entry point for agent execution. The flow is:
        1. Resolve runtime config from message and set in context
        2. Validate message content against agent contract
        3. Execute graph nodes following routing decisions
        4. Validate final output against agent contract
        5. Build AgentResponse with execution metadata
        6. Clean up context and return result

        For controlled business errors (validation failures, routing errors,
        node execution errors handled by the error handler), returns an
        AgentResponse with status=ERROR and error_message describing the
        problem. Infrastructure failures (unexpected exceptions not derived
        from AgentSDKError) are re-raised.

        Args:
            message: Typed input envelope containing domain data and metadata.

        Returns:
            AgentResponse with validated output and execution metadata.

        Raises:
            Exception: Only for infrastructure failures not covered by
                AgentSDKError (unexpected crashes, connectivity issues, etc.).
        """
        run_id = str(uuid.uuid4())
        start_time = time.monotonic()

        # Create typed execution context — available immediately for
        # error handlers and nodes via get_execution_context()
        exec_ctx = ExecutionContext(
            execution_id=run_id,
            agent_name=self._name,
            agent_version=self._version,
            trace_id=message.trace_id,
            started_at=datetime.now(timezone.utc),
            metadata={
                **self._context,
                "conversation_id": message.conversation_id or "",
                "tenant_identifier": message.metadata.get("tenant_identifier", ""),
            },
        )
        set_execution_context(exec_ctx)

        logger.info(
            "Agent execution started",
            run_id=run_id,
            agent_name=self._name,
            agent_version=self._version,
            entry_node=self._entry_node,
            total_nodes=len(self._nodes),
        )

        response: AgentResponse | None = None

        try:
            # Step 1: Resolve runtime config
            if message.runtime_config is not None:
                self._settings_resolver.resolve_and_set(message.runtime_config)
            else:
                self._settings_resolver.set_current(
                    self._settings_resolver.settings_class()
                )

            # Step 2: Validate input content against agent contract
            validated_input = self._contract.validate_input(
                message.content.model_dump()
            )
            current_data = validated_input.model_dump()
            logger.debug("Input validation passed", run_id=run_id)

            # Step 3: Execute the graph
            current_node_name = self._entry_node

            while current_node_name is not None:
                exec_ctx.current_node = current_node_name
                exec_ctx.transition_history.append(current_node_name)

                # Get the node
                node = self._nodes.get(current_node_name)
                if node is None:
                    logger.error(
                        "Node not found", run_id=run_id, node_name=current_node_name
                    )
                    raise GraphExecutionError(
                        f"Node '{current_node_name}' not found in agent graph",
                        graph_name=self._name,
                    )

                # Run before_node middleware
                for mw in self._middleware:
                    if hasattr(mw, "before_node"):
                        modified = await mw.before_node(current_node_name, current_data)
                        if modified is not None:
                            current_data = modified

                # Execute node with error handling
                try:
                    current_data = await self._error_handler.handle_node_execution(
                        node_name=current_node_name,
                        execute_fn=node.execute,
                        input_data=current_data,
                    )
                except GraphExecutionError as e:
                    # Run on_error middleware
                    for mw in self._middleware:
                        if hasattr(mw, "on_error"):
                            await mw.on_error(current_node_name, e)
                    logger.error(
                        "Graph execution failed",
                        run_id=run_id,
                        failed_node=current_node_name,
                        error=str(e),
                        execution_path=exec_ctx.transition_history,
                    )
                    raise GraphExecutionError(
                        f"Graph execution failed at node '{current_node_name}': {e.message}",
                        graph_name=self._name,
                        cause=e.cause,
                    ) from e

                # Run after_node middleware
                for mw in self._middleware:
                    if hasattr(mw, "after_node"):
                        modified = await mw.after_node(current_node_name, current_data, current_data)
                        if modified is not None:
                            current_data = modified

                # Plan intercept: detect PlannerNode by TYPE, not by output
                if (
                    node.node_type == NodeType.PLANNER
                    and self._plan_executor is not None
                ):
                    plan = ExecutionPlan.model_validate(current_data)

                    validation_errors = self._plan_executor.validate_plan(plan)
                    if validation_errors:
                        raise GraphExecutionError(
                            f"Plan validation failed: {'; '.join(validation_errors)}",
                            graph_name=self._name,
                        )

                    plan_result = await self._plan_executor.execute(
                        plan, exec_ctx
                    )
                    current_data = plan_result.model_dump()

                # Resolve next node (may return str, list[str], or None)
                try:
                    next_target = self._router.resolve(current_node_name, current_data)
                except RoutingError as e:
                    logger.error(
                        "Routing failed",
                        run_id=run_id,
                        source_node=current_node_name,
                        error=str(e),
                    )
                    raise RoutingError(
                        f"Routing failed from node '{current_node_name}' in agent '{self._name}': {e.message}",
                        source_node=current_node_name,
                        available_targets=e.available_targets,
                    ) from e

                if isinstance(next_target, list):
                    # Fan-out: execute parallel branches
                    fan_out_config = self._router.get_fan_out_config(current_node_name)
                    parallel_results = await self._execute_fan_out(
                        node_names=next_target,
                        input_data=current_data,
                        exec_ctx=exec_ctx,
                        on_error=fan_out_config.on_error if fan_out_config else "fail_all",
                        timeout_seconds=fan_out_config.timeout_seconds if fan_out_config else 300,
                    )
                    current_data = {
                        **current_data,
                        "parallel_results": parallel_results,
                    }
                    current_node_name = fan_out_config.join_node if fan_out_config else None
                else:
                    current_node_name = next_target

            # Step 4: Validate output
            exec_ctx.current_node = None
            validated_output = self._contract.validate_output(current_data)
            elapsed_ms = (time.monotonic() - start_time) * 1000

            logger.info(
                "Agent execution completed",
                run_id=run_id,
                execution_path=exec_ctx.transition_history,
                nodes_executed=len(exec_ctx.transition_history),
                execution_time_ms=elapsed_ms,
            )

            # Step 5: Build success response
            token_data = exec_ctx.token_accumulator.to_dict()
            usage_list = exec_ctx.token_accumulator.to_usage_list()
            response = AgentResponse(
                content=validated_output.model_dump(),
                status=ResponseStatus.SUCCESS,
                agent_name=self._name,
                agent_version=self._version,
                execution_path=exec_ctx.transition_history,
                execution_time_ms=elapsed_ms,
                metadata={
                    "run_id": run_id,
                    "trace_id": message.trace_id,
                },
                usage=usage_list,
                **token_data,  # type: ignore[arg-type]
            )

        except AgentSDKError as e:
            # Business errors — build structured error response
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Agent execution failed (business error)",
                run_id=run_id,
                error_type=type(e).__name__,
                error=str(e),
                execution_path=exec_ctx.transition_history,
            )
            token_data = exec_ctx.token_accumulator.to_dict()
            usage_list = exec_ctx.token_accumulator.to_usage_list()
            response = AgentResponse(
                content={},
                status=ResponseStatus.ERROR,
                agent_name=self._name,
                agent_version=self._version,
                execution_path=exec_ctx.transition_history,
                error_message=str(e),
                execution_time_ms=elapsed_ms,
                metadata={
                    "run_id": run_id,
                    "trace_id": message.trace_id,
                    "error_type": type(e).__name__,
                },
                usage=usage_list,
                **token_data,  # type: ignore[arg-type]
            )

        except Exception as e:
            # Infrastructure errors — re-raise
            logger.error(
                "Agent execution failed (infrastructure error)",
                run_id=run_id,
                error_type=type(e).__name__,
                error=str(e),
            )
            raise

        finally:
            # Step 6: Deliver response via workflow (before cleanup)
            if response is not None and self._response_workflow is not None:
                try:
                    await self._response_workflow.send_response(response, exec_ctx)
                except Exception as wf_err:
                    logger.error(
                        "Response workflow failed",
                        run_id=run_id,
                        error_type=type(wf_err).__name__,
                        error=str(wf_err),
                    )

            # Step 7: Clean up context
            self._settings_resolver.clear()
            set_execution_context(None)

        assert response is not None  # guaranteed by try/except structure
        return response

    async def stream_execute(
        self, message: AgentMessage,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute the agent graph with streaming.

        Same flow as execute() but yields StreamEvent instances in real time.
        For LLMNodes, text chunks are yielded as they arrive from the LLM.
        For non-streaming nodes, only NodeStart/NodeComplete events are emitted.

        If a StreamingResponseWorkflow is configured, on_stream_event() is
        called for each event. send_response() is called at the end.

        Args:
            message: Typed input envelope containing domain data and metadata.

        Yields:
            StreamEvent instances for real-time visibility into execution.
        """
        from sofias_sdk_lite.agent.response_workflow import StreamingResponseWorkflow
        from sofias_sdk_lite.agent.streaming import (
            AgentCompleteEvent,
            AgentErrorEvent,
            NodeCompleteEvent,
            StreamEvent,
            TextChunkEvent,
        )

        run_id = str(uuid.uuid4())
        start_time = time.monotonic()

        exec_ctx = ExecutionContext(
            execution_id=run_id,
            agent_name=self._name,
            agent_version=self._version,
            trace_id=message.trace_id,
            started_at=datetime.now(timezone.utc),
            metadata={
                **self._context,
                "conversation_id": message.conversation_id or "",
                "tenant_identifier": message.metadata.get("tenant_identifier", ""),
            },
        )
        set_execution_context(exec_ctx)

        logger.info(
            "Agent streaming execution started",
            run_id=run_id,
            agent_name=self._name,
            agent_version=self._version,
            entry_node=self._entry_node,
            total_nodes=len(self._nodes),
        )

        streaming_workflow: StreamingResponseWorkflow | None = (
            self._response_workflow
            if isinstance(self._response_workflow, StreamingResponseWorkflow)
            else None
        )
        response: AgentResponse | None = None

        try:
            # Step 1: Resolve runtime config
            if message.runtime_config is not None:
                self._settings_resolver.resolve_and_set(message.runtime_config)
            else:
                self._settings_resolver.set_current(
                    self._settings_resolver.settings_class()
                )

            # Step 2: Validate input
            validated_input = self._contract.validate_input(
                message.content.model_dump()
            )
            current_data = validated_input.model_dump()
            logger.debug("Input validation passed", run_id=run_id)

            # Step 3: Execute graph with streaming
            current_node_name = self._entry_node

            while current_node_name is not None:
                exec_ctx.current_node = current_node_name
                exec_ctx.transition_history.append(current_node_name)

                node = self._nodes.get(current_node_name)
                if node is None:
                    logger.error(
                        "Node not found", run_id=run_id, node_name=current_node_name
                    )
                    raise GraphExecutionError(
                        f"Node '{current_node_name}' not found in agent graph",
                        graph_name=self._name,
                    )

                # Run before_node middleware
                for mw in self._middleware:
                    if hasattr(mw, "before_node"):
                        modified = await mw.before_node(current_node_name, current_data)
                        if modified is not None:
                            current_data = modified

                # Stream through the node
                is_terminal = current_node_name in self._router.terminals
                node_output = None
                async for event in node.stream_execute(current_data):
                    if streaming_workflow is not None:
                        # Only forward TextChunkEvents from terminal nodes to
                        # the workflow — intermediate LLMNode text (e.g. planner
                        # JSON) must not leak to the user stream.
                        if is_terminal or not isinstance(event, TextChunkEvent):
                            await streaming_workflow.on_stream_event(event, exec_ctx)
                    yield event
                    if isinstance(event, NodeCompleteEvent):
                        node_output = event.output

                if node_output is None:
                    raise GraphExecutionError(
                        f"Node '{current_node_name}' produced no output",
                        graph_name=self._name,
                    )
                current_data = node_output

                # Run after_node middleware
                for mw in self._middleware:
                    if hasattr(mw, "after_node"):
                        modified = await mw.after_node(current_node_name, current_data, current_data)
                        if modified is not None:
                            current_data = modified

                # Plan intercept
                if (
                    node.node_type == NodeType.PLANNER
                    and self._plan_executor is not None
                ):
                    plan = ExecutionPlan.model_validate(current_data)
                    validation_errors = self._plan_executor.validate_plan(plan)
                    if validation_errors:
                        raise GraphExecutionError(
                            f"Plan validation failed: {'; '.join(validation_errors)}",
                            graph_name=self._name,
                        )
                    plan_result = await self._plan_executor.execute(plan, exec_ctx)
                    current_data = plan_result.model_dump()

                # Resolve next node (may return str, list[str], or None)
                try:
                    next_target = self._router.resolve(current_node_name, current_data)
                except RoutingError as e:
                    logger.error(
                        "Routing failed",
                        run_id=run_id,
                        source_node=current_node_name,
                        error=str(e),
                    )
                    raise RoutingError(
                        f"Routing failed from node '{current_node_name}' in agent '{self._name}': {e.message}",
                        source_node=current_node_name,
                        available_targets=e.available_targets,
                    ) from e

                if isinstance(next_target, list):
                    # Fan-out: execute parallel branches
                    fan_out_config = self._router.get_fan_out_config(current_node_name)
                    parallel_results = await self._execute_fan_out(
                        node_names=next_target,
                        input_data=current_data,
                        exec_ctx=exec_ctx,
                        on_error=fan_out_config.on_error if fan_out_config else "fail_all",
                        timeout_seconds=fan_out_config.timeout_seconds if fan_out_config else 300,
                    )
                    current_data = {
                        **current_data,
                        "parallel_results": parallel_results,
                    }
                    current_node_name = fan_out_config.join_node if fan_out_config else None
                else:
                    current_node_name = next_target

            # Step 4: Validate output
            exec_ctx.current_node = None
            validated_output = self._contract.validate_output(current_data)
            elapsed_ms = (time.monotonic() - start_time) * 1000

            logger.info(
                "Agent streaming execution completed",
                run_id=run_id,
                execution_path=exec_ctx.transition_history,
                nodes_executed=len(exec_ctx.transition_history),
                execution_time_ms=elapsed_ms,
            )

            # Yield agent complete
            token_data = exec_ctx.token_accumulator.to_dict()
            usage_list = exec_ctx.token_accumulator.to_usage_list()
            complete_event = AgentCompleteEvent(
                content=validated_output.model_dump(),
                execution_path=exec_ctx.transition_history,
                execution_time_ms=elapsed_ms,
                usage=usage_list,
                **token_data,  # type: ignore[arg-type]
            )
            if streaming_workflow is not None:
                await streaming_workflow.on_stream_event(complete_event, exec_ctx)
            yield complete_event

            # Build and send final response
            response = AgentResponse(
                content=validated_output.model_dump(),
                status=ResponseStatus.SUCCESS,
                agent_name=self._name,
                agent_version=self._version,
                execution_path=exec_ctx.transition_history,
                execution_time_ms=elapsed_ms,
                metadata={
                    "run_id": run_id,
                    "trace_id": message.trace_id,
                },
                usage=usage_list,
                **token_data,  # type: ignore[arg-type]
            )

        except AgentSDKError as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Agent streaming execution failed (business error)",
                run_id=run_id,
                error_type=type(e).__name__,
                error=str(e),
                execution_path=exec_ctx.transition_history,
            )
            error_event = AgentErrorEvent(
                error_type=type(e).__name__,
                error_message=str(e),
                execution_path=exec_ctx.transition_history,
            )
            if streaming_workflow is not None:
                await streaming_workflow.on_stream_event(error_event, exec_ctx)
            yield error_event

            token_data = exec_ctx.token_accumulator.to_dict()
            usage_list = exec_ctx.token_accumulator.to_usage_list()
            response = AgentResponse(
                content={},
                status=ResponseStatus.ERROR,
                agent_name=self._name,
                agent_version=self._version,
                execution_path=exec_ctx.transition_history,
                error_message=str(e),
                execution_time_ms=elapsed_ms,
                metadata={
                    "run_id": run_id,
                    "trace_id": message.trace_id,
                    "error_type": type(e).__name__,
                },
                usage=usage_list,
                **token_data,  # type: ignore[arg-type]
            )

        except Exception as e:
            logger.error(
                "Agent streaming execution failed (infrastructure error)",
                run_id=run_id,
                error_type=type(e).__name__,
                error=str(e),
            )
            raise

        finally:
            if response is not None and self._response_workflow is not None:
                try:
                    await self._response_workflow.send_response(response, exec_ctx)
                except Exception as wf_err:
                    logger.error(
                        "Response workflow failed",
                        run_id=run_id,
                        error_type=type(wf_err).__name__,
                        error=str(wf_err),
                    )

            self._settings_resolver.clear()
            set_execution_context(None)

    async def _execute_fan_out(
        self,
        node_names: list[str],
        input_data: dict[str, Any],
        exec_ctx: ExecutionContext,
        on_error: str = "fail_all",
        timeout_seconds: int = 300,
    ) -> dict[str, dict[str, Any]]:
        """Execute multiple nodes in parallel (fan-out).

        Each node receives a copy of the input data. Results are collected
        into a dictionary keyed by node name.

        Args:
            node_names: Names of nodes to execute concurrently.
            input_data: Input data (copied to each branch).
            exec_ctx: Execution context for tracking.
            on_error: "fail_all" cancels remaining on first failure,
                "continue_partial" waits for all and collects what succeeded.
            timeout_seconds: Maximum time for all branches.

        Returns:
            Dictionary mapping node name to its output.

        Raises:
            ParallelExecutionError: If a branch fails under "fail_all" policy.
            GraphExecutionError: If a node is not found.
        """
        logger.info(
            "Fan-out execution started",
            targets=node_names,
            on_error=on_error,
            timeout_seconds=timeout_seconds,
        )

        # Validate all nodes exist
        for name in node_names:
            if name not in self._nodes:
                raise GraphExecutionError(
                    f"Fan-out target node '{name}' not found in agent graph",
                    graph_name=self._name,
                )

        # Track in execution context
        exec_ctx.transition_history.append(
            f"fan_out:[{','.join(node_names)}]"
        )

        # Launch all branches concurrently
        running_tasks: dict[asyncio.Task[dict[str, Any]], str] = {}
        for name in node_names:
            node = self._nodes[name]
            branch_input = dict(input_data)  # shallow copy per branch

            async def _run_branch(
                n: str,
                nd: Any,
                inp: dict[str, Any],
            ) -> dict[str, Any]:
                return await self._error_handler.handle_node_execution(
                    node_name=n,
                    execute_fn=nd.execute,
                    input_data=inp,
                )

            task = asyncio.create_task(
                _run_branch(name, node, branch_input),
                name=f"fan_out_{name}",
            )
            running_tasks[task] = name

        results: dict[str, dict[str, Any]] = {}
        failed: dict[str, str] = {}

        try:
            done, pending = await asyncio.wait(
                running_tasks.keys(),
                timeout=timeout_seconds,
                return_when=(
                    asyncio.FIRST_EXCEPTION
                    if on_error == "fail_all"
                    else asyncio.ALL_COMPLETED
                ),
            )

            # Process completed tasks
            for task in done:
                name = running_tasks[task]
                try:
                    results[name] = task.result()
                    exec_ctx.transition_history.append(name)
                except Exception as e:
                    failed[name] = str(e)
                    if on_error == "fail_all":
                        # Cancel remaining tasks
                        for p in pending:
                            p.cancel()
                        raise ParallelExecutionError(
                            f"Fan-out branch '{name}' failed: {e}",
                            failed_nodes=failed,
                            completed_nodes=list(results.keys()),
                            cause=e,
                        ) from e

            # Handle pending (timed out) tasks
            if pending:
                for task in pending:
                    task.cancel()
                timed_out = [running_tasks[t] for t in pending]
                if on_error == "fail_all":
                    raise ParallelExecutionError(
                        f"Fan-out timed out after {timeout_seconds}s. "
                        f"Pending: {timed_out}",
                        failed_nodes={n: "timeout" for n in timed_out},
                        completed_nodes=list(results.keys()),
                    )
                for name in timed_out:
                    failed[name] = "timeout"

        except ParallelExecutionError:
            raise
        except Exception as e:
            # Cancel any remaining tasks on unexpected error
            for task in running_tasks:
                if not task.done():
                    task.cancel()
            raise GraphExecutionError(
                f"Unexpected error during fan-out execution: {e}",
                graph_name=self._name,
                cause=e,
            ) from e

        logger.info(
            "Fan-out execution completed",
            completed=list(results.keys()),
            failed=list(failed.keys()) if failed else None,
        )

        return results

    def get_graph_info(self) -> dict[str, Any]:
        """Get information about the graph structure.

        Useful for debugging, logging, and observability.

        Returns:
            Dictionary containing:
            - name: Agent name
            - version: Agent version
            - description: Agent description
            - entry_node: Name of the entry node
            - nodes: List of all node names
            - terminals: Set of terminal node names
            - routes: Dictionary of source node to target nodes
            - fallbacks: Dictionary of node to fallback node mappings
        """
        # Extract route information
        routes: dict[str, list[str]] = {}
        for source, strategy in self._router.routes.items():
            targets = self._router._extract_strategy_targets(strategy)
            routes[source] = targets

        # Extract fallback information from error handler config
        fallbacks = dict(self._error_handler.config.fallback.fallbacks)

        return {
            "name": self._name,
            "version": self._version,
            "description": self._description,
            "entry_node": self._entry_node,
            "nodes": list(self._nodes.keys()),
            "terminals": list(self._router.terminals),
            "routes": routes,
            "fallbacks": fallbacks,
        }

    def __repr__(self) -> str:
        """Return a string representation of the agent."""
        return f"Agent(name={self._name!r}, version={self._version!r}, nodes={len(self._nodes)})"

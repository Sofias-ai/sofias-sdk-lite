"""DelegationNode for delegating tasks to other agents.

This module provides the DelegationNode class, which allows an agent to
delegate work to other agents in the ecosystem. Supports sequential,
parallel, and DAG execution modes.

The transport layer is abstracted via the DelegationTransport protocol,
allowing different implementations (RabbitMQ, HTTP, in-memory for testing).
The concrete RabbitMQ implementation lives in `sofias_sdk_lite.rabbitmq` and
is never imported from this module.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, Literal

from sofias_sdk_lite.contracts import DelegationStatus
from sofias_sdk_lite.errors import (
    DelegationError,
    DelegationPlanError,
    DelegationStepError,
    DelegationTimeoutError,
)
from sofias_sdk_lite.nodes.base_node import BaseNode, NodeType
from sofias_sdk_lite.nodes.delegation_config import (
    DelegationNodeConfig,
    DelegationPlan,
    DelegationStep,
    DelegationTarget,
)
from sofias_sdk_lite.observability._log import get_logger

if TYPE_CHECKING:
    from sofias_sdk_lite.config import SettingsResolver
    from sofias_sdk_lite.contracts import NodeContract
    from sofias_sdk_lite.nodes.delegation_transport import DelegationTransport

logger = get_logger("nodes.delegation_node")

__all__ = ["DelegationNode", "InputMapper"]

# Type alias for input mapper functions
InputMapper = Callable[[dict[str, Any]], dict[str, Any]]


class DelegationNode(BaseNode):
    """Node that delegates tasks to other agents.

    DelegationNode sends tasks to one or more target agents and waits for
    their responses. It supports three execution modes:

    - Sequential: Delegates to targets one at a time, in order.
    - Parallel: Delegates to all targets simultaneously and waits for all responses.
    - DAG: Executes steps respecting dependencies, with configurable failure policies.

    The node uses a DelegationTransport for communication, which handles
    the underlying message passing (RabbitMQ, HTTP, etc.).

    Example (static targets):
        config = DelegationNodeConfig(
            name="delegate_to_summarizer",
            targets=[DelegationTarget(agent_name="summarizer")],
        )

        node = DelegationNode(
            config=config,
            contract=my_contract,
            transport=my_transport,
        )

        result = await node.execute({"text": "Long document..."})
        # result = {"summarizer": {"summary": "..."}}

    Example (DAG with dynamic targets):
        config = DelegationNodeConfig(
            name="dag_executor",
            execution_mode="dag",
            dynamic_targets=True,
        )

        plan = {
            "steps": [
                {"id": "s1", "agent_name": "search", "input": {"q": "A"}},
                {"id": "s2", "agent_name": "search", "input": {"q": "B"}},
                {"id": "s3", "agent_name": "synth", "input": {}, "depends_on": ["s1", "s2"]},
            ]
        }

        result = await node.execute(plan)
        # result = {"results": {...}, "failed": {...}, "skipped": [...], "all_completed": bool}
    """

    def __init__(
        self,
        config: DelegationNodeConfig,
        contract: NodeContract,
        transport: DelegationTransport,
        input_mappers: dict[str, InputMapper] | None = None,
        source_agent_name: str | None = None,
        *,
        settings_resolver: SettingsResolver | None = None,
    ) -> None:
        """Initialize the DelegationNode.

        Args:
            config: Configuration for the delegation node.
            contract: Input/output contract for validation.
            transport: Transport for sending requests and receiving responses.
            input_mappers: Optional dict mapping mapper names to functions.
                DelegationTargets reference these by name via input_mapping.
            source_agent_name: Name of the agent that owns this node.
                Used in metadata for tracing.
            settings_resolver: Optional resolver for runtime settings.
                When present, agent_timeout from AgentSettings overrides
                per-target timeout_seconds.
        """
        super().__init__(
            name=config.name,
            contract=contract,
            description=config.description,
        )
        self._config = config
        self._transport = transport
        self._input_mappers = input_mappers or {}
        self._source_agent_name = source_agent_name
        self._settings_resolver = settings_resolver

        # Validate that all referenced mappers exist (only for static targets)
        if not config.dynamic_targets:
            for target in config.targets:
                if target.input_mapping and target.input_mapping not in self._input_mappers:
                    raise ValueError(
                        f"DelegationTarget '{target.agent_name}' references mapper "
                        f"'{target.input_mapping}' which is not registered"
                    )

    @property
    def node_type(self) -> NodeType:
        return NodeType.DELEGATION

    @property
    def config(self) -> DelegationNodeConfig:
        """The node's configuration."""
        return self._config

    def _apply_input_mapping(
        self,
        target: DelegationTarget,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply input mapping transformation if configured.

        Args:
            target: The delegation target.
            input_data: Original input data.

        Returns:
            Transformed input data, or original if no mapping.
        """
        if target.input_mapping is None:
            return input_data

        mapper = self._input_mappers.get(target.input_mapping)
        if mapper is None:
            raise DelegationError(
                f"Input mapper '{target.input_mapping}' not found",
                target_agent=target.agent_name,
            )

        try:
            return mapper(input_data)
        except Exception as e:
            raise DelegationError(
                f"Input mapping failed for target '{target.agent_name}': {e}",
                target_agent=target.agent_name,
                cause=e,
            ) from e

    def _resolve_target_timeout(self, target: DelegationTarget) -> int:
        """Resolve timeout for a target: agent_timeout (runtime) > target.timeout_seconds (static).

        Args:
            target: The delegation target.

        Returns:
            Timeout in seconds.
        """
        if self._settings_resolver is not None:
            settings = self._settings_resolver.get_current()
            if settings is not None and settings.agent_timeout is not None:
                return int(settings.agent_timeout)
        return target.timeout_seconds

    def _get_context_fields(self) -> tuple[str, str]:
        """Extract conversation_id and tenant_identifier from ExecutionContext.

        Returns:
            Tuple of (tenant_identifier, conversation_id).
        """
        from sofias_sdk_lite.agent import get_execution_context

        exec_ctx = get_execution_context()
        if exec_ctx is not None:
            return (
                exec_ctx.metadata.get("tenant_identifier", ""),
                exec_ctx.metadata.get("conversation_id", ""),
            )
        return ("", "")

    def _build_metadata(
        self,
        target_agent: str,
        context: dict[str, Any] | None = None,
        step_id: str | None = None,
    ) -> dict[str, Any]:
        """Build metadata for a delegation request.

        Args:
            target_agent: Name of the target agent.
            context: Optional execution context.
            step_id: Optional step ID for DAG execution.

        Returns:
            Metadata dictionary.
        """
        metadata = {
            "source_agent": self._source_agent_name or self._config.name,
            "target_agent": target_agent,
            "timestamp": time.time(),
            "delegation_node": self._config.name,
        }

        if step_id:
            metadata["step_id"] = step_id

        if context and "trace_id" in context:
            metadata["trace_id"] = context["trace_id"]

        return metadata

    @staticmethod
    def _build_error_result(
        error: Exception,
        target_agent: str,
        correlation_id: str,
    ) -> dict[str, Any]:
        """Build a DelegationResponse-shaped error dict for on_error='continue'.

        The returned dict is valid for ``DelegationResponse.model_validate()``.

        Args:
            error: The caught exception.
            target_agent: Name of the target agent that failed.
            correlation_id: Correlation ID of the failed request.

        Returns:
            Dict with status, result, error_message, and extras.
        """
        return {
            "status": DelegationStatus.ERROR,
            "result": None,
            "error_message": str(error),
            "extras": {
                "error_type": type(error).__name__,
                "target_agent": target_agent,
                "correlation_id": correlation_id,
            },
        }

    def _extract_delegation_tokens(self, response: dict[str, Any]) -> None:
        """Extract token usage from a delegation response's extras and accumulate.

        Supports two formats:
        - New (list): each entry has prompt/completion/cached/reasoning + model_requested/provider
        - Legacy (dict): flat dict with prompt/completion/cached/reasoning (no model info)
        """
        from sofias_sdk_lite.agent import get_execution_context

        extras = response.get("extras")
        if not isinstance(extras, dict):
            return
        token_usage = extras.get("token_usage")
        if token_usage is None:
            return
        exec_ctx = get_execution_context()
        if exec_ctx is None:
            return

        if isinstance(token_usage, list):
            for entry in token_usage:
                if not isinstance(entry, dict):
                    continue
                exec_ctx.token_accumulator.add_raw(
                    prompt=entry.get("prompt_tokens", 0),
                    completion=entry.get("completion_tokens", 0),
                    cached=entry.get("cached_tokens", 0),
                    reasoning=entry.get("reasoning_tokens", 0),
                    model_requested=entry.get("model_requested", ""),
                    provider=entry.get("provider", ""),
                )
        elif isinstance(token_usage, dict):
            exec_ctx.token_accumulator.add_raw(
                prompt=token_usage.get("prompt_tokens", 0),
                completion=token_usage.get("completion_tokens", 0),
                cached=token_usage.get("cached_tokens", 0),
                reasoning=token_usage.get("reasoning_tokens", 0),
            )

    def _validate_response(
        self,
        target: DelegationTarget,
        response: dict[str, Any],
        correlation_id: str,
    ) -> dict[str, Any]:
        """Validate the response against the target's output contract.

        Args:
            target: The delegation target.
            response: The response data from the target agent.
            correlation_id: Correlation ID for error context.

        Returns:
            Validated response data.

        Raises:
            DelegationError: If validation fails.
        """
        if target.output_contract is None:
            return response

        try:
            validated = target.output_contract.model_validate(response)
            return validated.model_dump()
        except Exception as e:
            raise DelegationError(
                f"Response validation failed for target '{target.agent_name}': {e}",
                target_agent=target.agent_name,
                correlation_id=correlation_id,
                cause=e,
            ) from e

    async def _run_sequential(
        self,
        input_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute sequential delegation to targets one at a time.

        Args:
            input_data: Input data to send to targets.
            context: Optional execution context.

        Returns:
            Dict mapping agent names to their responses.

        Raises:
            DelegationError: If sending or response validation fails.
            DelegationTimeoutError: If a target doesn't respond in time.
        """
        if not self._config.targets:
            return {}

        results: dict[str, Any] = {}
        tenant_identifier, conversation_id = self._get_context_fields()
        continue_on_error = self._config.on_error == "continue"

        for target in self._config.targets:
            correlation_id = str(uuid.uuid4())

            try:
                # Transform input if mapping is configured
                transformed_input = self._apply_input_mapping(target, input_data)

                # Build metadata
                metadata = self._build_metadata(target.agent_name, context)

                logger.debug(
                    f"Delegating to '{target.agent_name}' with correlation_id={correlation_id}"
                )

                # Send request
                try:
                    await self._transport.send_request(
                        target_queue=target.get_routing_key(),
                        payload=transformed_input,
                        correlation_id=correlation_id,
                        tenant_identifier=tenant_identifier,
                        conversation_id=conversation_id,
                        metadata=metadata,
                    )
                except Exception as e:
                    raise DelegationError(
                        f"Error sending to '{target.agent_name}': {e}",
                        target_agent=target.agent_name,
                        correlation_id=correlation_id,
                        cause=e,
                    ) from e

                # Wait for response
                effective_timeout = self._resolve_target_timeout(target)
                response = await self._transport.wait_response(
                    correlation_id=correlation_id,
                    timeout=float(effective_timeout),
                )
                if response is None:
                    raise DelegationTimeoutError(
                        f"Timeout waiting for response from '{target.agent_name}' "
                        f"after {effective_timeout}s",
                        target_agent=target.agent_name,
                        correlation_id=correlation_id,
                        timeout_seconds=effective_timeout,
                    )

                # Validate and store response
                validated_response = self._validate_response(
                    target, response, correlation_id
                )
                self._extract_delegation_tokens(response)
                results[target.agent_name] = validated_response

            except DelegationError as exc:
                if not continue_on_error:
                    raise
                logger.warning(
                    f"Delegation to '{target.agent_name}' failed (on_error=continue): {exc}"
                )
                results[target.agent_name] = self._build_error_result(
                    exc, target.agent_name, correlation_id
                )

        return results

    async def _run_parallel(
        self,
        input_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute parallel delegation to all targets simultaneously.

        Args:
            input_data: Input data to send to targets.
            context: Optional execution context.

        Returns:
            Dict mapping agent names to their responses.

        Raises:
            DelegationError: If sending or response validation fails.
            DelegationTimeoutError: If not all targets respond in time.
        """
        if not self._config.targets:
            return {}

        results: dict[str, Any] = {}
        pending_correlations: dict[str, DelegationTarget] = {}
        tenant_identifier, conversation_id = self._get_context_fields()
        continue_on_error = self._config.on_error == "continue"

        # Send to all targets
        for target in self._config.targets:
            correlation_id = str(uuid.uuid4())

            # Transform input if mapping is configured
            transformed_input = self._apply_input_mapping(target, input_data)

            # Build metadata
            metadata = self._build_metadata(target.agent_name, context)

            logger.debug(
                f"Delegating (parallel) to '{target.agent_name}' "
                f"with correlation_id={correlation_id}"
            )

            try:
                await self._transport.send_request(
                    target_queue=target.get_routing_key(),
                    payload=transformed_input,
                    correlation_id=correlation_id,
                    tenant_identifier=tenant_identifier,
                    conversation_id=conversation_id,
                    metadata=metadata,
                )
                pending_correlations[correlation_id] = target
            except Exception as e:
                if not continue_on_error:
                    raise DelegationError(
                        f"Error sending to '{target.agent_name}': {e}",
                        target_agent=target.agent_name,
                        correlation_id=correlation_id,
                        cause=e,
                    ) from e
                exc = DelegationError(
                    f"Error sending to '{target.agent_name}': {e}",
                    target_agent=target.agent_name,
                    correlation_id=correlation_id,
                    cause=e,
                )
                logger.warning(
                    f"Send to '{target.agent_name}' failed (on_error=continue): {exc}"
                )
                results[target.agent_name] = self._build_error_result(
                    exc, target.agent_name, correlation_id
                )

        # Wait for all responses using wait_any_response pattern
        pending_ids = set(pending_correlations.keys())
        start_time = time.time()

        # Track per-target deadlines
        target_deadlines: dict[str, float] = {}
        for cid, target in pending_correlations.items():
            effective_timeout = self._resolve_target_timeout(target)
            target_deadlines[cid] = start_time + effective_timeout

        while pending_ids:
            now = time.time()

            # Check global timeout
            global_remaining = self._config.global_timeout_seconds - (now - start_time)
            if global_remaining <= 0:
                if not continue_on_error:
                    pending_agents = [
                        pending_correlations[cid].agent_name for cid in pending_ids
                    ]
                    raise DelegationTimeoutError(
                        f"Timeout waiting for parallel delegation responses after "
                        f"{self._config.global_timeout_seconds}s. Pending agents: {pending_agents}",
                        target_agent=pending_agents[0] if pending_agents else "unknown",
                        timeout_seconds=self._config.global_timeout_seconds,
                        pending_agents=pending_agents,
                    )
                for cid in list(pending_ids):
                    t = pending_correlations[cid]
                    exc = DelegationTimeoutError(
                        f"Timeout waiting for response from '{t.agent_name}' "
                        f"after {self._config.global_timeout_seconds}s",
                        target_agent=t.agent_name,
                        correlation_id=cid,
                        timeout_seconds=self._config.global_timeout_seconds,
                    )
                    logger.warning(
                        f"Parallel delegation to '{t.agent_name}' timed out (on_error=continue)"
                    )
                    results[t.agent_name] = self._build_error_result(
                        exc, t.agent_name, cid
                    )
                break

            # Check per-target deadlines
            expired_cids = [
                cid for cid in pending_ids if target_deadlines[cid] <= now
            ]
            for cid in expired_cids:
                t = pending_correlations[cid]
                effective_timeout = self._resolve_target_timeout(t)
                pending_ids.discard(cid)
                if not continue_on_error:
                    raise DelegationTimeoutError(
                        f"Timeout waiting for response from '{t.agent_name}' "
                        f"after {effective_timeout}s",
                        target_agent=t.agent_name,
                        correlation_id=cid,
                        timeout_seconds=effective_timeout,
                    )
                exc = DelegationTimeoutError(
                    f"Timeout waiting for response from '{t.agent_name}' "
                    f"after {effective_timeout}s",
                    target_agent=t.agent_name,
                    correlation_id=cid,
                    timeout_seconds=effective_timeout,
                )
                logger.warning(
                    f"Parallel delegation to '{t.agent_name}' per-target timed out "
                    f"(on_error=continue)"
                )
                results[t.agent_name] = self._build_error_result(
                    exc, t.agent_name, cid
                )

            if not pending_ids:
                break

            # Compute next wait timeout: min of global remaining and per-target deadlines
            now = time.time()
            global_remaining = self._config.global_timeout_seconds - (now - start_time)
            next_target_deadline = min(target_deadlines[cid] for cid in pending_ids)
            target_remaining = next_target_deadline - now
            wait_timeout = max(min(global_remaining, target_remaining), 0)

            wait_result = await self._transport.wait_any_response(
                correlation_ids=pending_ids,
                timeout=wait_timeout,
            )
            if wait_result is None:
                # Transport says timeout elapsed. Recheck real deadlines.
                now_after = time.time()
                global_hit = (now_after - start_time) >= self._config.global_timeout_seconds
                per_target_hit = any(
                    target_deadlines[cid] <= now_after for cid in pending_ids
                )
                if global_hit or per_target_hit:
                    # A real deadline fired — loop back to the checks above.
                    continue
                # No real deadline fired (instant mock return). Treat all
                # remaining targets as timed out by the nearest deadline.
                for cid in list(pending_ids):
                    t = pending_correlations[cid]
                    effective_timeout = self._resolve_target_timeout(t)
                    pending_ids.discard(cid)
                    if not continue_on_error:
                        raise DelegationTimeoutError(
                            f"Timeout waiting for response from '{t.agent_name}' "
                            f"after {effective_timeout}s",
                            target_agent=t.agent_name,
                            correlation_id=cid,
                            timeout_seconds=effective_timeout,
                        )
                    exc = DelegationTimeoutError(
                        f"Timeout waiting for response from '{t.agent_name}' "
                        f"after {effective_timeout}s",
                        target_agent=t.agent_name,
                        correlation_id=cid,
                        timeout_seconds=effective_timeout,
                    )
                    logger.warning(
                        f"Parallel delegation to '{t.agent_name}' timed out "
                        f"(on_error=continue)"
                    )
                    results[t.agent_name] = self._build_error_result(
                        exc, t.agent_name, cid
                    )
                break

            correlation_id, response = wait_result
            pending_ids.discard(correlation_id)
            target = pending_correlations[correlation_id]

            # Validate and store response
            self._extract_delegation_tokens(response)
            try:
                validated_response = self._validate_response(
                    target, response, correlation_id
                )
                results[target.agent_name] = validated_response
            except DelegationError as exc:
                if not continue_on_error:
                    raise
                logger.warning(
                    f"Validation for '{target.agent_name}' failed (on_error=continue): {exc}"
                )
                results[target.agent_name] = self._build_error_result(
                    exc, target.agent_name, correlation_id
                )

        return results

    async def _run(
        self,
        input_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the delegation logic based on execution_mode.

        Args:
            input_data: Validated input dictionary.
            context: Optional context dictionary with trace_id, etc.

        Returns:
            For sequential/parallel modes:
                Dict mapping agent names to their responses.
                Example: {"summarizer": {...}, "classifier": {...}}

            For DAG mode:
                Dict with execution results:
                {
                    "results": {step_id: response_dict, ...},
                    "failed": {step_id: error_info, ...},
                    "skipped": [step_id, ...],
                    "all_completed": bool,
                }

        Raises:
            DelegationError: If delegation fails.
            DelegationTimeoutError: If targets don't respond in time.
            DelegationPlanError: If DAG execution fails with fail_plan policy.
        """
        if self._config.execution_mode == "sequential":
            result = await self._run_sequential(input_data, context)
        elif self._config.execution_mode == "parallel":
            result = await self._run_parallel(input_data, context)
        elif self._config.execution_mode == "dag":
            result = await self._run_dag(input_data, context)
        else:
            raise DelegationError(
                f"Unknown execution mode: {self._config.execution_mode}",
                target_agent="unknown",
            )

        if self._config.output_field is not None:
            return {**input_data, self._config.output_field: result}
        return result

    def _resolve_plan(self, input_data: dict[str, Any]) -> DelegationPlan:
        """Resolve the delegation plan from input or config.

        Args:
            input_data: Input data which may contain the plan.

        Returns:
            Validated DelegationPlan.

        Raises:
            DelegationError: If plan parsing fails.
        """
        if self._config.dynamic_targets:
            # Parse plan from input
            try:
                return DelegationPlan.model_validate(input_data)
            except Exception as e:
                raise DelegationError(
                    f"Failed to parse delegation plan from input: {e}",
                    target_agent="unknown",
                    cause=e,
                ) from e
        else:
            # Build plan from static targets (no dependencies)
            steps = [
                DelegationStep(
                    id=f"target_{i}",
                    agent_name=target.agent_name,
                    input=input_data,
                    depends_on=[],
                    routing_key=target.routing_key,
                )
                for i, target in enumerate(self._config.targets)
            ]
            return DelegationPlan(steps=steps)

    def _get_failure_policy(
        self, agent_name: str
    ) -> Literal["fail_plan", "skip_dependents", "continue_partial"]:
        """Get the failure policy for a specific agent.

        Args:
            agent_name: Name of the agent.

        Returns:
            The applicable failure policy.
        """
        return self._config.failure_overrides.get(
            agent_name, self._config.default_failure_policy
        )

    def _get_dependents(
        self, step_id: str, steps: list[DelegationStep]
    ) -> set[str]:
        """Get all steps that depend (directly or transitively) on a given step.

        Args:
            step_id: The step ID to find dependents for.
            steps: All steps in the plan.

        Returns:
            Set of step IDs that depend on the given step.
        """
        dependents: set[str] = set()
        to_process = {step_id}

        while to_process:
            current = to_process.pop()
            for step in steps:
                if current in step.depends_on and step.id not in dependents:
                    dependents.add(step.id)
                    to_process.add(step.id)

        return dependents

    def _build_enriched_input(
        self,
        step: DelegationStep,
        results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Build the enriched input for a step with dependency results.

        Args:
            step: The step to build input for.
            results: Results from completed steps.

        Returns:
            Enriched input dictionary with _dependency_results field.
        """
        dependency_results = {
            dep_id: results[dep_id]
            for dep_id in step.depends_on
            if dep_id in results
        }

        return {
            **step.input,
            "_dependency_results": dependency_results,
        }

    async def _run_dag(
        self,
        input_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute DAG-based delegation respecting step dependencies.

        Args:
            input_data: Input data (may contain DelegationPlan if dynamic_targets).
            context: Optional execution context.

        Returns:
            Dict with execution results:
            {
                "results": {step_id: response_dict, ...},
                "failed": {step_id: error_info, ...},
                "skipped": [step_id, ...],
                "all_completed": bool,
            }

        Raises:
            DelegationError: If sending fails.
            DelegationTimeoutError: If global timeout is reached.
            DelegationPlanError: If fail_plan policy is triggered.
        """
        # Resolve the plan
        plan = self._resolve_plan(input_data)

        if not plan.steps:
            return {
                "results": {},
                "failed": {},
                "skipped": [],
                "all_completed": True,
            }

        # State tracking
        status: dict[str, str] = {step.id: "pending" for step in plan.steps}
        results: dict[str, dict[str, Any]] = {}
        failed: dict[str, dict[str, Any]] = {}

        # Map step_id -> step for quick lookup
        steps_by_id: dict[str, DelegationStep] = {step.id: step for step in plan.steps}

        # Map correlation_id -> step_id for response matching
        correlation_to_step: dict[str, str] = {}

        # Track running steps
        running_correlations: set[str] = set()

        start_time = time.time()

        # Find and launch initial steps (those with no dependencies)
        for step in plan.steps:
            if not step.depends_on:
                correlation_id = await self._send_step(
                    step=step,
                    enriched_input=self._build_enriched_input(step, results),
                    context=context,
                )
                correlation_to_step[correlation_id] = step.id
                running_correlations.add(correlation_id)
                status[step.id] = "running"

        # Main processing loop
        while not self._all_terminal(status):
            # Check timeout
            elapsed = time.time() - start_time
            remaining = self._config.global_timeout_seconds - elapsed

            if remaining <= 0:
                # Timeout reached
                pending_step_ids = [
                    sid for sid, st in status.items() if st in ("pending", "running")
                ]
                pending_agents = [
                    steps_by_id[sid].agent_name for sid in pending_step_ids
                ]

                if self._config.default_failure_policy == "fail_plan":
                    raise DelegationTimeoutError(
                        f"Timeout waiting for DAG delegation after "
                        f"{self._config.global_timeout_seconds}s",
                        target_agent=pending_agents[0] if pending_agents else "unknown",
                        timeout_seconds=self._config.global_timeout_seconds,
                        pending_agents=pending_agents,
                    )

                # continue_partial / skip_dependents: mark pending/running
                # as failed/skipped and return partial results
                logger.warning(
                    f"DAG timeout after {self._config.global_timeout_seconds}s "
                    f"(policy={self._config.default_failure_policy}). "
                    f"Pending steps: {pending_step_ids}"
                )

                for sid in pending_step_ids:
                    step = steps_by_id[sid]
                    if status[sid] == "running":
                        status[sid] = "failed"
                        failed[sid] = {
                            "agent_name": step.agent_name,
                            "error": (
                                f"Timeout waiting for response after "
                                f"{self._config.global_timeout_seconds}s"
                            ),
                            "timeout": True,
                        }
                    elif status[sid] == "pending":
                        if self._config.default_failure_policy == "skip_dependents":
                            status[sid] = "skipped"
                        else:
                            # continue_partial: also mark as skipped (never started)
                            status[sid] = "skipped"

                break

            # If no running steps, something is wrong (shouldn't happen)
            if not running_correlations:
                break

            # Wait for next response
            wait_result = await self._transport.wait_any_response(
                correlation_ids=running_correlations,
                timeout=min(remaining, 5.0),  # Check every 5s max
            )
            if wait_result is None:
                # No response yet, loop to check global timeout and other conditions
                continue

            correlation_id, response = wait_result
            running_correlations.discard(correlation_id)

            if correlation_id not in correlation_to_step:
                logger.warning(
                    f"Received response with unknown correlation_id: {correlation_id}"
                )
                continue

            step_id = correlation_to_step[correlation_id]
            step = steps_by_id[step_id]

            # Check for error in response
            is_error = response.get("error")

            if is_error:
                # Step failed
                error_info = {
                    "agent_name": step.agent_name,
                    "error": response.get("error", "Unknown error"),
                    "correlation_id": correlation_id,
                }

                policy = self._get_failure_policy(step.agent_name)

                if policy == "fail_plan":
                    # Mark all pending/running as skipped and raise
                    skipped_steps = [
                        sid for sid, st in status.items()
                        if st in ("pending", "running") and sid != step_id
                    ]
                    completed_steps = [
                        sid for sid, st in status.items() if st == "completed"
                    ]

                    raise DelegationPlanError(
                        f"Step '{step_id}' failed with fail_plan policy",
                        failed_step_id=step_id,
                        target_agent=step.agent_name,
                        completed_steps=completed_steps,
                        skipped_steps=skipped_steps,
                        correlation_id=correlation_id,
                        cause=DelegationStepError(
                            f"Step '{step_id}' failed: {error_info.get('error')}",
                            step_id=step_id,
                            target_agent=step.agent_name,
                            correlation_id=correlation_id,
                        ),
                    )

                elif policy == "skip_dependents":
                    # Mark step as failed
                    status[step_id] = "failed"
                    failed[step_id] = error_info

                    # Mark all dependents as skipped
                    dependents = self._get_dependents(step_id, plan.steps)
                    for dep_id in dependents:
                        if status[dep_id] in ("pending",):
                            status[dep_id] = "skipped"
                            logger.debug(
                                f"Skipping step '{dep_id}' due to failed dependency '{step_id}'"
                            )

                elif policy == "continue_partial":
                    # Mark step as failed but continue
                    status[step_id] = "failed"
                    failed[step_id] = error_info
                    # Dependents will execute without this result

            else:
                # Step completed successfully
                status[step_id] = "completed"
                self._extract_delegation_tokens(response)
                results[step_id] = response

                logger.debug(f"Step '{step_id}' completed successfully")

            # Check for newly unblocked steps
            for candidate in plan.steps:
                if status[candidate.id] != "pending":
                    continue

                # Check if all dependencies are resolved
                deps_resolved = all(
                    status[dep_id] in ("completed", "failed", "skipped")
                    for dep_id in candidate.depends_on
                )

                if not deps_resolved:
                    continue

                # Check if any dependency was skipped (then this should be skipped too
                # unless continue_partial applies)
                any_skipped = any(
                    status[dep_id] == "skipped" for dep_id in candidate.depends_on
                )

                if any_skipped:
                    # This step should be skipped (its dependency chain is broken)
                    status[candidate.id] = "skipped"
                    continue

                # Launch this step
                correlation_id = await self._send_step(
                    step=candidate,
                    enriched_input=self._build_enriched_input(candidate, results),
                    context=context,
                )
                correlation_to_step[correlation_id] = candidate.id
                running_correlations.add(correlation_id)
                status[candidate.id] = "running"

        # Build final result
        skipped_list = [sid for sid, st in status.items() if st == "skipped"]
        all_completed = all(st == "completed" for st in status.values())

        return {
            "results": results,
            "failed": failed,
            "skipped": skipped_list,
            "all_completed": all_completed,
        }

    async def _send_step(
        self,
        step: DelegationStep,
        enriched_input: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> str:
        """Send a step to the transport.

        Args:
            step: The step to send.
            enriched_input: The input with dependency results injected.
            context: Optional execution context.

        Returns:
            The correlation_id for this request.

        Raises:
            DelegationError: If sending fails.
        """
        correlation_id = str(uuid.uuid4())
        tenant_identifier, conversation_id = self._get_context_fields()

        metadata = self._build_metadata(
            target_agent=step.agent_name,
            context=context,
            step_id=step.id,
        )

        logger.debug(
            f"Publishing DAG step '{step.id}' to '{step.agent_name}' "
            f"with correlation_id={correlation_id}"
        )

        try:
            await self._transport.send_request(
                target_queue=step.get_routing_key(),
                payload=enriched_input,
                correlation_id=correlation_id,
                tenant_identifier=tenant_identifier,
                conversation_id=conversation_id,
                metadata=metadata,
            )
            return correlation_id
        except Exception as e:
            raise DelegationError(
                f"Error sending step '{step.id}' to '{step.agent_name}': {e}",
                target_agent=step.agent_name,
                correlation_id=correlation_id,
                cause=e,
            ) from e

    def _all_terminal(self, status: dict[str, str]) -> bool:
        """Check if all steps are in a terminal state.

        Args:
            status: Map of step_id to status.

        Returns:
            True if all steps are completed, failed, or skipped.
        """
        terminal_states = {"completed", "failed", "skipped"}
        return all(st in terminal_states for st in status.values())

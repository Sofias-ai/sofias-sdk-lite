"""PlanExecutor for DAG-based execution of plans produced by PlannerNode.

This module provides the PlanExecutor class which takes an ExecutionPlan
and executes it as a DAG, running independent steps in parallel via
asyncio.create_task().
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Literal

from sofias_sdk_lite.errors import NodeExecutionError, PlanExecutionError
from sofias_sdk_lite.nodes.planning_models import ExecutionPlan, PlanResult, PlanStep
from sofias_sdk_lite.observability._log import get_logger

if TYPE_CHECKING:
    from sofias_sdk_lite.agent import ExecutionContext
    from sofias_sdk_lite.errors import ErrorHandler
    from sofias_sdk_lite.nodes.base_node import BaseNode

logger = get_logger("nodes.plan_executor")

__all__ = ["PlanExecutor"]


class PlanExecutor:
    """Executes an ExecutionPlan as a DAG of node invocations.

    Unlike DelegationNode._run_dag() which publishes to a message transport,
    PlanExecutor invokes nodes directly via node.execute() and uses
    asyncio.create_task() for parallelism.
    """

    def __init__(
        self,
        nodes: dict[str, BaseNode],
        error_handler: ErrorHandler | None = None,
        default_failure_policy: Literal[
            "fail_plan", "skip_dependents", "continue_partial"
        ] = "fail_plan",
        failure_overrides: dict[str, str] | None = None,
        global_timeout_seconds: int = 600,
    ) -> None:
        """Initialize the PlanExecutor.

        Args:
            nodes: All nodes available for execution, keyed by name.
            error_handler: Optional error handler for retry/fallback on step execution.
            default_failure_policy: Policy when a step fails.
            failure_overrides: Per-node failure policy overrides.
            global_timeout_seconds: Maximum total time for plan execution.
        """
        self._nodes = nodes
        self._error_handler = error_handler
        self._default_failure_policy = default_failure_policy
        self._failure_overrides = failure_overrides or {}
        self._global_timeout_seconds = global_timeout_seconds

    def validate_plan(self, plan: ExecutionPlan) -> list[str]:
        """Validate that a plan can be executed.

        Checks:
        1. All node_name references exist in self._nodes
        2. Input of each step against the InputContract of the referenced node
           - Steps WITHOUT dependencies: full validation (error if fails)
           - Steps WITH dependencies: best-effort (warning, not error)

        Args:
            plan: The execution plan to validate.

        Returns:
            List of validation error messages. Empty means valid.
        """
        errors: list[str] = []

        for step in plan.steps:
            node = self._nodes.get(step.node_name)
            if node is None:
                errors.append(
                    f"Step '{step.id}' references unknown node '{step.node_name}'"
                )
                continue

            # Validate input against contract
            has_dependencies = bool(step.depends_on)
            try:
                node.contract.validate_input(step.input)
            except Exception as e:
                if has_dependencies:
                    logger.debug(
                        "Step input validation warning (dependencies may provide missing fields)",
                        step_id=step.id,
                        node_name=step.node_name,
                        warning=str(e),
                    )
                else:
                    errors.append(
                        f"Step '{step.id}' input validation failed for node "
                        f"'{step.node_name}': {e}"
                    )

        return errors

    async def execute(
        self,
        plan: ExecutionPlan,
        execution_context: ExecutionContext | None = None,
    ) -> PlanResult:
        """Execute the plan as a DAG.

        Args:
            plan: The validated execution plan.
            execution_context: Optional execution context for transition tracking.

        Returns:
            PlanResult with results, failures, and skipped steps.

        Raises:
            PlanExecutionError: If fail_plan policy is triggered.
            asyncio.TimeoutError: If global timeout is exceeded.
        """
        if not plan.steps:
            return PlanResult()

        # State tracking
        status: dict[str, str] = {step.id: "pending" for step in plan.steps}
        results: dict[str, dict[str, Any]] = {}
        failed: dict[str, dict[str, Any]] = {}

        steps_by_id: dict[str, PlanStep] = {step.id: step for step in plan.steps}

        # Track running tasks: task -> step_id
        running_tasks: dict[asyncio.Task[dict[str, Any]], str] = {}

        start_time = time.monotonic()

        try:
            # Launch initial steps (no dependencies)
            for step in plan.steps:
                if not step.depends_on:
                    task = asyncio.create_task(
                        self._execute_step(step, results, execution_context),
                        name=f"plan_step_{step.id}",
                    )
                    running_tasks[task] = step.id
                    status[step.id] = "running"

            # Main processing loop
            while not self._all_terminal(status):
                if not running_tasks:
                    # No running tasks but not all terminal — deadlock or all remaining are skipped
                    break

                # Check timeout
                elapsed = time.monotonic() - start_time
                remaining = self._global_timeout_seconds - elapsed
                if remaining <= 0:
                    # Cancel running tasks
                    for task in running_tasks:
                        task.cancel()
                    pending_ids = [
                        sid for sid, st in status.items()
                        if st in ("pending", "running")
                    ]
                    raise PlanExecutionError(
                        f"Plan execution timed out after {self._global_timeout_seconds}s",
                        failed_step_id=pending_ids[0] if pending_ids else "unknown",
                        failed_node_name="timeout",
                        completed_steps=[
                            sid for sid, st in status.items() if st == "completed"
                        ],
                        skipped_steps=pending_ids,
                    )

                # Wait for at least one task to complete
                done, _ = await asyncio.wait(
                    running_tasks.keys(),
                    timeout=min(remaining, 5.0),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if not done:
                    continue  # Timeout on wait, loop to check global timeout

                for task in done:
                    step_id = running_tasks.pop(task)
                    step = steps_by_id[step_id]

                    try:
                        result = task.result()
                        status[step_id] = "completed"
                        results[step_id] = result

                        logger.debug(
                            "Plan step completed",
                            step_id=step_id,
                            node_name=step.node_name,
                        )

                    except Exception as e:
                        error_info = {
                            "node_name": step.node_name,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        }

                        policy = self._get_failure_policy(step.node_name)

                        if policy == "fail_plan":
                            # Cancel all running tasks
                            for t in running_tasks:
                                t.cancel()

                            skipped_steps = [
                                sid for sid, st in status.items()
                                if st in ("pending", "running") and sid != step_id
                            ]
                            completed_steps = [
                                sid for sid, st in status.items()
                                if st == "completed"
                            ]

                            raise PlanExecutionError(
                                f"Step '{step_id}' failed with fail_plan policy: {e}",
                                failed_step_id=step_id,
                                failed_node_name=step.node_name,
                                completed_steps=completed_steps,
                                skipped_steps=skipped_steps,
                                cause=e,
                            ) from e

                        elif policy == "skip_dependents":
                            status[step_id] = "failed"
                            failed[step_id] = error_info

                            dependents = self._get_dependents(step_id, plan.steps)
                            for dep_id in dependents:
                                if status[dep_id] == "pending":
                                    status[dep_id] = "skipped"
                                    logger.debug(
                                        "Skipping step due to failed dependency",
                                        step_id=dep_id,
                                        failed_dependency=step_id,
                                    )

                        elif policy == "continue_partial":
                            status[step_id] = "failed"
                            failed[step_id] = error_info

                # Launch newly unblocked steps
                for candidate in plan.steps:
                    if status[candidate.id] != "pending":
                        continue

                    deps_resolved = all(
                        status[dep_id] in ("completed", "failed", "skipped")
                        for dep_id in candidate.depends_on
                    )
                    if not deps_resolved:
                        continue

                    any_skipped = any(
                        status[dep_id] == "skipped"
                        for dep_id in candidate.depends_on
                    )
                    if any_skipped:
                        status[candidate.id] = "skipped"
                        continue

                    task = asyncio.create_task(
                        self._execute_step(candidate, results, execution_context),
                        name=f"plan_step_{candidate.id}",
                    )
                    running_tasks[task] = candidate.id
                    status[candidate.id] = "running"

        except PlanExecutionError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise PlanExecutionError(
                f"Unexpected error during plan execution: {e}",
                failed_step_id="unknown",
                failed_node_name="unknown",
                completed_steps=[
                    sid for sid, st in status.items() if st == "completed"
                ],
                skipped_steps=[
                    sid for sid, st in status.items()
                    if st in ("pending", "running")
                ],
                cause=e,
            ) from e

        skipped_list = [sid for sid, st in status.items() if st == "skipped"]
        all_completed = all(st == "completed" for st in status.values())

        return PlanResult(
            results=results,
            failed=failed,
            skipped=skipped_list,
            all_completed=all_completed,
        )

    async def _execute_step(
        self,
        step: PlanStep,
        results: dict[str, dict[str, Any]],
        execution_context: ExecutionContext | None = None,
    ) -> dict[str, Any]:
        """Execute a single plan step.

        Enriches input with dependency results and invokes the node.

        Args:
            step: The step to execute.
            results: Results from completed steps (for dependency injection).
            execution_context: Optional execution context.

        Returns:
            Output dictionary from the node.

        Raises:
            NodeExecutionError: If node execution fails.
        """
        node = self._nodes[step.node_name]
        enriched_input = self._build_enriched_input(step, results)

        # Update execution context if available
        if execution_context is not None:
            execution_context.transition_history.append(
                f"plan:{step.id}:{step.node_name}"
            )

        logger.info(
            "Executing plan step",
            step_id=step.id,
            node_name=step.node_name,
        )

        start_time = time.monotonic()

        try:
            if self._error_handler is not None:
                result = await self._error_handler.handle_node_execution(
                    node_name=step.node_name,
                    execute_fn=node.execute,
                    input_data=enriched_input,
                )
            else:
                result = await node.execute(enriched_input)

            duration_ms = (time.monotonic() - start_time) * 1000
            logger.info(
                "Plan step completed",
                step_id=step.id,
                node_name=step.node_name,
                duration_ms=round(duration_ms, 2),
            )
            return result

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Plan step failed",
                step_id=step.id,
                node_name=step.node_name,
                duration_ms=round(duration_ms, 2),
                error_type=type(e).__name__,
                error=str(e),
            )
            raise

    def _build_enriched_input(
        self,
        step: PlanStep,
        results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Build enriched input for a step with dependency results.

        If the step has dependencies, their outputs are injected as
        ``dependency_results`` (a regular Pydantic-friendly field name).
        Steps without dependencies receive only their declared input.

        Args:
            step: The step to build input for.
            results: Results from completed steps.

        Returns:
            Enriched input dictionary, with ``dependency_results`` added
            only when the step declares dependencies.
        """
        enriched = dict(step.input)

        if step.depends_on:
            enriched["dependency_results"] = {
                dep_id: results[dep_id]
                for dep_id in step.depends_on
                if dep_id in results
            }

        return enriched

    def _get_failure_policy(
        self, node_name: str
    ) -> Literal["fail_plan", "skip_dependents", "continue_partial"]:
        """Get the failure policy for a specific node."""
        return self._failure_overrides.get(  # type: ignore[return-value]
            node_name, self._default_failure_policy
        )

    def _get_dependents(
        self, step_id: str, steps: list[PlanStep]
    ) -> set[str]:
        """Get all steps that depend (directly or transitively) on a given step."""
        dependents: set[str] = set()
        to_process = {step_id}

        while to_process:
            current = to_process.pop()
            for step in steps:
                if current in step.depends_on and step.id not in dependents:
                    dependents.add(step.id)
                    to_process.add(step.id)

        return dependents

    def _all_terminal(self, status: dict[str, str]) -> bool:
        """Check if all steps are in a terminal state."""
        terminal_states = {"completed", "failed", "skipped"}
        return all(st in terminal_states for st in status.values())

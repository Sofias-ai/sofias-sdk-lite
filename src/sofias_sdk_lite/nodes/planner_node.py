"""PlannerNode for dynamic graph compilation via LLM.

The PlannerNode uses an LLM to generate an ExecutionPlan (DAG of steps)
based on the input data and the list of available nodes. It ONLY plans —
execution is handled by PlanExecutor, triggered by Agent.execute().
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from sofias_sdk_lite.errors import NodeExecutionError
from sofias_sdk_lite.nodes.base_node import BaseNode, NodeType
from sofias_sdk_lite.nodes.planning_models import ExecutionPlan, NodeDescriptor
from sofias_sdk_lite.observability._log import get_logger

if TYPE_CHECKING:
    from sofias_sdk_lite.contracts import NodeContract
    from sofias_sdk_lite.llm import LLMCallable

logger = get_logger("nodes.planner_node")

__all__ = ["PlannerNode"]


class PlannerNode(BaseNode):
    """Node that generates an ExecutionPlan via LLM.

    The PlannerNode:
    1. Receives input data (task description, context, or re-plan info)
    2. Calls an LLM with a prompt describing available nodes
    3. Parses the LLM response as an ExecutionPlan
    4. Returns the plan as its output

    The Agent detects this node by type (node_type == PLANNER) and
    routes the output to PlanExecutor for execution.

    Example:
        planner = PlannerNode(
            name="planner",
            contract=planner_contract,
            llm=my_llm,
            available_nodes=[
                NodeDescriptor(name="search", node_type=NodeType.FUNCTION, description="Search docs"),
                NodeDescriptor(name="summarize", node_type=NodeType.LLM, description="Summarize text"),
            ],
        )
    """

    def __init__(
        self,
        name: str,
        contract: NodeContract,
        llm: LLMCallable,
        available_nodes: list[NodeDescriptor],
        system_prompt: str | None = None,
        description: str | None = None,
    ) -> None:
        """Initialize the PlannerNode.

        Args:
            name: Unique name identifying this node.
            contract: Input/output contract for validation.
            llm: LLM callable for generating plans.
            available_nodes: List of nodes the planner can include in plans.
            system_prompt: Custom system prompt. If None, a default is built.
            description: Human-readable description.
        """
        super().__init__(name=name, contract=contract, description=description)
        self._llm = llm
        self._available_nodes = available_nodes
        self._system_prompt = system_prompt or self._build_default_prompt()

    @property
    def node_type(self) -> NodeType:
        return NodeType.PLANNER

    def _build_default_prompt(self) -> str:
        """Build the default system prompt for plan generation."""
        nodes_desc = self._build_nodes_description()
        plan_schema = json.dumps(
            ExecutionPlan.model_json_schema(), indent=2
        )

        return f"""You are a planning agent. Your job is to analyze the input and create an execution plan.

## Available Nodes

{nodes_desc}

## Output Format

You MUST respond with a valid JSON object matching this schema:

{plan_schema}

## Rules

1. Each step must reference a node from the available nodes list by its `node_name` field.
2. Use `depends_on` to express dependencies between steps (list of step IDs).
3. Steps without dependencies will run in parallel.
4. Step IDs must be unique strings (e.g., "s1", "s2", ...).
5. Include relevant input data for each step in the `input` field.
6. If you receive previous plan results with failures, create a revised plan that accounts for the failures.

Respond ONLY with the JSON object. No markdown, no explanation."""

    def _build_nodes_description(self) -> str:
        """Format available nodes for the system prompt."""
        lines = []
        for node in self._available_nodes:
            desc = node.description or "No description"
            lines.append(f"- **{node.name}** (type: {node.node_type.value}): {desc}")
        return "\n".join(lines)

    async def _run(
        self,
        input_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate an execution plan via LLM.

        Args:
            input_data: Validated input (task description, or re-plan context with failures).
            context: Optional execution context.

        Returns:
            ExecutionPlan serialized as a dict.

        Raises:
            NodeExecutionError: If LLM invocation or plan parsing fails.
        """
        start_time = time.perf_counter()

        logger.info(
            "PlannerNode execution started",
            node_name=self.name,
            available_nodes=len(self._available_nodes),
        )

        try:
            # Build user message from input
            user_message = json.dumps(input_data, default=str)
            messages = [{"role": "user", "content": user_message}]

            # Call LLM
            response = await self._llm.invoke(
                prompt=self._system_prompt,
                tools=None,
                messages=messages,
            )

            if not response.content:
                raise NodeExecutionError(
                    f"PlannerNode '{self.name}' received empty response from LLM",
                    node_name=self.name,
                )

            # Parse JSON response
            try:
                plan_data = json.loads(response.content)
            except json.JSONDecodeError as e:
                raise NodeExecutionError(
                    f"PlannerNode '{self.name}' LLM response is not valid JSON: {e}",
                    node_name=self.name,
                    cause=e,
                ) from e

            # Validate as ExecutionPlan
            try:
                plan = ExecutionPlan.model_validate(plan_data)
            except Exception as e:
                raise NodeExecutionError(
                    f"PlannerNode '{self.name}' LLM response is not a valid ExecutionPlan: {e}",
                    node_name=self.name,
                    cause=e,
                ) from e

            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "PlannerNode execution completed",
                node_name=self.name,
                steps_planned=len(plan.steps),
                duration_ms=round(duration_ms, 2),
            )

            return plan.model_dump()

        except NodeExecutionError:
            raise
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "PlannerNode execution failed",
                node_name=self.name,
                duration_ms=round(duration_ms, 2),
                error_type=type(e).__name__,
                error=str(e),
            )
            raise NodeExecutionError(
                f"PlannerNode '{self.name}' execution failed: {e}",
                node_name=self.name,
                cause=e,
            ) from e

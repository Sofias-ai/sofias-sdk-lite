"""Configuration for DelegationNode.

This module defines configuration structures for nodes that delegate
tasks to other agents via an event bus (e.g. RabbitMQ).
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from sofias_sdk_lite.contracts import StrictContract
from sofias_sdk_lite.nodes.llm_node_config import NodeRetryConfig

__all__ = [
    "DelegationNodeConfig",
    "DelegationPlan",
    "DelegationStep",
    "DelegationTarget",
]


class DelegationTarget(BaseModel):
    """Configuration for a target agent in delegation.

    Defines where to send delegation requests and how to handle
    the communication with a specific target agent.

    Example:
        ```python
        target = DelegationTarget(
            agent_name="summarizer",
            routing_key="agent.summarizer",
            input_mapping="transform_for_summarizer",
            output_contract=SummaryOutput,
            timeout_seconds=120,
        )
        ```
    """

    model_config = ConfigDict(frozen=True)

    agent_name: str
    """Name of the target agent to delegate to."""

    routing_key: str | None = None
    """Routing key for the target agent's queue.
    If None, defaults to 'agent.{agent_name}'."""

    input_mapping: str | None = None
    """Name of a registered mapper function to transform input before delegation.
    If None, the input is sent as-is."""

    output_contract: type[StrictContract] | None = None
    """Expected output contract from the target agent.
    If defined, the response is validated against this schema.
    Accepts OutputContract, DelegationResponse, or any StrictContract subclass.
    If None, any dict response is accepted."""

    timeout_seconds: int = 300
    """Timeout in seconds for waiting for this target's response."""

    def get_routing_key(self) -> str:
        """Get the effective routing key for this target."""
        return self.routing_key or f"agent.{self.agent_name}"


class DelegationStep(BaseModel):
    """A single step in a delegation plan for DAG execution.

    Represents one delegation task within a plan, including its dependencies
    on other steps. Used when dynamic_targets is enabled.

    Example:
        ```python
        step = DelegationStep(
            id="step_1",
            agent_name="web-search",
            input={"query": "Bitcoin price"},
            depends_on=[],
        )
        ```
    """

    model_config = ConfigDict(frozen=True)

    id: str
    """Unique identifier for this step within the plan."""

    agent_name: str
    """Name of the target agent to delegate to."""

    input: dict[str, Any]
    """Payload to send to the target agent."""

    depends_on: list[str] = []
    """List of step IDs that must complete before this step can run."""

    routing_key: str | None = None
    """Override for the agent's routing key. If None, uses 'agent.{agent_name}'."""

    def get_routing_key(self) -> str:
        """Get the effective routing key for this step."""
        return self.routing_key or f"agent.{self.agent_name}"


class DelegationPlan(BaseModel):
    """A complete delegation plan for DAG execution.

    Contains a list of steps with their dependencies, forming a directed
    acyclic graph (DAG). This is the expected input format for DelegationNode
    when execution_mode='dag' and dynamic_targets=True.

    Example:
        ```python
        plan = DelegationPlan(
            steps=[
                DelegationStep(id="s1", agent_name="search", input={"q": "A"}),
                DelegationStep(id="s2", agent_name="search", input={"q": "B"}),
                DelegationStep(id="s3", agent_name="synthesizer", input={}, depends_on=["s1", "s2"]),
            ]
        )
        ```
    """

    model_config = ConfigDict(frozen=True)

    steps: list[DelegationStep]
    """List of steps in the delegation plan."""

    @field_validator("steps")
    @classmethod
    def validate_no_duplicate_ids(cls, steps: list[DelegationStep]) -> list[DelegationStep]:
        """Ensure all step IDs are unique."""
        ids = [step.id for step in steps]
        duplicates = [id_ for id_ in ids if ids.count(id_) > 1]
        if duplicates:
            raise ValueError(f"Duplicate step IDs found: {set(duplicates)}")
        return steps

    @model_validator(mode="after")
    def validate_dependencies(self) -> "DelegationPlan":
        """Validate that all dependencies reference existing steps and there are no cycles."""
        step_ids = {step.id for step in self.steps}

        # Check all depends_on reference existing steps
        for step in self.steps:
            for dep_id in step.depends_on:
                if dep_id not in step_ids:
                    raise ValueError(
                        f"Step '{step.id}' depends on '{dep_id}' which does not exist"
                    )

        # Check for cycles using DFS
        self._check_for_cycles(step_ids)

        return self

    def _check_for_cycles(self, step_ids: set[str]) -> None:
        """Detect cycles in the dependency graph using DFS."""
        # Build adjacency list
        graph: dict[str, list[str]] = {step.id: list(step.depends_on) for step in self.steps}

        visited: set[str] = set()
        rec_stack: set[str] = set()

        def dfs(node: str) -> bool:
            """Returns True if a cycle is detected."""
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for step_id in step_ids:
            if step_id not in visited:
                if dfs(step_id):
                    raise ValueError("Cycle detected in step dependencies")


class DelegationNodeConfig(BaseModel):
    """Configuration for a DelegationNode that delegates to other agents.

    DelegationNode sends tasks to other agents via a message transport
    (e.g. RabbitMQ) and waits for their responses. It supports sequential,
    parallel, and DAG execution modes.

    Example (sequential delegation):
        config = DelegationNodeConfig(
            name="delegate_analysis",
            description="Delegates to analyzer and summarizer",
            targets=[
                DelegationTarget(agent_name="analyzer"),
                DelegationTarget(agent_name="summarizer"),
            ],
            execution_mode="sequential",
        )

    Example (parallel delegation):
        config = DelegationNodeConfig(
            name="parallel_processors",
            description="Delegates to multiple processors simultaneously",
            targets=[
                DelegationTarget(agent_name="processor_a"),
                DelegationTarget(agent_name="processor_b"),
                DelegationTarget(agent_name="processor_c"),
            ],
            execution_mode="parallel",
            global_timeout_seconds=300,
        )

    Example (DAG delegation with dynamic targets):
        config = DelegationNodeConfig(
            name="dag_executor",
            description="Executes a plan with dependencies",
            targets=[],  # Ignored when dynamic_targets=True
            execution_mode="dag",
            dynamic_targets=True,
            default_failure_policy="skip_dependents",
            failure_overrides={"critical-agent": "fail_plan"},
        )
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    name: str
    """Unique name identifying this node within the agent."""

    description: str | None = None
    """Human-readable description of what this node does."""

    # Delegation configuration
    targets: list[DelegationTarget] = []
    """List of agents to delegate to. Ignored when dynamic_targets=True."""

    execution_mode: Literal["sequential", "parallel", "dag"] = "sequential"
    """Execution mode for delegation:
    - 'sequential': Execute targets one at a time, in order.
    - 'parallel': Execute all targets simultaneously.
    - 'dag': Execute according to dependency graph (supports depends_on)."""

    dynamic_targets: bool = False
    """If True, targets come from the input at runtime (as DelegationPlan).
    If False, uses the static targets from config."""

    default_failure_policy: Literal["fail_plan", "skip_dependents", "continue_partial"] = "fail_plan"
    """Default policy when a step fails:
    - 'fail_plan': Cancel entire plan, raise DelegationPlanError.
    - 'skip_dependents': Skip steps that depend on the failed step.
    - 'continue_partial': Continue execution, dependents run without failed step's result."""

    failure_overrides: dict[str, Literal["fail_plan", "skip_dependents", "continue_partial"]] = {}
    """Per-agent failure policy overrides. Key is agent_name."""

    global_timeout_seconds: int = 600
    """Maximum total time in seconds to wait for all responses."""

    reply_routing_key: str | None = None
    """Routing key for the replies queue. If None, generated as
    'agent.{current_agent_name}.replies'."""

    retry_on_timeout: bool = False
    """Whether to retry when a target times out."""

    on_error: Literal["raise", "continue"] = "raise"
    """Behavior when a delegation target fails (sequential/parallel modes only):
    - 'raise': Re-raise the exception immediately (default, preserves current behavior).
    - 'continue': Capture the error as a DelegationResponse-shaped dict in the results
      so the orchestrating LLM can see it and decide what to do.
    DAG mode is not affected — it uses its own failure_policies."""

    output_field: str | None = None
    """When set, _run() merges delegation results into the original input_data
    under this key instead of replacing it. This preserves pipeline context
    for downstream nodes.
    When None (default), _run() returns only the delegation results (original behavior)."""

    # Retry configuration (optional)
    retry: NodeRetryConfig | None = None
    """Retry settings for this node."""

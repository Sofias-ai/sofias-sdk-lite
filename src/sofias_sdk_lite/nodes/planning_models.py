"""Data models for plan-based execution.

This module provides the models used by PlannerNode and PlanExecutor
to define, validate, and report on execution plans.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from sofias_sdk_lite.nodes.base_node import NodeType

__all__ = ["ExecutionPlan", "NodeDescriptor", "PlanResult", "PlanStep"]


class PlanStep(BaseModel):
    """A single step in an execution plan."""

    model_config = ConfigDict(frozen=True)

    id: str
    """Unique identifier for this step within the plan."""

    node_name: str
    """Name of the node to execute (must be pre-registered in the agent)."""

    input: dict[str, Any] = {}
    """Input data for the node."""

    depends_on: list[str] = []
    """IDs of steps that must complete before this one can start."""

    description: str | None = None
    """Human-readable description of what this step does."""


class ExecutionPlan(BaseModel):
    """Complete execution plan produced by a PlannerNode.

    Validated as a DAG: unique step IDs, valid dependency references,
    and no cycles.
    """

    model_config = ConfigDict(frozen=True)

    steps: list[PlanStep]
    """Ordered list of steps to execute."""

    metadata: dict[str, Any] = {}
    """Plan metadata (reasoning, confidence, etc.)."""

    @field_validator("steps")
    @classmethod
    def validate_no_duplicate_ids(cls, steps: list[PlanStep]) -> list[PlanStep]:
        """Ensure all step IDs are unique."""
        ids = [step.id for step in steps]
        duplicates = [sid for sid in ids if ids.count(sid) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate step IDs found: {sorted(set(duplicates))}"
            )
        return steps

    @model_validator(mode="after")
    def validate_dependencies(self) -> ExecutionPlan:
        """Validate that all dependencies reference existing steps and detect cycles."""
        step_ids = {step.id for step in self.steps}

        for step in self.steps:
            for dep_id in step.depends_on:
                if dep_id not in step_ids:
                    raise ValueError(
                        f"Step '{step.id}' depends on unknown step '{dep_id}'"
                    )

        self._check_for_cycles(step_ids)
        return self

    def _check_for_cycles(self, step_ids: set[str]) -> None:
        """DFS-based cycle detection."""
        adjacency: dict[str, list[str]] = {sid: [] for sid in step_ids}
        for step in self.steps:
            for dep_id in step.depends_on:
                adjacency[dep_id].append(step.id)

        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {sid: WHITE for sid in step_ids}

        def dfs(node: str) -> None:
            color[node] = GRAY
            for neighbor in adjacency[node]:
                if color[neighbor] == GRAY:
                    raise ValueError(
                        f"Cycle detected in execution plan involving step '{neighbor}'"
                    )
                if color[neighbor] == WHITE:
                    dfs(neighbor)
            color[node] = BLACK

        for sid in step_ids:
            if color[sid] == WHITE:
                dfs(sid)


class PlanResult(BaseModel):
    """Result of executing an ExecutionPlan via PlanExecutor."""

    model_config = ConfigDict(frozen=True)

    results: dict[str, dict[str, Any]] = {}
    """Map of step_id to output dict for completed steps."""

    failed: dict[str, dict[str, Any]] = {}
    """Map of step_id to error info for failed steps."""

    skipped: list[str] = []
    """Step IDs that were skipped due to failed dependencies."""

    all_completed: bool = True
    """Whether all steps completed successfully."""


class NodeDescriptor(BaseModel):
    """Description of an available node for the PlannerNode's LLM prompt."""

    model_config = ConfigDict(frozen=True)

    name: str
    """Node name as registered in the agent."""

    node_type: NodeType
    """Type classification of the node."""

    description: str | None = None
    """Human-readable description of what this node does."""

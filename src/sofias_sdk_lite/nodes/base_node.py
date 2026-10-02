"""Base node abstraction for the Sofias Agents SDK.

This module provides the BaseNode abstract class, which defines the common
interface for all node types in the SDK. Specific implementations (LLMNode,
FunctionNode) inherit from this base class.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from enum import Enum
from typing import TYPE_CHECKING, Any

from sofias_sdk_lite.contracts import NodeContract

if TYPE_CHECKING:
    # NOTE: streaming event types live in sofias_sdk_lite.agent (Agent, AgentBuilder,
    # ExecutionContext, streaming) per the public SDK layout. That module is being
    # vendored separately and may not exist yet at import time.
    from sofias_sdk_lite.agent import StreamEvent

__all__ = ["BaseNode", "NodeType"]


class NodeType(str, Enum):
    """Type classification for nodes in the agent graph."""

    LLM = "llm"
    FUNCTION = "function"
    DELEGATION = "delegation"
    PLANNER = "planner"
    AGGREGATOR = "aggregator"


class BaseNode(ABC):
    """Abstract base class for all node types in the Agent SDK.

    BaseNode defines the common interface that all nodes must implement.
    It provides a template method pattern for execution: the public execute()
    method handles input/output validation, while subclasses implement the
    _run() method with their specific logic.

    This design allows the router, error handler, and other components to
    work with any node type uniformly through the execute() interface.

    Subclasses:
        - LLMNode: Nodes that invoke an LLM with optional tool loop.
        - FunctionNode: Nodes that execute deterministic Python logic.

    Example:
        ```python
        class MyCustomNode(BaseNode):
            async def _run(self, input_data: dict, context: dict | None = None) -> dict:
                # Custom logic here
                return {"result": "processed"}
        ```
    """

    def __init__(
        self,
        name: str,
        contract: NodeContract,
        description: str | None = None,
    ) -> None:
        """Initialize the base node.

        Args:
            name: Unique name identifying this node within the agent.
            contract: Input/output contract for validation.
            description: Human-readable description of what this node does.
        """
        self._name = name
        self._contract = contract
        self._description = description

    @property
    def name(self) -> str:
        """The node's unique name."""
        return self._name

    @property
    def contract(self) -> NodeContract:
        """The node's input/output contract."""
        return self._contract

    @property
    def node_type(self) -> NodeType:
        """The type of this node. Subclasses override to declare their type."""
        return NodeType.FUNCTION

    @property
    def description(self) -> str | None:
        """Human-readable description of the node."""
        return self._description

    async def execute(
        self,
        input_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the node with the given input.

        This is the main entry point for node execution. It implements a
        template method pattern:
        1. Validates input against the contract
        2. Calls _run() (implemented by subclass)
        3. Validates output against the contract
        4. Returns the validated result

        Args:
            input_data: Input dictionary to process.
            context: Optional context dictionary with additional information.

        Returns:
            Output dictionary conforming to the output contract.

        Raises:
            InputValidationError: If input validation fails.
            OutputValidationError: If output validation fails.
            NodeExecutionError: If execution fails in _run().
        """
        # Step 1: Validate input
        validated_input = self._contract.validate_input(input_data)

        # Step 2: Execute subclass-specific logic
        output_data = await self._run(validated_input.model_dump(), context)

        # Step 3: Validate output
        validated_output = self._contract.validate_output(output_data)

        # Step 4: Return validated result
        return validated_output.model_dump()

    async def stream_execute(
        self,
        input_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute the node with streaming.

        For LLMNodes that implement _stream_run(), yields streaming events
        as they occur. For other node types, falls back to _run() and yields
        only NodeStartEvent and NodeCompleteEvent.

        Args:
            input_data: Input dictionary to process.
            context: Optional context dictionary with additional information.

        Yields:
            StreamEvent instances for each notable event during execution.
        """
        from sofias_sdk_lite.agent import (
            NodeCompleteEvent,
            NodeStartEvent,
            StreamEvent,
        )

        # Step 1: Validate input
        validated_input = self._contract.validate_input(input_data)

        # Step 2: Yield node start
        start_time = time.perf_counter()
        yield NodeStartEvent(
            node_name=self._name,
            node_type=self.node_type.value,
        )

        # Step 3: Execute with streaming if available, otherwise fallback
        stream_run = getattr(self, "_stream_run", None)
        if stream_run is not None:
            async for event in stream_run(validated_input.model_dump(), context):
                yield event
            output_data: dict[str, Any] = getattr(self, "_last_stream_output", None) or {}
        else:
            output_data = await self._run(validated_input.model_dump(), context)

        # Step 4: Validate output
        validated_output = self._contract.validate_output(output_data)
        final_output = validated_output.model_dump()

        # Step 5: Yield node complete
        duration_ms = (time.perf_counter() - start_time) * 1000
        yield NodeCompleteEvent(
            node_name=self._name,
            node_type=self.node_type.value,
            output=final_output,
            duration_ms=round(duration_ms, 2),
        )

    @abstractmethod
    async def _run(
        self,
        input_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the node's specific logic.

        Subclasses must implement this method with their specific execution
        logic. Input has already been validated when this method is called.

        Args:
            input_data: Validated input dictionary.
            context: Optional context dictionary with additional information.

        Returns:
            Output dictionary (will be validated by execute()).

        Raises:
            NodeExecutionError: If execution fails.
        """
        ...

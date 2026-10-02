"""FunctionNode implementation for the Sofias Agents SDK.

This module provides the FunctionNode class, a node type for deterministic
Python logic without LLM involvement.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import TYPE_CHECKING, Any, Callable

from sofias_sdk_lite.contracts import NodeContract
from sofias_sdk_lite.errors import NodeExecutionError
from sofias_sdk_lite.nodes.base_node import BaseNode, NodeType
from sofias_sdk_lite.observability._log import get_logger

if TYPE_CHECKING:
    # Ephemeral, flow-scoped conversation state (not the long-lived memory
    # stack, which is explicitly out of scope for v1 — see MemoryProvider).
    from sofias_sdk_lite.state import ConversationStateProvider

logger = get_logger("nodes.function_node")

__all__ = ["FunctionNode", "ProcessFn"]


# Type alias for the process function
ProcessFn = Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]]
"""Type alias for a process function: takes (input_data, context) and returns output dict."""


class FunctionNode(BaseNode):
    """A node that executes deterministic Python logic without an LLM.

    FunctionNode is designed for pure data transformations, filtering,
    validation, merging outputs, or any logic that doesn't require an LLM.
    It inherits input/output validation from BaseNode.

    There are two ways to use FunctionNode:

    1. Pass a callable to the constructor (for simple transformations):

        def format_data(input_data: dict, context: dict | None = None) -> dict:
            return {"formatted": input_data["raw"].upper()}

        node = FunctionNode(
            name="formatter",
            contract=contract,
            process_fn=format_data,
        )

    2. Subclass for complex logic:

        class DataMerger(FunctionNode):
            def __init__(self, name: str, contract: NodeContract):
                super().__init__(name=name, contract=contract)
                self._cache = {}

            async def process(self, input_data: dict, context: dict | None = None) -> dict:
                # Complex logic with state
                return {"merged": ...}

    Example:
        ```python
        node = FunctionNode(
            name="transformer",
            contract=NodeContract(input_schema=RawInput, output_schema=FormattedOutput),
            process_fn=lambda data, ctx: {"value": data["input"] * 2},
        )
        result = await node.execute({"input": 5})
        # result: {"value": 10}
        ```
    """

    def __init__(
        self,
        name: str,
        contract: NodeContract,
        process_fn: ProcessFn | None = None,
        description: str | None = None,
    ) -> None:
        """Initialize the function node.

        Args:
            name: Unique name identifying this node within the agent.
            contract: Input/output contract for validation.
            process_fn: Optional function that executes the node logic.
                If not provided, subclass must override process().
            description: Human-readable description of what this node does.
        """
        super().__init__(name=name, contract=contract, description=description)
        self._process_fn = process_fn
        self._conversation_state: ConversationStateProvider | None = None

    @property
    def node_type(self) -> NodeType:
        return NodeType.FUNCTION

    async def _run(
        self,
        input_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the function node logic.

        Delegates to either the provided process_fn or the process() method.
        Input/output validation is handled by the parent BaseNode class.

        Args:
            input_data: Validated input dictionary.
            context: Optional context dictionary.

        Returns:
            Output dictionary from function execution.

        Raises:
            NodeExecutionError: If execution fails.
        """
        start_time = time.perf_counter()

        logger.info(
            "Node execution started",
            node_name=self.name,
            node_type="function",
        )

        try:
            if self._process_fn is not None:
                result = await self._invoke_process_fn(input_data, context)
            else:
                result = await self.process(input_data, context)

            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "Node execution completed",
                node_name=self.name,
                node_type="function",
                duration_ms=round(duration_ms, 2),
            )

            return result

        except NotImplementedError:
            raise
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Node execution failed",
                node_name=self.name,
                node_type="function",
                duration_ms=round(duration_ms, 2),
                error_type=type(e).__name__,
                error=str(e),
            )
            raise NodeExecutionError(
                f"Function node '{self.name}' execution failed: {e}",
                node_name=self.name,
                cause=e,
            ) from e

    async def _invoke_process_fn(
        self,
        input_data: dict[str, Any],
        context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Invoke the process function, handling both sync and async callables.

        Args:
            input_data: Validated input dictionary.
            context: Optional context dictionary.

        Returns:
            Output dictionary from the process function.
        """
        assert self._process_fn is not None
        if inspect.iscoroutinefunction(self._process_fn):
            return await self._process_fn(input_data, context)
        else:
            # Run sync function in thread pool to avoid blocking
            return await asyncio.to_thread(self._process_fn, input_data, context)

    # ------------------------------------------------------------------
    # Conversation state helpers (multi-turn flow persistence)
    # ------------------------------------------------------------------

    def _resolve_conversation_id(self) -> str:
        """Resolve the conversation ID from the current ExecutionContext."""
        from sofias_sdk_lite.agent import get_execution_context

        ctx = get_execution_context()
        if ctx is not None:
            return ctx.metadata.get("conversation_id", ctx.execution_id)
        return "unknown"

    async def get_state(self, key: str) -> dict[str, Any] | None:
        """Retrieve persisted state for the current conversation.

        Reads the conversation_id automatically from ExecutionContext.
        Requires that AgentBuilder.with_conversation_state() was called.

        Args:
            key: State key (e.g. "validation:doc123").

        Returns:
            The stored dict, or None if not found.

        Raises:
            NodeExecutionError: If no conversation state is configured.
        """
        if self._conversation_state is None:
            raise NodeExecutionError(
                f"No conversation state configured for node '{self.name}'. "
                "Use AgentBuilder.with_conversation_state() to configure.",
                node_name=self.name,
            )
        conv_id = self._resolve_conversation_id()
        return await self._conversation_state.get(conv_id, key)

    async def set_state(self, key: str, value: dict[str, Any]) -> None:
        """Persist state for the current conversation.

        Reads the conversation_id automatically from ExecutionContext.
        Requires that AgentBuilder.with_conversation_state() was called.

        Args:
            key: State key (e.g. "validation:doc123").
            value: Dict to persist. Overwrites any existing value.

        Raises:
            NodeExecutionError: If no conversation state is configured.
        """
        if self._conversation_state is None:
            raise NodeExecutionError(
                f"No conversation state configured for node '{self.name}'. "
                "Use AgentBuilder.with_conversation_state() to configure.",
                node_name=self.name,
            )
        conv_id = self._resolve_conversation_id()
        await self._conversation_state.set(conv_id, key, value)

    async def delete_state(self, key: str) -> None:
        """Delete persisted state for the current conversation.

        Reads the conversation_id automatically from ExecutionContext.
        Requires that AgentBuilder.with_conversation_state() was called.

        Args:
            key: State key to delete.

        Raises:
            NodeExecutionError: If no conversation state is configured.
        """
        if self._conversation_state is None:
            raise NodeExecutionError(
                f"No conversation state configured for node '{self.name}'. "
                "Use AgentBuilder.with_conversation_state() to configure.",
                node_name=self.name,
            )
        conv_id = self._resolve_conversation_id()
        await self._conversation_state.delete(conv_id, key)

    # ------------------------------------------------------------------

    async def process(
        self,
        input_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the node's processing logic.

        Override this method in subclasses to implement custom logic.
        This method is called when no process_fn is provided to the constructor.

        Args:
            input_data: Validated input dictionary.
            context: Optional context dictionary.

        Returns:
            Output dictionary.

        Raises:
            NotImplementedError: If neither process_fn was provided nor
                this method was overridden in a subclass.
        """
        raise NotImplementedError(
            f"FunctionNode '{self.name}' has no implementation. "
            "Provide process_fn in the constructor or subclass FunctionNode "
            "and override the process() method."
        )

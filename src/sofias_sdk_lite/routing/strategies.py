"""Routing strategies for deciding the next node in the graph.

This module provides a set of built-in routing strategies that can be used
to determine which node should be executed next based on the output of
the current node. Strategies are pure decision functions that can be
tested in isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol

from sofias_sdk_lite.errors.exceptions import RoutingError

if TYPE_CHECKING:
    from sofias_sdk_lite.contracts.node_contracts import OutputContract

__all__ = [
    "RoutingStrategy",
    "FieldValueStrategy",
    "ConfidenceThresholdStrategy",
    "FieldPresenceStrategy",
    "ConditionalStrategy",
    "FanOutStrategy",
    "CompositeStrategy",
    "StaticRoute",
]


class RoutingStrategy(Protocol):
    """Protocol defining the interface for routing strategies.

    A routing strategy receives the output from a node (already validated
    against its OutputContract) and decides which node should be executed next.
    """

    def decide(self, output: dict[str, Any], contract: type[OutputContract] | None = None) -> str:
        """Decide the next node based on the output.

        Args:
            output: The validated output from the current node.
            contract: The output contract type (for type-based decisions).

        Returns:
            The name of the next node to execute.

        Raises:
            RoutingError: If a routing decision cannot be made.
        """
        ...


class FieldValueStrategy:
    """Routes based on the value of a specific field in the output.

    This strategy maps field values to target nodes. Useful for classification
    outputs where different categories lead to different processing paths.

    Example:
        strategy = FieldValueStrategy(
            field="category",
            mapping={"urgent": "priority_handler", "spam": "discard"},
            default="normal_handler",
        )
        # If output["category"] == "urgent" -> "priority_handler"
        # If output["category"] == "spam" -> "discard"
        # If output["category"] == "other" -> "normal_handler" (default)
    """

    def __init__(
        self,
        field: str,
        mapping: dict[Any, str],
        default: str | None = None,
    ) -> None:
        """Initialize the field value strategy.

        Args:
            field: The name of the field to check in the output.
            mapping: Dictionary mapping field values to target node names.
            default: Default node name if the field value is not in mapping.
                If None and value not found, raises RoutingError.
        """
        self._field = field
        self._mapping = mapping
        self._default = default

    def decide(self, output: dict[str, Any], contract: type[OutputContract] | None = None) -> str:
        """Decide the next node based on the field value.

        Args:
            output: The validated output from the current node.
            contract: The output contract type (unused by this strategy).

        Returns:
            The name of the next node.

        Raises:
            RoutingError: If the field doesn't exist or value is not mapped
                and no default is configured.
        """
        if self._field not in output:
            raise RoutingError(
                f"Field '{self._field}' not found in output",
                source_node="unknown",
                available_targets=list(self._mapping.values()),
            )

        value = output[self._field]
        if value in self._mapping:
            return self._mapping[value]

        if self._default is not None:
            return self._default

        raise RoutingError(
            f"Value '{value}' for field '{self._field}' not in mapping and no default configured",
            source_node="unknown",
            available_targets=list(self._mapping.values()),
        )


class ConfidenceThresholdStrategy:
    """Routes based on whether a numeric field exceeds a threshold.

    Useful for confidence scores or probability outputs where high confidence
    leads to automated processing and low confidence requires human review.

    Example:
        strategy = ConfidenceThresholdStrategy(
            field="confidence",
            threshold=0.8,
            above_or_equal="auto_respond",
            below="human_review",
        )
        # If output["confidence"] >= 0.8 -> "auto_respond"
        # If output["confidence"] < 0.8 -> "human_review"
    """

    def __init__(
        self,
        field: str,
        threshold: float,
        above_or_equal: str,
        below: str,
    ) -> None:
        """Initialize the confidence threshold strategy.

        Args:
            field: The name of the numeric field to check.
            threshold: The threshold value for comparison.
            above_or_equal: Target node if value >= threshold.
            below: Target node if value < threshold.
        """
        self._field = field
        self._threshold = threshold
        self._above_or_equal = above_or_equal
        self._below = below

    def decide(self, output: dict[str, Any], contract: type[OutputContract] | None = None) -> str:
        """Decide the next node based on the threshold comparison.

        Args:
            output: The validated output from the current node.
            contract: The output contract type (unused by this strategy).

        Returns:
            The name of the next node.

        Raises:
            RoutingError: If the field doesn't exist or is not numeric.
        """
        if self._field not in output:
            raise RoutingError(
                f"Field '{self._field}' not found in output",
                source_node="unknown",
                available_targets=[self._above_or_equal, self._below],
            )

        value = output[self._field]
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as e:
            raise RoutingError(
                f"Field '{self._field}' value '{value}' is not numeric",
                source_node="unknown",
                available_targets=[self._above_or_equal, self._below],
            ) from e

        if numeric_value >= self._threshold:
            return self._above_or_equal
        return self._below


class FieldPresenceStrategy:
    """Routes based on whether a field exists and has a truthy value.

    Useful for checking if optional outputs are present. Considers a field
    "present" if it exists, is not None, and is not empty (for collections).

    Example:
        strategy = FieldPresenceStrategy(
            field="tool_calls",
            present="tool_executor",
            absent="final_response",
        )
        # If output["tool_calls"] has content -> "tool_executor"
        # If output["tool_calls"] is None/empty -> "final_response"
    """

    def __init__(
        self,
        field: str,
        present: str,
        absent: str,
    ) -> None:
        """Initialize the field presence strategy.

        Args:
            field: The name of the field to check.
            present: Target node if field has a truthy value.
            absent: Target node if field is missing, None, or empty.
        """
        self._field = field
        self._present = present
        self._absent = absent

    def _is_present(self, value: Any) -> bool:
        """Check if a value is considered present.

        A value is present if:
        - It is not None
        - If it's a collection (list, dict, str, set), it's not empty

        Args:
            value: The value to check.

        Returns:
            True if the value is considered present.
        """
        if value is None:
            return False
        if isinstance(value, (list, dict, str, set, tuple)):
            return len(value) > 0
        return True

    def decide(self, output: dict[str, Any], contract: type[OutputContract] | None = None) -> str:
        """Decide the next node based on field presence.

        Args:
            output: The validated output from the current node.
            contract: The output contract type (unused by this strategy).

        Returns:
            The name of the next node.
        """
        if self._field not in output:
            return self._absent

        if self._is_present(output[self._field]):
            return self._present
        return self._absent


class ConditionalStrategy:
    """Routes using a custom callable function.

    This is the escape hatch when none of the built-in strategies fit.
    The developer provides their own function that receives the output
    and returns the target node name.

    Example:
        def custom_router(output: dict) -> str:
            if output["score"] > 0.9 and output["verified"]:
                return "premium_handler"
            return "standard_handler"

        strategy = ConditionalStrategy(condition=custom_router)
    """

    def __init__(
        self,
        condition: Callable[[dict[str, Any]], str],
        targets: list[str] | None = None,
    ) -> None:
        """Initialize the conditional strategy.

        Args:
            condition: A callable that takes the output dict and returns
                the target node name.
            targets: Optional list of all possible target node names that
                the condition may return. Required when compiling to
                LangGraph so the adapter can build the path map.
        """
        self._condition = condition
        self._targets = targets

    def decide(self, output: dict[str, Any], contract: type[OutputContract] | None = None) -> str:
        """Decide the next node using the custom condition.

        Args:
            output: The validated output from the current node.
            contract: The output contract type (unused by this strategy).

        Returns:
            The name of the next node.

        Raises:
            RoutingError: If the condition function raises an exception.
        """
        try:
            return self._condition(output)
        except Exception as e:
            raise RoutingError(
                f"Conditional routing function failed: {e}",
                source_node="unknown",
            ) from e


class FanOutStrategy:
    """Routes to multiple nodes for parallel execution (fan-out).

    Unlike other strategies that return a single target, this strategy
    returns a list of nodes to be executed concurrently. A join node
    collects the results after all parallel branches complete.

    Example:
        strategy = FanOutStrategy(
            targets=["sentiment", "entities", "summary"],
            join_node="merger",
        )
        # Dispatches to all three nodes in parallel.
        # After all complete, execution continues at "merger".
    """

    def __init__(
        self,
        targets: list[str],
        join_node: str,
        on_error: str = "fail_all",
        timeout_seconds: int = 300,
    ) -> None:
        """Initialize the fan-out strategy.

        Args:
            targets: List of node names to execute in parallel.
            join_node: Node that receives merged results after fan-out.
            on_error: Error policy — "fail_all" cancels remaining on
                first failure, "continue_partial" waits for all and
                collects partial results.
            timeout_seconds: Maximum time for all parallel branches.
        """
        if len(targets) < 2:
            raise ValueError("FanOutStrategy requires at least 2 targets")
        if join_node in targets:
            raise ValueError(
                f"join_node '{join_node}' cannot be one of the fan-out targets"
            )
        self._targets = list(targets)
        self._join_node = join_node
        self._on_error = on_error
        self._timeout_seconds = timeout_seconds

    def decide(self, output: dict[str, Any], contract: type[OutputContract] | None = None) -> list[str]:
        """Return the list of parallel targets.

        Args:
            output: The validated output from the current node.
            contract: The output contract type (unused by this strategy).

        Returns:
            List of target node names for parallel execution.
        """
        return list(self._targets)

    @property
    def targets(self) -> list[str]:
        """The parallel target nodes."""
        return list(self._targets)

    @property
    def join_node(self) -> str:
        """The node where parallel branches converge."""
        return self._join_node

    @property
    def on_error(self) -> str:
        """Error policy for parallel execution."""
        return self._on_error

    @property
    def timeout_seconds(self) -> int:
        """Timeout for parallel execution."""
        return self._timeout_seconds


class CompositeStrategy:
    """Executes multiple strategies in order, using the first valid result.

    Strategies are tried in order. The first strategy that returns a result
    (doesn't raise RoutingError) wins. If all strategies fail, uses the
    default.

    Example:
        strategy = CompositeStrategy(
            strategies=[
                FieldPresenceStrategy("error", "error_handler", None),
                ConfidenceThresholdStrategy("confidence", 0.9, "auto", "manual"),
            ],
            default="fallback",
        )
    """

    def __init__(
        self,
        strategies: list[RoutingStrategy],
        default: str,
    ) -> None:
        """Initialize the composite strategy.

        Args:
            strategies: List of strategies to try in order.
            default: Required default node if all strategies fail to decide.
        """
        self._strategies = strategies
        self._default = default

    def decide(self, output: dict[str, Any], contract: type[OutputContract] | None = None) -> str:
        """Decide the next node by trying strategies in order.

        Args:
            output: The validated output from the current node.
            contract: The output contract type (passed to sub-strategies).

        Returns:
            The name of the next node from the first successful strategy,
            or the default if all fail.
        """
        for strategy in self._strategies:
            try:
                result = strategy.decide(output, contract)
                if result is not None:
                    return result
            except RoutingError:
                # This strategy couldn't decide, try the next one
                continue

        return self._default


class StaticRoute:
    """Always routes to the same target node.

    Internal strategy used by add_edge() and default terminal routing.
    For conditional routing, use FieldValueStrategy or ConditionalStrategy.

    Example:
        strategy = StaticRoute("responder")
        # output is ignored — always returns "responder"
    """

    def __init__(self, target: str) -> None:
        """Initialize the static route.

        Args:
            target: Name of the node to always route to.
        """
        self._target = target

    @property
    def target(self) -> str:
        """The target node name."""
        return self._target

    def decide(self, output: dict[str, Any], contract: type[OutputContract] | None = None) -> str:
        """Always return the configured target node.

        Args:
            output: The validated output from the current node (ignored).
            contract: The output contract type (ignored).

        Returns:
            The target node name.
        """
        return self._target

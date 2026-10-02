"""Router for managing graph edges and node transitions.

This module provides the Router class which is the central registry for
edges in the agent graph. It connects nodes with routing strategies and
handles graph validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sofias_sdk_lite.errors.exceptions import RoutingError
from sofias_sdk_lite.observability._log import get_logger

if TYPE_CHECKING:
    from sofias_sdk_lite.routing.strategies import FanOutStrategy, RoutingStrategy

__all__ = ["Router"]

logger = get_logger("routing.router")


class Router:
    """Central registry for graph edges and routing decisions.

    The Router manages the mapping from source nodes to routing strategies,
    determining which node should be executed next based on the output of
    the current node.

    The router does NOT execute nodes. It only decides the path through the
    graph. Execution is handled by the agent (Phase 5).

    Example:
        ```python
        router = Router()
        router.add_route(
            "classifier",
            FieldValueStrategy(
                field="category",
                mapping={"urgent": "priority_handler", "spam": "discard"},
                default="normal_handler",
            ),
        )
        router.add_terminal("priority_handler")
        router.add_terminal("discard")
        router.add_terminal("normal_handler")

        # Later, during execution:
        next_node = router.resolve("classifier", {"category": "urgent"})
        # Returns "priority_handler"
        ```
    """

    def __init__(self) -> None:
        """Initialize the router."""
        self._routes: dict[str, RoutingStrategy] = {}
        self._terminals: set[str] = set()

    def add_route(self, from_node: str, strategy: RoutingStrategy) -> None:
        """Register a routing strategy for a source node.

        Args:
            from_node: The name of the source node.
            strategy: The routing strategy to use when this node produces output.

        Raises:
            ValueError: If the node already has a route or is marked as terminal.
        """
        if from_node in self._routes:
            raise ValueError(f"Node '{from_node}' already has a route registered")
        if from_node in self._terminals:
            raise ValueError(f"Node '{from_node}' is marked as terminal and cannot have routes")
        self._routes[from_node] = strategy

    def add_terminal(self, node_name: str) -> None:
        """Mark a node as terminal (no outgoing edges).

        Terminal nodes represent the end of the graph execution.
        They cannot have routes registered.

        Args:
            node_name: The name of the terminal node.

        Raises:
            ValueError: If the node already has a route registered.
        """
        if node_name in self._routes:
            raise ValueError(f"Node '{node_name}' already has a route and cannot be terminal")
        self._terminals.add(node_name)

    def resolve(self, from_node: str, output: dict[str, Any]) -> str | list[str] | None:
        """Resolve the next node given a source node and its output.

        Args:
            from_node: The name of the node that produced the output.
            output: The validated output from the source node.

        Returns:
            The name of the next node to execute, a list of node names
            for parallel fan-out, or None if the source node is terminal.

        Raises:
            RoutingError: If the node has no route and is not terminal,
                or if the strategy fails to decide.
        """
        # Terminal nodes return None (end of graph)
        if from_node in self._terminals:
            logger.debug(
                "Routing to terminal",
                source_node=from_node,
            )
            return None

        # Non-terminal nodes must have a route
        if from_node not in self._routes:
            logger.error(
                "No route registered for non-terminal node",
                source_node=from_node,
                available_terminals=list(self._terminals),
            )
            raise RoutingError(
                f"Node '{from_node}' has no route registered and is not terminal",
                source_node=from_node,
                available_targets=list(self._terminals),
            )

        strategy = self._routes[from_node]
        try:
            result = strategy.decide(output)
            if isinstance(result, list):
                logger.info(
                    "Fan-out route resolved",
                    source_node=from_node,
                    targets=result,
                    strategy=type(strategy).__name__,
                )
            else:
                logger.info(
                    "Route resolved",
                    source_node=from_node,
                    next_node=result,
                    strategy=type(strategy).__name__,
                )
            return result
        except RoutingError:
            logger.error(
                "Routing decision failed",
                source_node=from_node,
                strategy=type(strategy).__name__,
            )
            # Re-raise with source_node context if missing
            raise
        except Exception as e:
            logger.error(
                "Routing strategy error",
                source_node=from_node,
                strategy=type(strategy).__name__,
                error=str(e),
            )
            raise RoutingError(
                f"Routing strategy failed for node '{from_node}': {e}",
                source_node=from_node,
            ) from e

    def get_fan_out_config(self, from_node: str) -> FanOutStrategy | None:
        """Get the FanOutStrategy for a node, if it has one.

        Args:
            from_node: The name of the source node.

        Returns:
            The FanOutStrategy if the node routes via fan-out, else None.
        """
        from sofias_sdk_lite.routing.strategies import FanOutStrategy

        strategy = self._routes.get(from_node)
        if isinstance(strategy, FanOutStrategy):
            return strategy
        return None

    def validate_graph(
        self,
        node_names: list[str],
        fan_out_targets: set[str] | None = None,
    ) -> list[str]:
        """Validate the graph structure for consistency.

        Performs static validation to check that:
        1. All non-terminal nodes have a route registered
        2. All route targets point to nodes that exist in the graph
        3. At least one terminal node exists

        Args:
            node_names: List of all node names in the graph.
            fan_out_targets: Optional set of node names that are fan-out
                targets (they don't need their own routes).

        Returns:
            List of validation warnings/errors. Empty list means valid.
        """
        errors: list[str] = []
        node_set = set(node_names)
        skip_route_check = fan_out_targets or set()

        # Check for at least one terminal
        if not self._terminals:
            errors.append("Graph has no terminal nodes defined")

        # Check all non-terminal nodes have routes (skip fan-out targets)
        for node in node_names:
            if node in skip_route_check:
                continue
            if node not in self._terminals and node not in self._routes:
                errors.append(f"Node '{node}' has no route and is not terminal")

        # Check all terminals are valid nodes
        for terminal in self._terminals:
            if terminal not in node_set:
                errors.append(f"Terminal '{terminal}' is not a valid node in the graph")

        # Check route targets point to valid nodes
        # We need to probe each strategy with possible outputs to find targets
        # For now, we collect known targets from strategy attributes when possible
        for source, strategy in self._routes.items():
            targets = self._extract_strategy_targets(strategy)
            for target in targets:
                if target not in node_set:
                    errors.append(
                        f"Route from '{source}' points to unknown node '{target}'"
                    )

        return errors

    def _extract_strategy_targets(self, strategy: RoutingStrategy) -> list[str]:
        """Extract known target nodes from a strategy.

        This is a best-effort extraction for validation purposes.
        Not all strategies expose their targets.

        Args:
            strategy: The routing strategy to inspect.

        Returns:
            List of target node names that could be returned by this strategy.
        """
        from sofias_sdk_lite.routing.strategies import ConditionalStrategy, FanOutStrategy

        targets: list[str] = []

        # FanOutStrategy
        if isinstance(strategy, FanOutStrategy):
            targets.extend(strategy.targets)
            targets.append(strategy.join_node)
            return targets

        # ConditionalStrategy
        if isinstance(strategy, ConditionalStrategy):
            cond_targets = getattr(strategy, "_targets", None)
            if cond_targets:
                targets.extend(cond_targets)
                return targets

        # FieldValueStrategy
        mapping = getattr(strategy, "_mapping", None)
        if mapping is not None:
            targets.extend(mapping.values())
        default = getattr(strategy, "_default", None)
        if default is not None:
            targets.append(default)

        # ConfidenceThresholdStrategy
        above = getattr(strategy, "_above_or_equal", None)
        if above is not None:
            targets.append(above)
        below = getattr(strategy, "_below", None)
        if below is not None:
            targets.append(below)

        # FieldPresenceStrategy
        present = getattr(strategy, "_present", None)
        if present is not None:
            targets.append(present)
        absent = getattr(strategy, "_absent", None)
        if absent is not None:
            targets.append(absent)

        # CompositeStrategy
        sub_strategies = getattr(strategy, "_strategies", None)
        if sub_strategies is not None:
            for sub_strategy in sub_strategies:
                targets.extend(self._extract_strategy_targets(sub_strategy))

        return targets

    @property
    def routes(self) -> dict[str, RoutingStrategy]:
        """Get a copy of the registered routes."""
        return dict(self._routes)

    @property
    def terminals(self) -> set[str]:
        """Get a copy of the terminal nodes."""
        return set(self._terminals)

"""AgentBuilder for declarative agent construction.

This module provides the AgentBuilder class, which implements the builder
pattern for constructing Agent instances in a clean, declarative way.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, Callable, Literal

from sofias_sdk_lite.agent.agent import Agent
from sofias_sdk_lite.agent.agent_config import AgentConfig, AgentLLMConfig
from sofias_sdk_lite.config.runtime_config import AgentSettings, SettingsResolver
from sofias_sdk_lite.config.sdk_config import SDKConfig
from sofias_sdk_lite.contracts import AgentContract, InputContract, NodeContract, OutputContract
from sofias_sdk_lite.errors.error_handler import (
    BackoffStrategy,
    CircuitBreakerConfig,
    ErrorHandler,
    ErrorHandlerConfig,
    FallbackConfig,
    RetryPolicy,
)
from sofias_sdk_lite.errors.exceptions import AgentSDKError
from sofias_sdk_lite.llm import LLMCallable
from sofias_sdk_lite.llm.factory import ENV_LLM_MODEL, resolve_default_llm
from sofias_sdk_lite.nodes.aggregator_config import AggregatorNodeConfig
from sofias_sdk_lite.nodes.base_node import BaseNode
from sofias_sdk_lite.nodes.delegation_config import DelegationNodeConfig
from sofias_sdk_lite.nodes.function_node import FunctionNode
from sofias_sdk_lite.nodes.llm_node import LLMNode
from sofias_sdk_lite.tools import BaseTool
from sofias_sdk_lite.routing.router import Router
from sofias_sdk_lite.routing.strategies import FanOutStrategy

if TYPE_CHECKING:
    from sofias_sdk_lite.agent.response_workflow import ResponseWorkflow
    from sofias_sdk_lite.nodes.delegation_node import DelegationNode
    from sofias_sdk_lite.nodes.delegation_transport import DelegationTransport
    from sofias_sdk_lite.nodes.llm_node_config import FunctionNodeConfig, LLMNodeConfig
    from sofias_sdk_lite.tools import ToolRegistry
    from sofias_sdk_lite.routing.strategies import RoutingStrategy

__all__ = ["AgentBuildError", "AgentBuilder"]


class AgentBuildError(AgentSDKError):
    """Error when building an agent fails validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        message = "Agent build failed with the following errors:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        super().__init__(message, errors=errors)


class AgentBuilder:
    """Builder for constructing Agent instances declaratively.

    The builder pattern allows configuring all aspects of an agent
    in a fluent, method-chaining style. Validation happens at build()
    time, catching configuration errors before runtime.

    Example:
        agent = (
            AgentBuilder("inbox_assist", version="1.0.0")
            .with_description("Agent that processes incoming emails")
            .with_sdk_config(sdk_config)
            .with_settings_class(InboxAssistSettings)
            .with_contract(input_schema=EmailInput, output_schema=EmailResponse)
            .add_llm_node("classifier", classifier_config, classifier_contract)
            .add_llm_node("drafter", drafter_config, drafter_contract)
            .add_function_node("merger", merger_contract, process_fn=merge_outputs)
            .set_entry_node("classifier")
            .add_route("classifier", FieldValueStrategy(
                field="category",
                mapping={"draft": "drafter", "review": "reviewer"},
            ))
            .set_terminal("drafter")
            .set_terminal("reviewer")
            .with_retry_policy(RetryPolicy(max_retries=3))
            .with_circuit_breaker(CircuitBreakerConfig(failure_threshold=5))
            .build()
        )
    """

    def __init__(self, name: str, *, version: str = "1.0.0") -> None:
        """Initialize the builder with agent identity.

        Args:
            name: Unique name identifying this agent.
            version: Semantic version of the agent (e.g., "1.0.0").
        """
        self._name = name
        self._version = version
        self._description: str | None = None

        # SDK and settings
        self._sdk_config: SDKConfig | None = None
        self._settings_class: type[AgentSettings] | None = None

        # Contract
        self._input_schema: type[InputContract] | None = None
        self._output_schema: type[OutputContract] | None = None

        # LLM Nodes
        self._llm_node_configs: dict[str, LLMNodeConfig] = {}
        self._llm_node_contracts: dict[str, NodeContract] = {}
        self._llm_node_tools: dict[str, list[BaseTool]] = {}
        self._llm_nodes: dict[str, LLMNode] = {}  # Pre-built LLM node instances
        self._llm: LLMCallable | None = None

        # Function Nodes
        self._function_nodes: dict[str, FunctionNode] = {}

        # Delegation Nodes
        self._delegation_node_configs: dict[str, DelegationNodeConfig] = {}
        self._delegation_node_contracts: dict[str, NodeContract] = {}
        self._delegation_node_mappers: dict[str, dict[str, Callable[[dict], dict]]] = {}

        # Aggregator Nodes
        self._aggregator_node_configs: dict[str, AggregatorNodeConfig] = {}
        self._aggregator_node_contracts: dict[str, NodeContract] = {}
        self._aggregator_retry_handlers: dict[str, Callable[[list[str]], Awaitable[list[str]]]] = {}

        # Transport for delegation nodes
        self._delegation_transport: DelegationTransport | None = None

        # All node names (for validation)
        self._all_node_names: set[str] = set()

        # Routing
        self._entry_node: str | None = None
        self._routes: dict[str, Any] = {}
        self._terminals: set[str] = set()

        # Error handling
        self._default_retry_policy: RetryPolicy | None = None
        self._node_retry_policies: dict[str, RetryPolicy] = {}
        self._fallbacks: dict[str, str] = {}
        self._circuit_breaker_config: CircuitBreakerConfig | None = None

        # Context
        self._context: dict[str, Any] = {}

        # Response workflow
        self._response_workflow: ResponseWorkflow | None = None

        # Middleware
        self._middleware: list[Any] = []

        # Tool registry
        self._tool_registry: ToolRegistry | None = None


        # Agent-level LLM config
        self._agent_llm_config: AgentLLMConfig | None = None

        # Agent-level system prompt (propagated to all LLM nodes)
        self._system_prompt: str | None = None

        # Fan-out configurations
        self._fan_out_targets: set[str] = set()

        # Planner Nodes
        self._planner_node_configs: dict[str, dict[str, Any]] = {}
        self._plan_only_nodes: set[str] = set()
        self._plan_failure_policy: Literal["fail_plan", "skip_dependents", "continue_partial"] = "fail_plan"
        self._plan_failure_overrides: dict[str, str] = {}
        self._plan_timeout_seconds: int = 600

        # Conversation state (multi-turn flow persistence)
        self._conversation_state: Any = None  # ConversationState | None

        # Default route to terminal (auto-route unrouted nodes)
        self._default_route_to_terminal: bool = False

    def with_description(self, description: str) -> AgentBuilder:
        """Set the agent's description.

        Args:
            description: Human-readable description of what this agent does.

        Returns:
            Self for method chaining.
        """
        self._description = description
        return self

    def with_sdk_config(self, config: SDKConfig) -> AgentBuilder:
        """Set the SDK configuration.

        Args:
            config: Global SDK configuration.

        Returns:
            Self for method chaining.
        """
        self._sdk_config = config
        return self

    def with_settings_class(self, cls: type[AgentSettings]) -> AgentBuilder:
        """Set the runtime settings class.

        Args:
            cls: The AgentSettings subclass for runtime configuration.

        Returns:
            Self for method chaining.
        """
        self._settings_class = cls
        return self

    def with_contract(
        self,
        input_schema: type[InputContract],
        output_schema: type[OutputContract],
    ) -> AgentBuilder:
        """Set the agent's input/output contract.

        Args:
            input_schema: Pydantic model class for agent input validation.
            output_schema: Pydantic model class for agent output validation.

        Returns:
            Self for method chaining.
        """
        self._input_schema = input_schema
        self._output_schema = output_schema
        return self

    _MISSING_LLM_MESSAGE = (
        "No LLM available for LLM nodes. Either call with_llm(), run the agent through "
        "AgentRunner with model_name/router_url/router_api_key in its settings, or export "
        f"{ENV_LLM_MODEL} (plus SOFIAS_LLM_BASE_URL / SOFIAS_LLM_API_KEY)."
    )

    def with_llm(self, llm: LLMCallable) -> AgentBuilder:
        """Set the LLM callable for all LLM nodes.

        Optional. When omitted, `build()` uses the LLM installed by
        `sofias_sdk_lite.llm.default_llm` (what `AgentRunner` does with the
        per-task settings) or, failing that, one built from the
        ``SOFIAS_LLM_*`` environment variables via
        `sofias_sdk_lite.llm.create_llm`.

        Args:
            llm: LLM implementation for making completions.

        Returns:
            Self for method chaining.
        """
        self._llm = llm
        return self

    def _needs_agent_llm(self) -> bool:
        """Whether any node still to be built relies on the agent-level LLM."""
        if self._llm_node_configs:
            return True
        return any(cfg["llm"] is None for cfg in self._planner_node_configs.values())

    def with_agent_llm_config(self, config: AgentLLMConfig) -> AgentBuilder:
        """Set agent-level LLM configuration.

        Args:
            config: Agent-level LLM settings.

        Returns:
            Self for method chaining.
        """
        self._agent_llm_config = config
        return self

    def with_system_prompt(self, prompt: str) -> AgentBuilder:
        """Set an agent-level system prompt propagated to all LLM nodes.

        Node-level system prompts (set via LLMNodeConfig.system_prompt)
        take precedence over this agent-level prompt.

        Args:
            prompt: System prompt to inject into every LLM call.

        Returns:
            Self for method chaining.
        """
        self._system_prompt = prompt
        return self

    def add_llm_node(
        self,
        name: str,
        node_config: LLMNodeConfig | None = None,
        contract: NodeContract | None = None,
        tools: list[BaseTool] | None = None,
        node: LLMNode | None = None,
    ) -> AgentBuilder:
        """Register an LLM node in the agent.

        There are two ways to register an LLM node:
        1. Provide node_config and contract (for standard LLMNode instances)
        2. Provide a pre-built LLMNode instance (for subclassed nodes)

        Args:
            name: Unique name for this node within the agent.
            node_config: Configuration for the LLM node (required if node not provided).
            contract: Input/output contract for the node (required if node not provided).
            tools: Optional list of tools available to this node.
            node: Optional pre-built LLMNode instance (for subclassed nodes).

        Returns:
            Self for method chaining.
        """
        if node is not None:
            # Use the provided node instance
            self._llm_nodes[name] = node
        else:
            # Store configuration for building later
            if node_config is None or contract is None:
                raise ValueError(
                    "Either 'node' or both 'node_config' and 'contract' must be provided."
                )
            self._llm_node_configs[name] = node_config
            self._llm_node_contracts[name] = contract
            if tools:
                self._llm_node_tools[name] = tools
        self._all_node_names.add(name)
        return self

    def add_function_node(
        self,
        name: str,
        contract: NodeContract,
        process_fn: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]] | None = None,
        config: FunctionNodeConfig | None = None,
        node: FunctionNode | None = None,
    ) -> AgentBuilder:
        """Register a FunctionNode in the agent.

        There are two ways to register a function node:
        1. Provide a process_fn callable (for simple transformations)
        2. Provide a FunctionNode instance (for subclassed nodes with complex logic)

        Args:
            name: Unique name for this node within the agent.
            contract: Input/output contract for the node.
            process_fn: Optional callable that executes the node logic.
            config: Optional configuration for the node.
            node: Optional pre-built FunctionNode instance (for subclassed nodes).

        Returns:
            Self for method chaining.
        """
        if node is not None:
            # Use the provided node instance
            self._function_nodes[name] = node
        else:
            # Build a new FunctionNode
            description = config.description if config else None
            self._function_nodes[name] = FunctionNode(
                name=name,
                contract=contract,
                process_fn=process_fn,
                description=description,
            )
        self._all_node_names.add(name)
        return self

    def with_delegation_transport(self, transport: DelegationTransport) -> AgentBuilder:
        """Set the transport for delegation nodes.

        The transport is required if the agent contains any DelegationNodes.
        It provides the send/receive capabilities for delegation communication.

        Args:
            transport: Transport implementing the DelegationTransport protocol.

        Returns:
            Self for method chaining.
        """
        self._delegation_transport = transport
        return self

    def add_delegation_node(
        self,
        name: str,
        config: DelegationNodeConfig,
        contract: NodeContract,
        input_mappers: dict[str, Callable[[dict], dict]] | None = None,
    ) -> AgentBuilder:
        """Register a DelegationNode in the agent.

        DelegationNode delegates tasks to other agents and waits for their
        responses. It requires a DelegationTransport to be configured
        via with_delegation_transport().

        Args:
            name: Unique name for this node within the agent.
            config: Configuration for the delegation node.
            contract: Input/output contract for the node.
            input_mappers: Optional dict of mapper functions.
                DelegationTargets reference these by name via input_mapping.
                Keys are mapper names, values are functions that transform
                input dicts.

        Returns:
            Self for method chaining.

        Example:
            builder.with_delegation_transport(my_transport).add_delegation_node(
                name="delegate_analysis",
                config=DelegationNodeConfig(
                    name="delegate_analysis",
                    targets=[
                        DelegationTarget(
                            agent_name="analyzer",
                            input_mapping="transform_for_analyzer",
                        ),
                    ],
                ),
                contract=analysis_contract,
                input_mappers={
                    "transform_for_analyzer": lambda d: {"text": d["content"]},
                },
            )
        """
        self._delegation_node_configs[name] = config
        self._delegation_node_contracts[name] = contract
        if input_mappers:
            self._delegation_node_mappers[name] = input_mappers
        self._all_node_names.add(name)
        return self

    def add_aggregator_node(
        self,
        name: str,
        config: AggregatorNodeConfig,
        contract: NodeContract,
        retry_handler: Callable[[list[str]], Awaitable[list[str]]] | None = None,
    ) -> AgentBuilder:
        """Register an AggregatorNode in the agent.

        AggregatorNode collects responses from previously dispatched
        delegations using configurable resolution policies (all, any, majority).
        It requires a DelegationTransport to be configured via
        with_delegation_transport().

        Args:
            name: Unique name for this node within the agent.
            config: Configuration for the aggregator node.
            contract: Input/output contract for the node.
            retry_handler: Async callable for retry_missing policy. Receives
                list of missing correlation IDs, returns new IDs to wait for.
                Required when config.on_timeout='retry_missing'.

        Returns:
            Self for method chaining.
        """
        self._aggregator_node_configs[name] = config
        self._aggregator_node_contracts[name] = contract
        if retry_handler is not None:
            self._aggregator_retry_handlers[name] = retry_handler
        self._all_node_names.add(name)
        return self

    def set_entry_node(self, name: str) -> AgentBuilder:
        """Set the entry point of the graph.

        Args:
            name: Name of the node where execution starts.

        Returns:
            Self for method chaining.
        """
        self._entry_node = name
        return self

    def add_route(
        self,
        from_node: str,
        strategy: RoutingStrategy,
    ) -> AgentBuilder:
        """Register a routing strategy for a node.

        Args:
            from_node: Name of the source node.
            strategy: Routing strategy to determine the next node.

        Returns:
            Self for method chaining.
        """
        self._routes[from_node] = strategy
        return self

    def add_fan_out(
        self,
        from_node: str,
        targets: list[str],
        join_node: str,
        on_error: str = "fail_all",
        timeout_seconds: int = 300,
    ) -> AgentBuilder:
        """Declare a fan-out from a node to multiple parallel branches.

        After ``from_node`` executes, all ``targets`` run concurrently.
        When every branch completes, execution continues at ``join_node``
        which receives the original data plus a ``parallel_results`` dict
        keyed by target node name.

        Args:
            from_node: Node whose output triggers the fan-out.
            targets: Nodes to execute in parallel (min 2).
            join_node: Node that collects the parallel results.
            on_error: "fail_all" (default) cancels remaining branches
                on first failure; "continue_partial" waits for all.
            timeout_seconds: Maximum time for all branches.

        Returns:
            Self for method chaining.
        """
        strategy = FanOutStrategy(
            targets=targets,
            join_node=join_node,
            on_error=on_error,
            timeout_seconds=timeout_seconds,
        )
        self._routes[from_node] = strategy
        self._fan_out_targets.update(targets)
        return self

    def set_terminal(self, name: str) -> AgentBuilder:
        """Mark a node as terminal (end of graph).

        Args:
            name: Name of the terminal node.

        Returns:
            Self for method chaining.
        """
        self._terminals.add(name)
        return self

    def add_edge(self, from_node: str, to_node: str) -> AgentBuilder:
        """Add a static edge between two nodes.

        The source node will always route to the target node,
        regardless of its output. For conditional routing, use add_route().

        Args:
            from_node: Source node name.
            to_node: Target node name.

        Returns:
            Self for method chaining.
        """
        from sofias_sdk_lite.routing.strategies import StaticRoute

        self._routes[from_node] = StaticRoute(to_node)
        return self

    def with_default_route_to_terminal(self) -> AgentBuilder:
        """Auto-route unrouted nodes to the terminal node.

        When enabled, nodes without an explicit route (via add_route or
        add_edge) will automatically route to the terminal node at build
        time. This eliminates boilerplate for graphs where most nodes
        converge to a single terminal.

        Returns:
            Self for method chaining.
        """
        self._default_route_to_terminal = True
        return self

    def with_retry_policy(self, policy: RetryPolicy) -> AgentBuilder:
        """Set the default retry policy for all nodes.

        Args:
            policy: Default retry policy.

        Returns:
            Self for method chaining.
        """
        self._default_retry_policy = policy
        return self

    def with_retry_policy_for_node(
        self,
        name: str,
        policy: RetryPolicy,
    ) -> AgentBuilder:
        """Set a node-specific retry policy.

        Args:
            name: Name of the node.
            policy: Retry policy for this specific node.

        Returns:
            Self for method chaining.
        """
        self._node_retry_policies[name] = policy
        return self

    def with_fallback(self, node: str, fallback_node: str) -> AgentBuilder:
        """Register a fallback node for when a node fails.

        Args:
            node: Name of the node that may fail.
            fallback_node: Name of the node to execute on failure.

        Returns:
            Self for method chaining.
        """
        self._fallbacks[node] = fallback_node
        return self

    def with_circuit_breaker(self, config: CircuitBreakerConfig) -> AgentBuilder:
        """Set the circuit breaker configuration.

        Args:
            config: Circuit breaker configuration.

        Returns:
            Self for method chaining.
        """
        self._circuit_breaker_config = config
        return self

    def with_context(self, ctx: dict[str, Any]) -> AgentBuilder:
        """Set the static context shared with all nodes.

        The context is read-only during execution. It contains
        information that doesn't change (e.g., user metadata,
        conversation history).

        Args:
            ctx: Context dictionary.

        Returns:
            Self for method chaining.
        """
        self._context = ctx
        return self

    def with_response_workflow(self, workflow: ResponseWorkflow) -> AgentBuilder:
        """Set the response workflow for delivering agent output.

        The workflow is invoked after execution completes (success or error)
        to deliver the response through the appropriate channel (chat stream,
        reply queue, webhook, etc.).

        Args:
            workflow: Implementation of the ResponseWorkflow protocol.

        Returns:
            Self for method chaining.
        """
        self._response_workflow = workflow
        return self

    def with_middleware(self, middleware: Any) -> AgentBuilder:
        """Add a middleware to the agent execution pipeline.

        Middleware hooks are called before and after each node execution.
        Multiple middleware can be added — they execute in registration order.

        Args:
            middleware: Object implementing before_node/after_node/on_error.

        Returns:
            Self for method chaining.
        """
        if not hasattr(self, "_middleware"):
            self._middleware: list[Any] = []
        self._middleware.append(middleware)
        return self

    def with_tool_registry(self, registry: ToolRegistry) -> AgentBuilder:
        """Set the tool registry for dynamic tool discovery.

        The ToolRegistry is injected into LLM nodes built from configs,
        enabling them to discover and execute tools from external providers
        in addition to static BaseTool instances.

        Pre-built nodes (registered via node=) are not affected.

        Args:
            registry: ToolRegistry wrapping a ToolProvider implementation.

        Returns:
            Self for method chaining.
        """
        self._tool_registry = registry
        return self

    def with_conversation_state(self, state: Any) -> AgentBuilder:
        """Set the conversation state store for multi-turn flow persistence.

        The ConversationState is injected into all FunctionNodes at build
        time, enabling them to persist and retrieve temporary state across
        multiple agent executions within the same conversation.

        Args:
            state: ConversationState adapter wrapping a
                ConversationStateProvider implementation.

        Returns:
            Self for method chaining.

        Example:
            from sofias_sdk_lite.state import (
                ConversationState,
                InMemoryStateProvider,
            )

            builder.with_conversation_state(
                ConversationState(InMemoryStateProvider())
            )
        """
        self._conversation_state = state
        return self

    def add_planner_node(
        self,
        name: str,
        contract: NodeContract,
        llm: LLMCallable | None = None,
        system_prompt: str | None = None,
        description: str | None = None,
        available_nodes: list[str] | None = None,
    ) -> AgentBuilder:
        """Register a PlannerNode in the agent.

        The PlannerNode uses an LLM to generate an ExecutionPlan (DAG)
        based on the input and the list of available nodes. The available
        nodes are collected automatically from all registered nodes at
        build time, unless explicitly specified.

        Args:
            name: Unique name for this node within the agent.
            contract: Input/output contract for the planner node.
            llm: Optional LLM callable. Falls back to the agent-level LLM.
            system_prompt: Custom system prompt for plan generation.
            description: Human-readable description.
            available_nodes: Optional list of node names the planner can use.
                If None, all non-planner nodes are available.

        Returns:
            Self for method chaining.
        """
        self._planner_node_configs[name] = {
            "contract": contract,
            "llm": llm,
            "system_prompt": system_prompt,
            "description": description,
            "available_nodes": available_nodes,
        }
        self._all_node_names.add(name)
        return self

    def set_plan_only(self, name: str) -> AgentBuilder:
        """Mark a node as plan-only.

        Plan-only nodes exist solely for PlanExecutor to invoke.
        They do not participate in the main graph routing and are
        excluded from the validation rule 'every non-terminal node
        must have a route'.

        Args:
            name: Name of the node to mark as plan-only.

        Returns:
            Self for method chaining.
        """
        self._plan_only_nodes.add(name)
        return self

    def with_plan_failure_policy(
        self,
        default_policy: Literal["fail_plan", "skip_dependents", "continue_partial"] = "fail_plan",
        overrides: dict[str, str] | None = None,
        timeout_seconds: int = 600,
    ) -> AgentBuilder:
        """Configure failure handling for plan execution.

        Args:
            default_policy: Default policy when a plan step fails.
                One of "fail_plan", "skip_dependents", "continue_partial".
            overrides: Per-node policy overrides (node_name -> policy).
            timeout_seconds: Global timeout for plan execution.

        Returns:
            Self for method chaining.
        """
        self._plan_failure_policy = default_policy
        if overrides:
            self._plan_failure_overrides = overrides
        self._plan_timeout_seconds = timeout_seconds
        return self

    def _apply_default_routes(self) -> None:
        """Apply default routes to terminal for unrouted nodes.

        Called before validation so that auto-routed nodes pass
        the "all non-terminal nodes must have routes" check.
        """
        if self._default_route_to_terminal and self._terminals:
            from sofias_sdk_lite.routing.strategies import StaticRoute

            terminal = next(iter(self._terminals))
            for node_name in self._all_node_names:
                if (
                    node_name not in self._terminals
                    and node_name not in self._routes
                    and node_name not in self._plan_only_nodes
                ):
                    self._routes[node_name] = StaticRoute(terminal)

    def _validate(self) -> list[str]:
        """Validate the builder configuration.

        Returns:
            List of validation error messages. Empty means valid.
        """
        errors: list[str] = []

        # Required fields
        if self._settings_class is None:
            errors.append("Settings class not defined. Use with_settings_class().")

        if self._input_schema is None or self._output_schema is None:
            errors.append("Agent contract not defined. Use with_contract().")

        if self._entry_node is None:
            errors.append("Entry node not defined. Use set_entry_node().")

        # Entry node must exist
        if self._entry_node is not None and self._entry_node not in self._all_node_names:
            errors.append(
                f"Entry node '{self._entry_node}' not found in registered nodes."
            )

        # At least one node must exist
        if not self._all_node_names:
            errors.append("No nodes registered. Use add_llm_node() or add_function_node() to add at least one node.")

        # At least one terminal must exist
        if not self._terminals:
            errors.append("No terminal nodes defined. Use set_terminal() to mark end nodes.")

        # Terminals must exist as nodes
        for terminal in self._terminals:
            if terminal not in self._all_node_names:
                errors.append(
                    f"Terminal '{terminal}' is not a registered node."
                )

        # Non-terminal nodes must have routes (excluding plan-only and fan-out target nodes)
        # unless default_route_to_terminal is enabled
        for node_name in self._all_node_names:
            if node_name in self._plan_only_nodes:
                continue
            if node_name in self._fan_out_targets:
                continue
            if node_name not in self._terminals and node_name not in self._routes:
                if not self._default_route_to_terminal:
                    errors.append(
                        f"Node '{node_name}' has no route and is not terminal. "
                        "Add a route with add_route()/add_edge(), mark as terminal "
                        "with set_terminal(), or enable with_default_route_to_terminal()."
                    )

        # Route sources must be registered nodes
        for from_node in self._routes:
            if from_node not in self._all_node_names:
                errors.append(
                    f"Route source '{from_node}' is not a registered node."
                )

        # Fallback nodes must exist
        for node, fallback in self._fallbacks.items():
            if fallback not in self._all_node_names:
                errors.append(
                    f"Fallback node '{fallback}' for '{node}' is not a registered node."
                )

        # Fan-out target validations
        for node_name in self._fan_out_targets:
            if node_name not in self._all_node_names:
                errors.append(
                    f"Fan-out target '{node_name}' is not a registered node."
                )
            if node_name in self._routes:
                errors.append(
                    f"Fan-out target '{node_name}' should not have its own route."
                )

        # Build router and validate graph structure (excluding plan-only nodes)
        routing_node_names = self._all_node_names - self._plan_only_nodes
        if routing_node_names and self._routes:
            temp_router = Router()
            for from_node, strategy in self._routes.items():
                try:
                    temp_router.add_route(from_node, strategy)
                except ValueError as e:
                    errors.append(str(e))
            for terminal in self._terminals:
                try:
                    temp_router.add_terminal(terminal)
                except ValueError as e:
                    errors.append(str(e))

            # Use router's built-in validation (only routing-participating nodes)
            graph_errors = temp_router.validate_graph(
                list(routing_node_names),
                fan_out_targets=self._fan_out_targets,
            )
            errors.extend(graph_errors)

        # LLM must be resolvable if there are LLM nodes to be built (not pre-built)
        if self._llm_node_configs and self._llm is None:
            errors.append(self._MISSING_LLM_MESSAGE)

        # Delegation transport must be provided if there are delegation nodes
        if self._delegation_node_configs and self._delegation_transport is None:
            errors.append(
                "DelegationTransport not provided but DelegationNodes exist. "
                "Use with_delegation_transport() to set the transport."
            )

        # Delegation transport must be provided if there are aggregator nodes
        if self._aggregator_node_configs and self._delegation_transport is None:
            errors.append(
                "DelegationTransport not provided but AggregatorNodes exist. "
                "Use with_delegation_transport() to set the transport."
            )

        # retry_handler required when on_timeout='retry_missing'
        for node_name, config in self._aggregator_node_configs.items():
            if config.on_timeout == "retry_missing" and node_name not in self._aggregator_retry_handlers:
                errors.append(
                    f"AggregatorNode '{node_name}' has on_timeout='retry_missing' but no "
                    "retry_handler provided. Pass retry_handler to add_aggregator_node()."
                )

        # Validate that all input_mapping references have corresponding mappers
        for node_name, config in self._delegation_node_configs.items():
            mappers = self._delegation_node_mappers.get(node_name, {})
            for target in config.targets:
                if target.input_mapping and target.input_mapping not in mappers:
                    errors.append(
                        f"DelegationNode '{node_name}' target '{target.agent_name}' "
                        f"references mapper '{target.input_mapping}' which is not registered."
                    )

        # Planner node validations
        if self._planner_node_configs:
            # LLM must be available for planner nodes
            for name, config in self._planner_node_configs.items():
                if config["llm"] is None and self._llm is None:
                    errors.append(
                        f"PlannerNode '{name}' has no LLM and no agent-level LLM could be "
                        f"resolved. {self._MISSING_LLM_MESSAGE}"
                    )

                # Validate available_nodes references
                if config["available_nodes"] is not None:
                    for avail_name in config["available_nodes"]:
                        if avail_name not in self._all_node_names:
                            errors.append(
                                f"PlannerNode '{name}' references unknown available node '{avail_name}'."
                            )

        # Plan-only node validations
        for node_name in self._plan_only_nodes:
            if node_name not in self._all_node_names:
                errors.append(
                    f"Plan-only node '{node_name}' is not a registered node."
                )
            if node_name == self._entry_node:
                errors.append(
                    f"Plan-only node '{node_name}' cannot be the entry node."
                )
            if node_name in self._routes:
                errors.append(
                    f"Plan-only node '{node_name}' should not have routes."
                )

        return errors

    @staticmethod
    def _node_retry_config_to_policy(
        node_retry: Any,
        base: RetryPolicy,
    ) -> RetryPolicy:
        """Convert a NodeRetryConfig to a RetryPolicy.

        Fields present in *node_retry* override the corresponding fields in
        *base*; ``None`` fields fall back to the base policy values.
        """
        max_retries = node_retry.max_retries if node_retry.max_retries is not None else base.max_retries
        delay = node_retry.delay_seconds if node_retry.delay_seconds is not None else base.delay
        max_delay = node_retry.max_delay_seconds if node_retry.max_delay_seconds is not None else base.max_delay

        if node_retry.backoff_multiplier is not None:
            if node_retry.backoff_multiplier <= 1.0:
                backoff_strategy = BackoffStrategy.FIXED
            else:
                backoff_strategy = BackoffStrategy.EXPONENTIAL
        else:
            backoff_strategy = base.backoff_strategy

        return RetryPolicy(
            max_retries=max_retries,
            delay=delay,
            max_delay=max_delay,
            backoff_strategy=backoff_strategy,
            retryable_exceptions=base.retryable_exceptions,
        )

    def build(self) -> Agent:
        """Build the Agent instance.

        Validates all configuration and constructs the Agent.
        If validation fails, raises AgentBuildError with ALL errors
        found (not just the first one).

        Returns:
            Configured Agent instance ready for execution.

        Raises:
            AgentBuildError: If validation fails (contains all errors).
        """
        # Apply default routes before validation
        self._apply_default_routes()

        # Resolve the agent-level LLM when none was given explicitly: the
        # runner's per-task default (from settings) first, then the environment.
        if self._llm is None and self._needs_agent_llm():
            self._llm = resolve_default_llm()

        # Validate configuration
        errors = self._validate()
        if errors:
            raise AgentBuildError(errors)

        # Create SDK config if not provided
        sdk_config = self._sdk_config or SDKConfig()

        # Create agent contract
        agent_contract = AgentContract(
            input_schema=self._input_schema,  # type: ignore[arg-type]
            output_schema=self._output_schema,  # type: ignore[arg-type]
        )

        # Create settings resolver
        settings_resolver = SettingsResolver(self._settings_class)  # type: ignore[arg-type]

        # Create router
        router = Router()
        for from_node, strategy in self._routes.items():
            router.add_route(from_node, strategy)
        for terminal in self._terminals:
            router.add_terminal(terminal)

        # Convert NodeRetryConfig from node configs into RetryPolicy overrides.
        # Explicit with_retry_policy_for_node() calls take precedence.
        global_retry = self._default_retry_policy or RetryPolicy()
        node_retry_overrides = dict(self._node_retry_policies)

        for name, config in self._llm_node_configs.items():
            if name not in node_retry_overrides and config.retry is not None:
                node_retry_overrides[name] = self._node_retry_config_to_policy(
                    config.retry, global_retry,
                )

        for name, config in self._delegation_node_configs.items():
            if name not in node_retry_overrides and config.retry is not None:
                node_retry_overrides[name] = self._node_retry_config_to_policy(
                    config.retry, global_retry,
                )

        # Create error handler config
        error_handler_config = ErrorHandlerConfig(
            retry=global_retry,
            circuit_breaker=self._circuit_breaker_config or CircuitBreakerConfig(),
            fallback=FallbackConfig(fallbacks=self._fallbacks),
            node_retry_overrides=node_retry_overrides,
        )

        # Create error handler
        error_handler = ErrorHandler(config=error_handler_config)

        # Build all nodes
        nodes: dict[str, BaseNode] = {}

        # Build LLM nodes from configs
        for name, config in self._llm_node_configs.items():
            contract = self._llm_node_contracts[name]
            tools = self._llm_node_tools.get(name)

            node = LLMNode(
                config=config,
                contract=contract,
                llm=self._llm,  # type: ignore[arg-type]
                tools=tools,
                system_prompt=self._system_prompt,
                settings_resolver=settings_resolver,
                tool_registry=self._tool_registry,
            )
            nodes[name] = node

        # Add pre-built LLM nodes
        for name, llm_node in self._llm_nodes.items():
            nodes[name] = llm_node

        # Add function nodes (inject conversation state if configured)
        for name, function_node in self._function_nodes.items():
            if self._conversation_state is not None:
                function_node._conversation_state = self._conversation_state
            nodes[name] = function_node

        # Build delegation nodes
        if self._delegation_node_configs:
            # Import here to avoid circular imports
            from sofias_sdk_lite.nodes.delegation_node import DelegationNode

            for name, config in self._delegation_node_configs.items():
                contract = self._delegation_node_contracts[name]
                mappers = self._delegation_node_mappers.get(name)

                node = DelegationNode(
                    config=config,
                    contract=contract,
                    transport=self._delegation_transport,  # type: ignore[arg-type]
                    input_mappers=mappers,
                    source_agent_name=self._name,
                    settings_resolver=settings_resolver,
                )
                nodes[name] = node

        # Build aggregator nodes
        if self._aggregator_node_configs:
            from sofias_sdk_lite.nodes.aggregator_node import AggregatorNode

            for name, config in self._aggregator_node_configs.items():
                contract = self._aggregator_node_contracts[name]
                retry_handler = self._aggregator_retry_handlers.get(name)

                node = AggregatorNode(
                    config=config,
                    contract=contract,
                    transport=self._delegation_transport,  # type: ignore[arg-type]
                    retry_handler=retry_handler,
                )
                nodes[name] = node

        # Build planner nodes
        plan_executor = None
        if self._planner_node_configs:
            from sofias_sdk_lite.nodes.base_node import NodeType
            from sofias_sdk_lite.nodes.planner_node import PlannerNode
            from sofias_sdk_lite.nodes.planning_models import NodeDescriptor
            from sofias_sdk_lite.nodes.plan_executor import PlanExecutor

            # Collect NodeDescriptors from all non-planner nodes
            node_descriptors = []
            for node_name, node_instance in nodes.items():
                if node_name not in self._planner_node_configs:
                    node_descriptors.append(
                        NodeDescriptor(
                            name=node_name,
                            node_type=node_instance.node_type,
                            description=node_instance.description,
                        )
                    )

            # Build each PlannerNode
            for name, config in self._planner_node_configs.items():
                if config["available_nodes"] is not None:
                    available = [
                        d
                        for d in node_descriptors
                        if d.name in config["available_nodes"]
                    ]
                else:
                    available = node_descriptors

                planner = PlannerNode(
                    name=name,
                    contract=config["contract"],
                    llm=config["llm"] or self._llm,  # type: ignore[arg-type]
                    available_nodes=available,
                    system_prompt=config["system_prompt"],
                    description=config["description"],
                )
                nodes[name] = planner

            # Build PlanExecutor
            plan_executor = PlanExecutor(
                nodes=nodes,
                error_handler=error_handler,
                default_failure_policy=self._plan_failure_policy,
                failure_overrides=self._plan_failure_overrides,
                global_timeout_seconds=self._plan_timeout_seconds,
            )

        # Register fallback executors
        # The error handler looks up executors by fallback node name
        for node_name, fallback_name in self._fallbacks.items():
            fallback_node = nodes[fallback_name]

            # Create a closure that properly captures the fallback node
            def make_executor(fb_node: BaseNode):
                async def executor(
                    input_data: dict[str, Any],
                    error_context: Any,
                ) -> dict[str, Any]:
                    return await fb_node.execute(input_data)

                return executor

            # Register with fallback_name (the error handler looks up by fallback node name)
            error_handler.register_fallback_executor(
                fallback_name,
                make_executor(fallback_node),
            )

        # Create agent
        return Agent(
            name=self._name,
            version=self._version,
            description=self._description,
            nodes=nodes,
            router=router,
            error_handler=error_handler,
            contract=agent_contract,
            entry_node=self._entry_node,  # type: ignore[arg-type]
            settings_resolver=settings_resolver,
            context=self._context,
            response_workflow=self._response_workflow,
            plan_executor=plan_executor,
            middleware=self._middleware if self._middleware else None,
        )

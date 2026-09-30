# Agents and graphs

An `Agent` is a pure orchestrator: it validates input, walks a graph of
nodes according to explicit routing, validates output, and returns an
`AgentResponse`. Business logic lives in nodes, never in the agent itself.

```
AgentMessage → validate InputContract → resolve AgentSettings
    → router-guided node traversal → error handling (retry/fallback/circuit breaker)
    → validate OutputContract → AgentResponse
```

## Node types

| Type | Class | Purpose |
|------|-------|---------|
| LLM | `LLMNode` | Invokes an `LLMCallable` with tool loops, prompt assembly, streaming |
| Function | `FunctionNode` | Deterministic Python logic, no LLM |
| Delegation | `DelegationNode` | Delegates to another agent via a `DelegationTransport` |
| Aggregator | `AggregatorNode` | Fan-in: collects delegation responses (all/any/majority policies) |
| Planner | `PlannerNode` | LLM generates a DAG execution plan; `PlanExecutor` runs it |

All nodes subclass `BaseNode` and implement `async _run()`. `BaseNode.execute()`
validates input/output against the node's `NodeContract` around your logic.

## Routing

Routing decides which node runs next, based on the previous node's output.
Built-in strategies (`sofias_sdk_lite.routing`):

- `StaticRoute` — always the same target (used by `add_edge`)
- `FieldValueStrategy` — route on a field's value
- `FieldPresenceStrategy` — route on whether a field is present/truthy
- `ConditionalStrategy` — a custom callable decides
- `ConfidenceThresholdStrategy` — route on a numeric threshold
- `CompositeStrategy` — try several strategies in order, first match wins
- `FanOutStrategy` — dispatch to multiple nodes in parallel, join at an
  `AggregatorNode` or merge point

```python
from sofias_sdk_lite import FieldValueStrategy

builder.add_route(
    "classifier",
    FieldValueStrategy(field="category", mapping={"billing": "billing_agent", "other": "fallback"}),
)
```

## Contracts

Every node and the agent itself has an `InputContract`/`OutputContract` pair
(a `NodeContract`/`AgentContract`). Contracts extend `StrictContract`
(`extra="ignore"` — unknown fields are silently dropped, not rejected;
`strict=True` — no implicit type coercion).

## Building

`AgentBuilder` is a fluent builder. Nothing runs until `.build()`, which
validates the whole graph up front — a missing entry node, an unrouted node,
or a mismatched contract raises `AgentBuildError` listing every problem, not
just the first one.

## Error handling

`ErrorHandler` (in `sofias_sdk_lite.errors`) wraps every node execution with
retry (configurable backoff), a circuit breaker (per-node, in-memory by
default), and fallback routing. See [Errors and retries](errors-and-retries.md).

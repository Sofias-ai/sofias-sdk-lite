# Errors and retries

Every node execution is wrapped by `ErrorHandler`, configured via
`AgentBuilder.with_retry_policy()`, `.with_retry_policy_for_node()`,
`.with_circuit_breaker()`, and `.with_fallback()`.

## Retry

```python
from sofias_sdk_lite import RetryPolicy
from sofias_sdk_lite.errors import BackoffStrategy

builder.with_retry_policy(
    RetryPolicy(
        max_retries=3,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,  # or FIXED, LINEAR
        delay=1.0,
        max_delay=30.0,
    )
)
```

Only `NodeExecutionError`, `ToolExecutionError`, and `OutputValidationError`
are retried by default (`RetryPolicy.retryable_exceptions`) —
`OutputValidationError` is included because some LLM providers return
error payloads inside a 200 response; retrying gives the model another
chance at a conforming response.

Override per node with `with_retry_policy_for_node("node_name", policy)`.

## Circuit breaker

```python
from sofias_sdk_lite import CircuitBreakerConfig

builder.with_circuit_breaker(
    CircuitBreakerConfig(failure_threshold=5, recovery_timeout=30.0)
)
```

After `failure_threshold` consecutive failures the circuit opens and routes
straight to the node's fallback (skipping retries) until `recovery_timeout`
elapses, at which point one request is let through to test recovery
(half-open).

The default `InMemoryCircuitBreakerStore` is per-process — if you run
multiple instances of the same agent consuming from the same queue, each
instance trips independently. Implement `CircuitBreakerStore` against Redis
or another shared store for multi-instance deployments; nothing else in the
SDK needs to change.

## Fallback

```python
builder.with_fallback("primary_node", "backup_node")
```

When `primary_node` exhausts its retries (or its circuit is open), execution
routes to `backup_node` instead of raising. The fallback receives the
original input plus an `ExecutionErrorContext` describing what failed.

## What still raises

If a node fails and has no fallback configured, `ErrorHandler` raises
`GraphExecutionError` — this propagates out of `Agent.execute()`. Contract
validation failures (`InputValidationError`/`OutputValidationError`) are not
swallowed by the retry loop after retries are exhausted; they surface as the
underlying cause.

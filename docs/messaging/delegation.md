# Delegation

Agents can delegate work to other agents over RabbitMQ. A `DelegationNode`
sends a `DelegationRequest` to a target queue and waits for a
`DelegationResponse`, correlated by ID.

## Delegating from an agent

```python
from sofias_sdk_lite import RabbitMQDelegationTransport, RabbitMQClient, RabbitMQConfig
from sofias_sdk_lite.nodes import DelegationNodeConfig

async with RabbitMQClient(RabbitMQConfig(host="localhost")) as client:
    transport = RabbitMQDelegationTransport(client)
    await transport.start()

    builder.with_delegation_transport(transport)
    builder.add_delegation_node(
        "translate",
        DelegationNodeConfig(target_queue="translator-agent-tasks", timeout_seconds=30),
        contract,
    )
```

`DelegationTransport` is a protocol (`sofias_sdk_lite.nodes.delegation_transport`)
— `RabbitMQDelegationTransport` is the RabbitMQ implementation,
`NullDelegationTransport` is an in-memory stand-in for tests.

## Responding to a delegation request

On the receiving side, `task.reply_to` and `task.correlation_id` will be
set. Use `BaseDelegationResponseWorkflow` instead of `ChatResponseWorkflow`
so the reply goes back over the delegation transport, not a chat stream:

```python
from sofias_sdk_lite import BaseDelegationResponseWorkflow

def create_response_workflow(self, client, task):
    if task.reply_to:
        async def publish(payload: dict) -> None:
            await client.publish_reply(task.reply_to, payload, correlation_id=task.correlation_id)
        return BaseDelegationResponseWorkflow(publish_fn=publish)
    return super().create_response_workflow(client, task)
```

Override `AgentRunner.create_response_workflow()` with logic like the above
in your runner subclass.

## Fan-out and aggregation

To delegate to several agents in parallel and merge results, use
`FanOutStrategy` to route to multiple delegation nodes, and an
`AggregatorNode` with a resolution policy (`all`, `any`, `majority`) to
collect responses:

```python
from sofias_sdk_lite import FanOutStrategy
from sofias_sdk_lite.nodes import AggregatorNodeConfig

builder.add_route("dispatch", FanOutStrategy(targets=["translate", "summarize"], join_node="merge"))
builder.add_aggregator_node("merge", AggregatorNodeConfig(resolution_policy="all", timeout_seconds=60), contract)
```

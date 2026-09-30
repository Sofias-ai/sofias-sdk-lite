# Messaging overview

The topology, end to end:

```
platform/caller --publish--> [task queue] --consume--> AgentRunner
                                                            |
                                                     Agent.execute()
                                                            |
                                                    ChatResponseWorkflow
                                                            |
                                                     [response stream] --> caller
```

## Wire models (`sofias_sdk_lite.messaging`)

| Model | Purpose |
|---|---|
| `AgentTaskMessage` | What a conversational agent receives from its queue |
| `ApiTaskMessage` | Same shape, for API-triggered agents (no conversation) |
| `StreamFragment` | A single chunk of a streamed response |
| `DelegationRequest` / `DelegationResponse` | Agent-to-agent delegation payload |
| `ChatRequest` / `ChatResponse` | Simple request/response, non-streaming |

Field names and types are the interop contract with whatever publishes to
your queue — see `tests/messaging/test_wire_contract.py` for the pinned
shape. Don't rename fields without a coordinated wire-format change on the
publisher side too.

## Transport (`sofias_sdk_lite.rabbitmq`)

| Class | Purpose |
|---|---|
| `RabbitMQConfig` | Connection settings (host, ports, credentials, vhost) |
| `RabbitMQClient` | Connection/channel lifecycle, used as an async context manager |
| `RabbitMQConsumer` | Declares a durable queue, dispatches messages to a handler |
| `RabbitMQPublisher` | Publishes to an exchange or a RabbitMQ Stream |
| `RabbitMQRPCClient` | Request/reply over a temporary exclusive queue |
| `RabbitMQDelegationTransport` | Implements `DelegationTransport` for agent-to-agent calls |

No queue-naming or stream-naming convention is baked in anywhere — every
name is an explicit constructor argument (see `RunnerConfig`).

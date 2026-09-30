# sofias-sdk-lite

Build multi-node agents on a declarative graph and run them against RabbitMQ.

## What you get

- **An agent graph** — LLM nodes, function nodes, delegation nodes (agent-to-agent),
  aggregator nodes (fan-in), and planner nodes (dynamic DAGs), connected with
  explicit routing strategies.
- **A messaging layer** — pydantic wire models (`AgentTaskMessage`,
  `StreamFragment`, delegation contracts) and a RabbitMQ transport built on
  `aio-pika` and RabbitMQ Streams.
- **A runner** — `AgentRunner` consumes tasks from a queue, resolves
  per-request settings, executes your agent, and streams the response back —
  with graceful shutdown, retries, and circuit breakers built in.
- **An LLM that is already wired** — LLM nodes talk to the Sofias gateway
  (or any OpenAI-compatible endpoint) using the model, URL and key
  from the agent's settings or the `SOFIAS_LLM_*` environment. `create_llm()`
  builds the client when you need it explicitly; `LLMCallable` stays a protocol
  for anything the bundled clients do not cover — see
  [LLM integration](concepts/llm-integration.md).

Nothing in this package depends on a private backend or an internal config
service. Where the model lives is deployment configuration, not agent code.

## Where to go next

- [Quickstart](quickstart.md) — build and execute your first agent in a few
  minutes, no broker required.
- [Concepts → Agents and graphs](concepts/agents-and-graphs.md) — nodes,
  routing, contracts.
- [Messaging → Running an agent](messaging/running-an-agent.md) — wire the
  same agent up to RabbitMQ with `AgentRunner`.
- [Reference](reference.md) — full API reference.

## Install

```bash
pip install sofias-sdk-lite
```

Requires Python 3.11+. A running RabbitMQ broker (with the
[Streams plugin](https://www.rabbitmq.com/docs/streams)) is only needed once
you use `AgentRunner` or streaming responses — the agent graph itself has no
broker dependency and can be built and executed directly, as in the
quickstart.

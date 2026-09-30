# sofias-sdk-lite

Build multi-node agents on a declarative graph and run them against RabbitMQ.

**Documentation:** [docs.sofias.ai](https://docs.sofias.ai/) · **Developer portal:** [developers.sofias.ai](https://developers.sofias.ai/) · **Source:** [github.com/Sofias-ai/sofias-sdk-lite](https://github.com/Sofias-ai/sofias-sdk-lite)

`sofias-sdk-lite` gives you:

- **An agent graph** — LLM nodes, function nodes, delegation nodes (agent-to-agent),
  aggregator nodes (fan-in), and planner nodes (dynamic DAGs), wired together with
  explicit routing strategies.
- **A messaging layer** — pydantic wire models (`AgentTaskMessage`, `StreamFragment`,
  delegation contracts) and a RabbitMQ transport (client, consumer, publisher, RPC,
  delegation transport) built on `aio-pika` and RabbitMQ Streams.
- **A runner** — `AgentRunner` consumes tasks from a queue, resolves per-request
  settings, executes your agent, and streams the response back — with graceful
  shutdown, retries, and circuit breakers built in.
- **An LLM that is already wired** — LLM nodes talk to the Sofias gateway
  (or any OpenAI-compatible endpoint) using the model, URL and key
  from the agent's settings or the `SOFIAS_LLM_*` environment. `create_llm()`
  builds the client when you need it explicitly; `LLMCallable` stays a protocol
  for anything the bundled clients do not cover.

Nothing here depends on a private backend or an internal config service.
Where the model lives is deployment configuration, not agent code.

## Install

```bash
pip install sofias-sdk-lite
# or
uv add sofias-sdk-lite
```

Requires Python 3.11+ and a running RabbitMQ broker (with the
[Streams plugin](https://www.rabbitmq.com/docs/streams) enabled) if you use
the runner or streaming responses.

## Quickstart

A minimal agent with a single function node:

```python
import asyncio

from sofias_sdk_lite import (
    AgentBuilder,
    AgentMessage,
    BaseAgentSettings,
    InputContract,
    NodeContract,
    OutputContract,
)


class GreetInput(InputContract):
    name: str


class GreetOutput(OutputContract):
    greeting: str


class HelloSettings(BaseAgentSettings):
    pass


def greet(data: dict, context: dict | None = None) -> dict:
    return {"greeting": f"Hello, {data['name']}!"}


async def main() -> None:
    contract = NodeContract(input_schema=GreetInput, output_schema=GreetOutput)
    agent = (
        AgentBuilder("hello_agent", version="0.1.0")
        .with_settings_class(HelloSettings)
        .with_contract(input_schema=GreetInput, output_schema=GreetOutput)
        .add_function_node("greeter", contract, process_fn=greet)
        .set_entry_node("greeter")
        .set_terminal("greeter")
        .build()
    )

    response = await agent.execute(AgentMessage(content=GreetInput(name="World")))
    print(response.content)  # {"greeting": "Hello, World!"}


asyncio.run(main())
```

See [`examples/`](examples/) for a runner against a local RabbitMQ
(`docker-compose.yml` included), tool loops, streaming, and agent-to-agent
delegation.

## Connecting to the LLM

An agent declares *what* the model should do; *where* the model lives is
resolved at `build()`:

1. Under `AgentRunner`, from the per-task settings the platform sends
   (`model_name`, `router_url`, `router_api_key` on `BaseAgentSettings`).
2. Otherwise from the environment: `SOFIAS_LLM_MODEL`, `SOFIAS_LLM_BASE_URL`,
   `SOFIAS_LLM_API_KEY` and `SOFIAS_LLM_PROVIDER` (default `sofias`). The
   Sofias platform injects them into every agent it runs, so an agent
   deployed through the developer portal needs no LLM configuration at all.
   There is no default URL: to run locally against a real gateway, set it.
3. Or explicitly: `.with_llm(create_llm(model="default", api_key=...))`.

```python
from sofias_sdk_lite import AgentBuilder, LLMNodeConfig, NodeContract

agent = (
    AgentBuilder("qa_agent")
    .with_settings_class(MySettings)
    .with_contract(input_schema=Question, output_schema=Answer)
    .add_llm_node(
        "answerer",
        LLMNodeConfig(name="answerer", input_contract=Question, output_contract=Answer,
                      system_prompt="Answer in one sentence."),
        NodeContract(input_schema=Question, output_schema=Answer),
    )
    .set_entry_node("answerer")
    .set_terminal("answerer")
    .build()          # no LLM client anywhere in this file
)
```

See `examples/03_llm_agent.py` and the
[LLM integration](docs/concepts/llm-integration.md) guide (streaming,
retries, bringing your own client).

## Running an agent against RabbitMQ

```python
from sofias_sdk_lite import AgentRunner, RunnerConfig, RabbitMQConfig, AgentMessage


class MyAgentRunner(AgentRunner):
    settings_class = MySettings

    def build_agent(self, settings, workflow):
        return (
            AgentBuilder("my_agent")
            .with_settings_class(type(settings))
            .with_response_workflow(workflow)
            # ... nodes, routing ...
            .build()
        )

    def prepare_input(self, task, history, role):
        return AgentMessage(
            content=MyInput(message=task.content, role=role),
            conversation_id=task.conversation_id,
        )


if __name__ == "__main__":
    MyAgentRunner(
        RunnerConfig(
            queue="my-agent-tasks",
            agent_name="my_agent",
            rabbitmq=RabbitMQConfig(host="localhost"),
        )
    ).run()
```

## Optional extras

- `sofias-sdk-lite[otel]` — installs `opentelemetry-api` so the runner opens
  an `agent.handle` span per turn and response fragments carry the active
  span's `traceparent`. Without it, the inbound `traceparent` is still echoed
  end to end.

## Scope (v1)

Core agent graph, RabbitMQ messaging, the runner, and LLM clients for the
Sofias gateway / any OpenAI-compatible endpoint ship today. Memory and MCP
tool discovery are intentionally out of scope for v1 — the SDK ships the
relevant protocols (`MemoryProvider`, `ToolProvider`) so you can plug in
your own, and these become optional extras in a later release.

## Source and contributions

The public repository holds read-only release snapshots: each release of
`sofias-sdk-lite` appears there as a single commit. Development happens
elsewhere, so pull requests cannot be merged and we do not accept
contributions at the moment. For questions or to report a problem, write to
support@sofias.ai.

## License

Source-available under the [Sofias SDK Lite License](https://github.com/Sofias-ai/sofias-sdk-lite/blob/main/LICENSE):
you may use `sofias-sdk-lite` to develop, test and run agents for the Sofias
platform. The agents you build with it are yours.

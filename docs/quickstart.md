# Quickstart

This builds and executes an agent entirely in-process — no RabbitMQ needed.
Once you're comfortable with the graph, see
[Running an agent](messaging/running-an-agent.md) to wire it up to a queue.

## 1. Define your contracts

Contracts are pydantic models with strict validation. They describe what
your agent accepts and returns.

```python
from sofias_sdk_lite import InputContract, OutputContract


class GreetInput(InputContract):
    name: str


class GreetOutput(OutputContract):
    greeting: str
```

## 2. Define your settings

Every agent needs a settings class — subclass `BaseAgentSettings` and add
whatever fields your agent needs (model name, API key, feature flags, ...).

```python
from sofias_sdk_lite import BaseAgentSettings


class HelloSettings(BaseAgentSettings):
    pass  # no extra fields needed for this example
```

## 3. Write a node

A `FunctionNode` runs plain Python — no LLM involved. (See
[Agents and graphs](concepts/agents-and-graphs.md) for `LLMNode`,
`DelegationNode`, `AggregatorNode`, and `PlannerNode`.)

```python
def greet(data: dict, context: dict | None = None) -> dict:
    return {"greeting": f"Hello, {data['name']}!"}
```

## 4. Build the agent

```python
from sofias_sdk_lite import AgentBuilder, NodeContract

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
```

`.build()` validates the whole graph (entry node exists, every node is
routed to a terminal, contracts line up) and raises `AgentBuildError` with a
list of every problem found, rather than failing on the first one.

## 5. Execute it

```python
import asyncio
from sofias_sdk_lite import AgentMessage


async def main():
    response = await agent.execute(AgentMessage(content=GreetInput(name="World")))
    print(response.status)      # ResponseStatus.SUCCESS
    print(response.content)     # {"greeting": "Hello, World!"}
    print(response.execution_path)  # ["greeter"]


asyncio.run(main())
```

That's the whole loop: validate input → run the graph → validate output →
return an `AgentResponse`.

## Next

- Add an `LLMNode` — see [LLM integration](concepts/llm-integration.md).
- Add retries and fallbacks — see [Errors and retries](concepts/errors-and-retries.md).
- Consume tasks from RabbitMQ — see [Running an agent](messaging/running-an-agent.md).

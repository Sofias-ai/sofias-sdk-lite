"""Minimal agent: build, execute, no broker required.

Run: python examples/01_hello_agent.py
"""

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
    print("status:", response.status)
    print("content:", response.content)
    print("execution_path:", response.execution_path)


if __name__ == "__main__":
    asyncio.run(main())

# Testing your agent

The graph runs entirely in-process — testing an agent doesn't require a
broker or a real LLM.

## Testing the graph without a broker

```python
import pytest
from sofias_sdk_lite import AgentMessage

@pytest.mark.asyncio
async def test_my_agent_greets():
    agent = build_my_agent()  # your AgentBuilder(...).build()
    response = await agent.execute(AgentMessage(content=MyInput(name="World")))
    assert response.status.value == "success"
    assert response.content == {"greeting": "Hello, World!"}
```

## Faking the LLM

Implement `LLMCallable` with canned responses instead of calling a real
provider:

```python
from sofias_sdk_lite.llm import LLMCallable, LLMResponse

class FakeLLM:
    def __init__(self, responses: list[LLMResponse]):
        self._responses = iter(responses)

    async def invoke(self, prompt, tools=None, messages=None, system_prompt=None, response_format=None):
        return next(self._responses)

builder.with_llm(FakeLLM([LLMResponse(content="canned answer")]))
```

When the builder lives inside an `AgentRunner.build_agent()` you do not want
to edit, install the fake as the default LLM around the build instead, or
override the runner's `create_llm()` hook:

```python
from sofias_sdk_lite import default_llm

with default_llm(FakeLLM([LLMResponse(content="canned answer")])):
    agent = runner.build_agent(settings, workflow)


class TestRunner(MyRunner):
    def create_llm(self, settings):
        return FakeLLM([LLMResponse(content="canned answer")])
```

## Testing without RabbitMQ

Use `NullWorkflow` in place of `ChatResponseWorkflow` — it records what was
sent instead of publishing anywhere:

```python
from sofias_sdk_lite import NullWorkflow

workflow = NullWorkflow()
builder.with_response_workflow(workflow)
agent = builder.build()

await agent.execute(message)
assert len(workflow.responses) == 1
```

For delegation, use `NullDelegationTransport` the same way — it stores
requests and lets you inject simulated responses without a broker.

## Testing an `AgentRunner`

The runner's `_handle_message()` is exercised indirectly by feeding it a
`MessageContext` built from a plain `AgentTaskMessage` and a fake
`RabbitMQClient`/consumer — or, more simply, unit test `build_agent()` and
`prepare_input()` directly (they're plain methods, not tied to RabbitMQ) and
rely on integration tests against a local broker (see
`examples/docker-compose.yml`) for the wiring itself.

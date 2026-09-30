# LLM integration

You do not construct an LLM client to build an agent. Declare LLM nodes,
run the agent, and the SDK wires the model in from wherever the deployment
says it lives. The pieces, from "never think about it" to "full control":

| Layer | What it is | When you touch it |
|---|---|---|
| **Settings / environment** | `model_name`, `router_url`, `router_api_key` on `BaseAgentSettings`, or `SOFIAS_LLM_*` env vars | Deploying |
| `create_llm()` | Factory that turns settings/env/kwargs into a client | Scripts, tests, custom runners |
| `GatewayLLM`, `OpenAICompatibleLLM` | Bundled clients for the Sofias gateway and any OpenAI-style endpoint | Tuning (`temperature`, `reasoning_effort`, ...) |
| `LLMCallable` protocol | The contract `LLMNode` drives | Bringing a client the bundled ones cannot cover |

## Zero-config: settings and environment

### Under `AgentRunner` (production)

The platform sends the gateway keys with every task (`agent_configuration`),
or your `ConfigSource` returns them. They land on `BaseAgentSettings`:

```python
class MySettings(BaseAgentSettings):
    pass  # model_name / router_url / router_api_key are inherited
```

`AgentRunner` reads them, builds one `GatewayLLM` per distinct gateway
configuration (cached across tasks), and installs it as the **default LLM**
for the duration of `build_agent()`. Your agent module never mentions it:

```python
class MyRunner(AgentRunner):
    settings_class = MySettings

    def build_agent(self, settings, workflow):
        return (
            AgentBuilder("my_agent")
            .with_settings_class(MySettings)
            .with_contract(input_schema=Ask, output_schema=Answer)
            .add_llm_node("answer", answer_config, answer_contract)   # no with_llm()
            .set_entry_node("answer")
            .set_terminal("answer")
            .with_response_workflow(workflow)
            .build()
        )
```

Override `AgentRunner.create_llm(settings)` to swap the client (a fake in
tests, per-tenant routing, a provider the bundled adapters do not cover).

### Standalone (scripts, local runs)

When there is no runner, `AgentBuilder.build()` falls back to the
environment:

| Variable | Meaning | Default |
|---|---|---|
| `SOFIAS_LLM_MODEL` | Model id or gateway alias (e.g. `default`) | required |
| `SOFIAS_LLM_BASE_URL` | OpenAI-compatible API root | required to build a client |
| `SOFIAS_LLM_API_KEY` | Bearer token | empty (no header) |
| `SOFIAS_LLM_PROVIDER` | `sofias` or an OpenAI-compatible label (`openai`, `litellm`, ...) | `sofias` |

The Sofias platform injects these variables into every sandbox and
production agent container, so an agent uploaded through the developer
portal needs **no** LLM configuration at all: pick the inference model in the
agent's runtime settings and the SDK finds it.

`MODEL_NAME`, `ROUTER_URL` and `ROUTER_API_KEY` — the same keys the agent
settings use — are accepted as fallbacks, so a container that already exports
them needs no renaming.

There is **no default URL**. Setting a model without a URL raises
`LLMConfigurationError` with `missing=["base_url"]`, which tells you what to
set instead of failing later with a connection error.

A base URL given as a bare host (`https://gateway.example`) gets `/v1`
appended automatically; a URL with a path is used as-is.

```bash
SOFIAS_LLM_MODEL=default SOFIAS_LLM_BASE_URL=http://localhost:8080/v1 SOFIAS_LLM_API_KEY=... \
python examples/03_llm_agent.py
```

If nothing is configured and the graph has LLM nodes, `build()` raises
`AgentBuildError` naming the three ways to fix it.

## `create_llm()`: one call, resolved for you

```python
from sofias_sdk_lite import create_llm

llm = create_llm()                                   # env
llm = create_llm(settings)                           # BaseAgentSettings (or anything with model_name)
llm = create_llm(model="default", api_key="...")     # explicit
llm = create_llm(settings, temperature=0.0, max_tokens=2048, reasoning_effort="medium")
```

Resolution order: explicit `LLMGatewayConfig`/settings → keyword overrides →
environment. Missing model → `LLMConfigurationError` with the fix spelled
out. Returns a `StreamableLLMCallable`, so streaming chat responses work out
of the box.

Pass the result to `AgentBuilder.with_llm()` when you want to be explicit,
or scope it with `default_llm()` so every `build()` inside picks it up:

```python
from sofias_sdk_lite import default_llm

with default_llm(create_llm(model="default")):
    agent = build_my_agent()
```

## The bundled clients

`GatewayLLM` is `OpenAICompatibleLLM` with the Sofias gateway's defaults:
a per-model-family `reasoning_effort` (thinking channel
off or minimal unless you ask for more) and a 16 384-token output budget so
structured output never truncates mid-JSON. Both:

- send `system_prompt`, `prompt`, history, tools, `response_format` and
  `parallel_tool_calls` in the OpenAI shape;
- retry transport errors and HTTP 408/409/429/5xx with backoff, honouring
  `Retry-After`; other statuses raise `LLMRequestError` with status and body;
- raise `EmptyLLMResponseError` on a completion with neither text nor tool
  calls (`LLMNode` retries the call);
- report `usage` (including cached and reasoning tokens) and, for gateways,
  the real upstream `provider` / `model_requested` from `extra_fields`;
- stream via SSE, yielding text deltas as they arrive, tool calls once
  complete, and a final usage event.

They depend on `httpx` only.

## The protocol: bringing your own client

`LLMNode` only needs an object with this shape:

```python
from typing import Protocol
from sofias_sdk_lite.llm import LLMResponse, ToolSpec

class LLMCallable(Protocol):
    async def invoke(
        self,
        prompt: str,
        tools: list[ToolSpec] | None = None,
        messages: list[dict] | None = None,
        system_prompt: str | None = None,
        response_format: dict | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> LLMResponse: ...
```

`StreamableLLMCallable` adds `stream_invoke()` with the same parameters,
written as an async generator yielding `LLMStreamEvent`s — see
[Streaming](#streaming) below.

!!! warning "Implement every parameter, even ones you ignore"
    `LLMNode` passes all of these keyword arguments (including
    `parallel_tool_calls` on the tool-loop path). A method that lacks one
    fails at runtime with `TypeError: got an unexpected keyword argument`,
    not at type-check time, because `LLMCallable` is a structural
    `Protocol`. Copy the full signature, or accept `**kwargs` in addition to
    the parameters you use.

A minimal wrapper over some other HTTP API:

```python
import httpx
from sofias_sdk_lite.llm import LLMResponse, TokenUsage, ToolCall


class MyLLM:
    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}
        )
        self._model = model

    async def invoke(
        self,
        prompt,
        tools=None,
        messages=None,
        system_prompt=None,
        response_format=None,
        parallel_tool_calls=None,
    ):
        payload = {
            "model": self._model,
            "messages": self._build_messages(prompt, messages, system_prompt),
        }
        if tools:
            payload["tools"] = [{"name": t.name, "parameters": t.parameters_schema} for t in tools]
        if response_format is not None:
            payload["response_format"] = response_format
        if parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = parallel_tool_calls

        resp = await self._client.post("/chat/completions", json=payload)
        data = resp.json()
        choice = data["choices"][0]["message"]

        return LLMResponse(
            content=choice.get("content"),
            tool_calls=[
                ToolCall(name=tc["name"], input=tc["arguments"], id=tc.get("id"))
                for tc in choice.get("tool_calls", [])
            ],
            usage=TokenUsage(**data["usage"]) if "usage" in data else None,
            model_requested=self._model,
        )

    def _build_messages(self, prompt, messages, system_prompt):
        chat_messages = list(messages) if messages else []
        if system_prompt and not any(m.get("role") == "system" for m in chat_messages):
            chat_messages.insert(0, {"role": "system", "content": system_prompt})
        chat_messages.append({"role": "user", "content": prompt})
        return chat_messages
```

Pass an instance to `AgentBuilder.with_llm(MyLLM(api_key=..., model=..., base_url=...))`,
or return it from `AgentRunner.create_llm()`.

## Streaming

Implement `StreamableLLMCallable` (same class, add `stream_invoke`) to make
`LLMNode.stream_execute()` — and therefore `AgentRunner`'s streamed chat
responses — emit real incremental deltas instead of one final chunk.
`stream_invoke` takes the exact same parameters as `invoke` and is an async
generator yielding `LLMStreamEvent(delta=...)`:

```python
import asyncio
import json
from sofias_sdk_lite.llm import LLMStreamEvent


class MyLLM:
    # ... __init__, invoke, _build_messages as above ...

    async def stream_invoke(
        self,
        prompt,
        tools=None,
        messages=None,
        system_prompt=None,
        response_format=None,
        parallel_tool_calls=None,
    ):
        payload = {
            "model": self._model,
            "messages": self._build_messages(prompt, messages, system_prompt),
            "stream": True,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {})
                if text := delta.get("content"):
                    yield LLMStreamEvent(delta=text, model_requested=self._model)
```

`LLMNode` checks `isinstance(self._llm, StreamableLLMCallable)` at runtime —
a plain `invoke`-only client still works everywhere, it just falls back to
yielding its single response as one chunk instead of incremental deltas.

## Raw text mode

Set `raw_text_field` on `LLMNodeConfig` (instead of relying on
`output_contract`'s JSON schema) when you want the model's plain-text
response streamed straight through, with no JSON parsing or schema
instructions injected into the prompt — the raw text is wrapped as
`{raw_text_field: content}` before output validation. This is the shape you
want for a conversational final answer; reserve the default JSON-schema mode
for nodes whose output feeds another node in the graph.

## Tools

`sofias_sdk_lite.llm.ToolSpec`/`ToolCall` are the *description* types passed
to `invoke()`. Actually running a tool is a separate concern — see
`sofias_sdk_lite.tools` (`BaseTool`, `ToolRegistry`, `ToolProvider`) for the
execution framework, which `LLMNode`'s tool loop uses.

### Tool-loop budget and working memory

Each iteration of the loop re-sends the fixed prefix (system prompt, user
prompt, every tool schema) plus the accumulated history. `LLMNode` measures
both against `NodeToolLoopConfig.max_tool_loop_tokens` (default 128k, the
model's whole window) minus `tool_loop_token_reserve` (default 16 384, kept
free for the final answer):

- **While it fits, history is kept intact.** Tool results from earlier
  iterations are never truncated, so a task that cross-references two
  documents read in different rounds works.
- **When it stops fitting, the node consolidates.** The oldest iterations are
  summarised by the model into a `[CONSOLIDATED WORKING MEMORY]` block that
  *replaces* them (the two most recent rounds stay verbatim). This is
  irreversible on purpose.
- **If not even the prefix fits**, consolidation is skipped (it could not
  help) and the turn closes through the graceful exit.

Set a larger `max_tool_loop_tokens` for models with bigger windows.

### Closing a turn honestly

All four loop exits (max iterations or a repeated-call stall, with or
without streaming) end with one final tool-less LLM call carrying the same
guidance: summarise what was done, state exactly what remains, and never
attribute the stop to missing tools, permissions or connectors. A batch in
which *every* tool call was skipped as a repeat closes the turn only when it
happens twice in a row; a single stalled batch gets another round.

### Growing the tool catalog mid-turn

Tool specs are no longer frozen for the whole turn. Between iterations the
node reads `ToolRegistry.revision`; when it moved, the node re-fetches the
specs and re-assembles the prompt from the original (so schemas are never
duplicated). When it did not move, the cost is one integer comparison.

The revision moves on `ToolRegistry.clear_cache()` and also includes the
provider's own `revision`, if your `ToolProvider` exposes one (an optional
integer property that increments whenever `get_tools` would return something
different). That enables on-demand loading: expose a small catalog tool that,
when called, activates a connector's tools and bumps the revision — the model
sees them on the very next call instead of paying for every schema from the
first message.

### Prompt assembler ownership

If you pass your own `prompt_assembler` to `LLMNode`, it is authoritative:
the node never rebuilds it to inject `ToolRegistry` schemas into the prompt
text. Pass `PromptAssembler(config=..., tools=[])` when you want schemas to
travel only in the API's `tools` field. Without a custom assembler the node
keeps injecting registry tools into the prompt as before.

## Using Sofias-hosted inference

When your agent runs on the Sofias platform — in the portal sandbox or in
production — the platform provisions inference for it and injects the
gateway URL, key and model as the `SOFIAS_LLM_*` variables described above.
The bundled `GatewayLLM` picks them up on its own: you do not write a client,
and you never see or manage the credential.

To choose the model, set it in the agent's runtime settings in the developer
portal. To run the same agent on your machine, see
[Testing your agent](../guides/testing-your-agent.md): a fake LLM needs no
gateway at all.

# Running an agent

`AgentRunner` consumes tasks from a RabbitMQ queue, resolves settings,
executes your agent, and delivers the response — handling connection
lifecycle, retries at the transport level, and graceful shutdown.

## Minimal runner

```python
from sofias_sdk_lite import (
    AgentBuilder, AgentMessage, AgentRunner, BaseAgentSettings,
    InputContract, NodeContract, OutputContract, RabbitMQConfig, RunnerConfig,
)


class MySettings(BaseAgentSettings):
    model_name: str = "gpt-4o-mini"


class MyInput(InputContract):
    message: str
    role: str


class MyOutput(OutputContract):
    answer: str


class MyAgentRunner(AgentRunner):
    settings_class = MySettings

    def build_agent(self, settings, workflow):
        contract = NodeContract(input_schema=MyInput, output_schema=MyOutput)
        return (
            AgentBuilder("my_agent")
            .with_settings_class(MySettings)
            .with_contract(input_schema=MyInput, output_schema=MyOutput)
            .with_response_workflow(workflow)
            .add_llm_node("responder", ..., contract)
            .set_entry_node("responder")
            .set_terminal("responder")
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
            response_stream="my_agent.responses",
        )
    ).run()
```

## `RunnerConfig`

Everything queue/stream-related is explicit — there is no naming convention
linking an agent's identity to its queue:

- `queue` — the queue this runner consumes from
- `agent_name` — the agent's identity (used in `StreamFragment.agent` and
  passed to `ConfigSource.fetch()`)
- `rabbitmq` — a `RabbitMQConfig`
- `response_stream` — where `ChatResponseWorkflow` publishes (default
  `"agent.responses"` — override for your deployment)
- `role_map` — optional mapping from an incoming `user_role` to your own
  role vocabulary

## Per-request configuration

Pass a `ConfigSource` to resolve settings per agent/conversation instead of
hardcoding them:

```python
from sofias_sdk_lite import StaticConfigSource

runner = MyAgentRunner(
    config,
    config_source=StaticConfigSource({"model_name": "gpt-4o"}),
)
```

`StaticConfigSource` and `EnvConfigSource` (reads one JSON env var) are
built in. Implement `ConfigSource` yourself to fetch from a database, a
feature-flag service, or your own backend API — the runner merges whatever
it returns with `task.agent_configuration` from the incoming message
(payload wins on key conflicts) before validating against `settings_class`.

## Conversation history

Pass a `HistoryProvider` to persist turns across messages (default is
`InMemoryHistory`, which does not survive a restart):

```python
runner = MyAgentRunner(config, history=MyRedisHistory())
```

The history handed to `prepare_input` is windowed with defaults sized for
128k-class models: the most recent `max_messages` (100) turns, further
trimmed until they fit `max_context_tokens` (65 536, estimated at 3.2
chars/token). Override both as class attributes on your runner.

If your provider also implements `SummarizingHistoryProvider` (a
`maybe_summarize(conversation_id)` coroutine), the runner calls it after each
delivered response as a **background task**: it never delays the next
message, failures are logged rather than raised, and pending summaries are
drained on graceful shutdown. Set `auto_summarize = False` to opt out.

## Response delivery

`ChatResponseWorkflow` (the default) chunks the *final* answer into
`StreamFragment`s after execution finishes. Two behaviours are built in:

- **No silent failures.** A turn whose `AgentResponse.content` has no
  `answer`/`content` text (a graph that failed a downstream contract, for
  instance) is delivered as a short fallback message instead of an empty
  `completed` fragment, so the user learns the request was not handled.
- **Usage for billing.** The turn's token usage (`AgentResponse.usage`,
  summed across every LLM call) is attached to the terminal fragment as
  `state.usage = {"prompt_tokens", "completion_tokens", "model"}`.

### Token-by-token streaming

For real-time streaming, subclass `BaseStreamingResponseWorkflow` and return
it from `create_response_workflow`. It implements
`StreamingResponseWorkflow` (`on_stream_event` + `send_response`), owns the
wire mechanics (word-boundary-safe chunking, `StreamFragment` construction,
traceparent echo, usage, `replace_content` on the authoritative final text)
and exposes hooks for agent-specific logic:

```python
from sofias_sdk_lite import BaseStreamingResponseWorkflow


class MyStreamingWorkflow(BaseStreamingResponseWorkflow):
    def _final_text(self, response):
        return response.content.get("reply", "")

    def _extra_state(self, response):
        return {"attachments": response.content.get("attachments", [])}


class MyAgentRunner(AgentRunner):
    def create_response_workflow(self, client, task, *, traceparent=None):
        return MyStreamingWorkflow(
            self._config.agent_name,
            client=client,
            stream_name=self._config.response_stream,
            conversation_id=task.conversation_id,
            message_id=task.message_id,
            traceparent=traceparent,
        )
```

Other hooks: `_empty_response_fallback` (return `None` to opt out of the
fallback message), `_safe_flush_len` (hold back an unclosed tag),
`_should_buffer_text` (drop text for a turn), `_on_tool_event` (react to
tool-call events), plus `send_notice` for a non-error closing message.

## Trace continuity

The consumer surfaces the inbound W3C `traceparent` header on
`MessageContext.traceparent`; the runner passes it to
`create_response_workflow`, and both workflows echo it on every published
fragment (as an AMQP 1.0 application property on the stream), so a chat
turn is one trace end to end. With the optional `[otel]` extra installed
(`pip install "sofias-sdk-lite[otel]"`) the runner additionally opens an
`agent.handle` span per turn and the fragments carry *that* span's context,
nesting the downstream consumer under the agent. Without OpenTelemetry
everything degrades to a pure echo of the inbound header.

## Delegation

When `task.reply_to` is set on the incoming message, the request came from
another agent, not an end user. Use
`BaseDelegationResponseWorkflow` instead of `ChatResponseWorkflow` in that
case — see [Delegation](delegation.md).

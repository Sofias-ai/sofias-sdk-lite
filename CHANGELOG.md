# Changelog

All notable changes to `sofias-sdk-lite` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). While the
version is below 1.0, a minor release (0.x.0) may contain breaking changes.

## [0.2.1] - 2026-10-02

### Fixed

- API reference: docstring examples are now rendered as code. Their comment
  lines were being read as headings, which polluted the page menu.
- The `Agent` example called `add_node`, which does not exist; it now uses
  `add_llm_node`.
- `create_llm()` raises the missing-URL `LLMConfigurationError` before logging
  the "no API key" warning, so a misconfigured agent reports one clear problem.

### Changed

- Documentation: the reference menu lists sections and classes only, with
  short names; private and dunder members are no longer listed.

## [0.2.0] - 2026-09-30

### Changed (breaking)

- Licence: from this release the SDK is distributed under the Sofias SDK Lite
  License. The source stays available to read, and the SDK may be used to
  develop, test and run agents for the Sofias platform; the agents you build
  remain yours. Earlier releases keep the licence they were published with.
- The bundled gateway client is now `GatewayLLM`, selected with the provider
  label `"sofias"`, together with `PROVIDER_GATEWAY`,
  `DEFAULT_GATEWAY_MAX_TOKENS` and `normalize_gateway_base_url`. They replace
  the previous gateway client class and its constants.
- There is no default gateway address any more. When your agent runs on the
  Sofias platform, `SOFIAS_LLM_BASE_URL` is injected for it. Configuring a
  model without a URL now raises `LLMConfigurationError`
  (`missing=["base_url"]`) with instructions, instead of silently pointing at
  an address that only resolves inside the platform.
- Gateway configuration is read from `SOFIAS_LLM_MODEL`,
  `SOFIAS_LLM_BASE_URL`, `SOFIAS_LLM_API_KEY` and `SOFIAS_LLM_PROVIDER`, then
  from `MODEL_NAME` / `ROUTER_URL` / `ROUTER_API_KEY`. The previous
  gateway-specific variable names are no longer read.

### Added

- Project links (homepage, documentation, source, changelog) in the package
  metadata.

### Migration

- If you imported the previous gateway client or its constants directly,
  import `GatewayLLM` and the `*_GATEWAY_*` names instead. Agents that rely on
  `create_llm()` or on the default LLM need no code change.
- To run locally against a real gateway, export `SOFIAS_LLM_BASE_URL` along
  with `SOFIAS_LLM_MODEL`. To run without one, install a fake LLM with
  `default_llm(...)` as shown in the testing guide.

## [0.1.5] - 2026-09-25

### Added

- Mid-turn tool catalog refresh: `ToolRegistry.revision` moves on
  `clear_cache()` and adds the provider's optional `revision`; `LLMNode`
  checks it between tool-loop iterations (streaming and non-streaming) and,
  only when it moved, re-fetches the specs and re-assembles the prompt from
  the original. The per-node tools cache is invalidated when the provider's
  revision changes, so a provider that swaps its catalog on its own is no
  longer served stale specs.

## [0.1.4] - 2026-09-21

### Added

- LLM abstraction layer: agents no longer construct an LLM client. A bundled
  gateway client (per-family `reasoning_effort`, 16 384-token output budget)
  and `OpenAICompatibleLLM` (any OpenAI-style `/chat/completions` endpoint,
  `httpx` only) implement `StreamableLLMCallable` with retries, SSE
  streaming, usage and gateway provider metadata. `create_llm()` resolves the
  client from an explicit `LLMGatewayConfig`, the agent settings
  (`model_name` / `router_url` / `router_api_key`) or the environment.
  `AgentBuilder.build()` uses the context default (`default_llm()`) or the
  environment when `with_llm()` is omitted; `AgentRunner` builds and caches
  one client per gateway configuration and exposes a `create_llm(settings)`
  hook.
- New errors: `LLMConfigurationError`, `LLMRequestError`,
  `EmptyLLMResponseError`.
- New dependency: `httpx`.

### Changed

- `BaseAgentSettings` gains `model_name`, `router_url`, `router_api_key`
  (`SecretStr`, listed in `CREDENTIAL_FIELDS`) and `router_provider`, all
  optional, plus `has_llm_gateway`.
- `StreamableLLMCallable.stream_invoke` is declared as a plain `def`
  returning `AsyncIterator` (the async-generator shape every implementation
  has), and both protocol methods accept `parallel_tool_calls`.
- `AgentBuilder` no longer requires `with_llm()` when LLM or planner nodes
  exist; the build error, when nothing resolves, names the three fixes.

## [0.1.3] - 2026-09-21

### Added

- `BaseStreamingResponseWorkflow`: first concrete `StreamingResponseWorkflow`
  (token-by-token publishing over RabbitMQ streams) with word-boundary-safe
  chunking, `replace_content` on the authoritative final text, usage
  attachment, traceparent echo and override hooks (`_final_text`,
  `_extra_state`, `_empty_response_fallback`, `_safe_flush_len`,
  `_should_buffer_text`, `_on_tool_event`, `send_notice`).
- `usage_state_from_list`: collapses `AgentResponse.usage` into the
  `state.usage` object billing consumers read from the terminal fragment.
- Tool-loop token budget: `NodeToolLoopConfig.max_tool_loop_tokens` (default
  128k, the whole window) and `tool_loop_token_reserve` (default 16 384); the
  fixed prefix (system prompt + tool schemas) is measured, not assumed.
- Working-memory consolidation: when the budget is exhausted the oldest tool
  iterations are summarised by the model into a `[CONSOLIDATED WORKING
  MEMORY]` block that replaces them, instead of stopping the loop.
- W3C `traceparent` continuity: `MessageContext.traceparent`,
  `PublishOptions.traceparent` (AMQP header on queues, AMQP 1.0 application
  property on streams), echoed by both response workflows; `AgentRunner`
  opens an `agent.handle` span per turn when the optional `[otel]` extra is
  installed and records exceptions on it.
- `BaseAgentSettings.CREDENTIAL_FIELDS`: fields excluded from prompt
  template variables (declare them as `SecretStr`).
- History window on `AgentRunner` (`max_messages=100`,
  `max_context_tokens=65_536`, `auto_summarize=True`) and the
  `SummarizingHistoryProvider` protocol: `maybe_summarize` runs as a
  fire-and-forget background task after each response and is drained on
  shutdown.
- The tool-call trace (`"Tool calls in iteration"`, names only) is now
  emitted on the streaming path too.
- `__version__` derived from installed metadata (`importlib.metadata`).

### Changed

- Response workflows never publish an empty `completed` fragment for a
  failed or empty `AgentResponse`; a fallback message is substituted.
- A caller-supplied `prompt_assembler` on `LLMNode` is authoritative: the
  runtime no longer rebuilds it to inject `ToolRegistry` schemas into the
  prompt text.
- All tool-loop exits share one honest closing instruction that asks the
  model to state what remains and forbids blaming tools, access, permissions
  or connectors. A batch of all-skipped repeated tool calls closes the turn
  only on the second consecutive occurrence.
- `AgentRunner.create_response_workflow` gains a keyword-only `traceparent`
  argument (forward it when overriding).

## [0.1.2] - 2026-07-31

### Changed

- Documentation.

## [0.1.1] - 2026-07-30

### Fixed

- The publisher now passes the configured RabbitMQ virtual host when it
  connects.

## [0.1.0] - 2026-07-29

Initial public release.

### Added

- Agent graph: `Agent`, `AgentBuilder`, `BaseNode`, `FunctionNode`, `LLMNode`,
  `DelegationNode`, `AggregatorNode`, `PlannerNode`, routing strategies.
- Contracts: `StrictContract`, `InputContract`/`OutputContract`,
  `AgentMessage`/`AgentResponse`, delegation contracts.
- Error handling: `ErrorHandler` with retry (configurable backoff), circuit
  breaker, and fallback routing.
- LLM protocol: `LLMCallable`/`StreamableLLMCallable`, no bundled provider
  adapter — bring your own client.
- Tool execution framework: `BaseTool`, `ToolRegistry`, `ToolProvider`.
- Messaging models: `AgentTaskMessage`, `StreamFragment`, delegation
  contracts, wire-compatible with the Sofias platform.
- RabbitMQ transport: `RabbitMQClient`, `RabbitMQConsumer`,
  `RabbitMQPublisher`, `RabbitMQRPCClient`, `RabbitMQDelegationTransport`.
- `AgentRunner`/`RunnerConfig`: consume tasks from RabbitMQ, resolve
  per-request settings via `ConfigSource`, execute, deliver via
  `ChatResponseWorkflow`, graceful shutdown on SIGTERM/SIGINT.
- `ConfigSource` protocol with `StaticConfigSource`/`EnvConfigSource`
  implementations.
- State protocols: `ConversationStateProvider`, `MemoryProvider` (protocol
  only, no concrete backend), `HistoryProvider`/`InMemoryHistory`.
- Observability protocols: `TracingProvider`/`Span`, `EventPublisher`/`EventListener`
  (protocol only — bring your own exporter/broker).

### Scope notes

- Memory (long-term summaries/facts), MCP tool discovery, and LLM adapter
  implementations were intentionally out of scope for 0.1.0 — only their
  protocols shipped (`MemoryProvider`, `ToolProvider`, `LLMCallable`).
- LangGraph compilation is out of scope (candidate for a future
  `[langgraph]` extra).

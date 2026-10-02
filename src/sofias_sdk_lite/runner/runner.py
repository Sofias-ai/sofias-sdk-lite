"""AgentRunner: consumes tasks from RabbitMQ, executes an agent, delivers the response.

Subclass and implement:
- build_agent(settings, workflow) -> Agent
- prepare_input(task, history, role) -> AgentMessage

Example:
    ```python

    class MyAgentRunner(AgentRunner):
        settings_class = MySettings

        def build_agent(self, settings, workflow):
            return (
                AgentBuilder("my_agent")
                .with_settings_class(type(settings))
                .with_response_workflow(workflow)
                ...
                .build()
            )

        def prepare_input(self, task, history, role):
            return AgentMessage(
                content=MyInput(message=task.content, role=role),
                conversation_id=task.conversation_id,
            )

    if __name__ == "__main__":
        MyAgentRunner(RunnerConfig(
            queue="my-agent-tasks",
            agent_name="my_agent",
            rabbitmq=RabbitMQConfig(host="localhost"),
        )).run()
    ```
"""

from __future__ import annotations

import asyncio
import signal
import sys
from abc import ABC, abstractmethod
from typing import Any

from sofias_sdk_lite.agent.agent import Agent
from sofias_sdk_lite.config.agent_settings import BaseAgentSettings
from sofias_sdk_lite.config.source import ConfigSource
from sofias_sdk_lite.errors.exceptions import ConfigSourceError
from sofias_sdk_lite.llm.factory import (
    LLMGatewayConfig,
    create_llm,
    default_llm,
    gateway_config_from_settings,
)
from sofias_sdk_lite.llm.protocol import LLMCallable
from sofias_sdk_lite.messaging.models import AgentTaskMessage
from sofias_sdk_lite.observability._log import get_logger
from sofias_sdk_lite.observability.trace_context import consumer_span
from sofias_sdk_lite.rabbitmq.client import RabbitMQClient
from sofias_sdk_lite.rabbitmq.consumer import RabbitMQConsumer
from sofias_sdk_lite.rabbitmq.types import MessageContext
from sofias_sdk_lite.runner.config import RunnerConfig
from sofias_sdk_lite.state.history import HistoryProvider, InMemoryHistory, Message
from sofias_sdk_lite.workflows.chat import ChatResponseWorkflow

logger = get_logger("runner")

__all__ = ["AgentRunner"]

# Rough chars-per-token ratio, consistent with LLMNode's tool-loop estimate.
_CHARS_PER_TOKEN = 3.2


class AgentRunner(ABC):
    """Base class for running an agent against RabbitMQ.

    Handles the RabbitMQ connection, consumer loop, per-message config
    resolution, conversation history, response delivery, and graceful
    shutdown on SIGTERM/SIGINT. Subclasses only need to describe how to
    build the agent and how to shape its input.

    Each turn runs inside an ``agent.handle`` span (when OpenTelemetry is
    installed) and the inbound W3C ``traceparent`` is echoed on every
    response fragment, so a chat turn is one trace end to end.

    History handed to ``prepare_input`` is windowed by ``max_messages`` and
    ``max_context_tokens``; a ``SummarizingHistoryProvider`` gets
    ``maybe_summarize`` called in the background after each response.
    """

    settings_class: type[BaseAgentSettings] = BaseAgentSettings

    # --- Conversation-history window (defaults sized for 128k-class models) ---
    max_messages: int = 100
    """Most recent turns handed to ``prepare_input`` (oldest dropped first)."""

    max_context_tokens: int = 65_536
    """Estimated token cap for the history handed to ``prepare_input``; oldest
    turns are dropped until the remainder fits (the latest turn always stays)."""

    auto_summarize: bool = True
    """After each delivered response, call ``maybe_summarize`` on the history
    provider (if it implements ``SummarizingHistoryProvider``) as a background
    task. Never blocks the next message; drained on graceful shutdown."""

    def __init__(
        self,
        config: RunnerConfig,
        *,
        config_source: ConfigSource | None = None,
        history: HistoryProvider | None = None,
    ) -> None:
        self._config = config
        self._config_source = config_source
        self._history = history or InMemoryHistory()
        self._background_tasks: set[asyncio.Task[None]] = set()
        # One client per distinct gateway configuration, reused across tasks.
        self._llm_cache: dict[LLMGatewayConfig, LLMCallable] = {}

    @abstractmethod
    def build_agent(self, settings: BaseAgentSettings, workflow: Any) -> Agent:
        """Build the Agent instance for a single message.

        Args:
            settings: Resolved settings for this invocation (from
                `ConfigSource.fetch()` merged with `task.agent_configuration`,
                validated against `settings_class`).
            workflow: The `ResponseWorkflow` to attach to the agent (a
                `ChatResponseWorkflow` for conversational tasks, or a
                `BaseDelegationResponseWorkflow` when `task.reply_to` is set).
        """
        ...

    def create_llm(self, settings: BaseAgentSettings) -> LLMCallable | None:
        """Build (or reuse) the LLM client for this task's settings.

        The default reads ``model_name`` / ``router_url`` / ``router_api_key``
        from ``settings`` through `sofias_sdk_lite.llm.create_llm` and caches
        one client per distinct gateway configuration. Returning ``None``
        means "no LLM from settings": `AgentBuilder.build()` then falls back
        to the ``SOFIAS_LLM_*`` environment variables.

        Override to plug in a different client (a fake in tests, a provider
        the bundled adapters do not cover, per-tenant routing, ...).
        """
        gateway = gateway_config_from_settings(settings)
        if gateway is None:
            return None
        llm = self._llm_cache.get(gateway)
        if llm is None:
            llm = create_llm(gateway)
            self._llm_cache[gateway] = llm
        return llm

    @abstractmethod
    def prepare_input(
        self,
        task: AgentTaskMessage,
        history: list[Message],
        role: str,
    ) -> Any:
        """Build the `AgentMessage` to pass to `Agent.execute()`.

        Args:
            task: The incoming task message.
            history: Prior messages for this conversation, oldest first.
            role: The resolved role for this request (see `resolve_role`).
        """
        ...

    def resolve_role(self, payload: dict[str, Any]) -> str:
        """Map an incoming `user_role` through `RunnerConfig.role_map`.

        Override for custom resolution logic. Unmapped roles pass through
        unchanged rather than raising or defaulting silently.
        """
        raw_role = payload.get("user_role") or ""
        return self._config.role_map.get(raw_role, raw_role)

    def create_response_workflow(
        self,
        client: RabbitMQClient,
        task: AgentTaskMessage,
        *,
        traceparent: str | None = None,
    ) -> Any:
        """Create the response workflow for a single message.

        Returns a `ChatResponseWorkflow` publishing to
        `RunnerConfig.response_stream`. Override to use a different
        transport or workflow (e.g. a `BaseStreamingResponseWorkflow`
        subclass for token-by-token streaming).

        Args:
            client: The connected RabbitMQ client.
            task: The incoming task message.
            traceparent: W3C ``traceparent`` received on the inbound message.
                Echoing it onto the response fragments keeps a chat turn a
                single trace across the whole pipeline. Forward it to your
                workflow when overriding.
        """
        message_id = task.message_id if task.message_id is not None else task.conversation_id
        return ChatResponseWorkflow(
            agent_name=self._config.agent_name,
            client=client,
            stream_name=self._config.response_stream,
            conversation_id=task.conversation_id,
            message_id=message_id,
            traceparent=traceparent,
        )

    async def after_execution(self, task: AgentTaskMessage, response: Any) -> None:
        """Hook called after successful agent execution. Override for custom logic."""

    def _window_history(self, messages: list[Message]) -> list[Message]:
        """Trim history to ``max_messages`` and ``max_context_tokens`` (most recent kept).

        The latest turn is always kept, even if it alone exceeds the token cap;
        a model call with no context beats a silently empty history.
        """
        window = messages[-self.max_messages :] if self.max_messages > 0 else list(messages)
        while len(window) > 1:
            estimated = sum(len(m.content) for m in window) / _CHARS_PER_TOKEN
            if estimated <= self.max_context_tokens:
                break
            window = window[1:]
        if len(window) < len(messages):
            logger.debug(
                "History windowed",
                kept=len(window),
                dropped=len(messages) - len(window),
                max_messages=self.max_messages,
                max_context_tokens=self.max_context_tokens,
            )
        return window

    async def _safe_summarize(self, conversation_id: str) -> None:
        try:
            await self._history.maybe_summarize(conversation_id)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.error(
                "Post-response summary failed",
                conversation_id=conversation_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )

    def _schedule_summary(self, conversation_id: str) -> None:
        """Fire-and-forget ``maybe_summarize`` so it never delays the next message."""
        if not self.auto_summarize or not hasattr(self._history, "maybe_summarize"):
            return
        task = asyncio.create_task(self._safe_summarize(conversation_id))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _drain_background_tasks(self) -> None:
        """Await pending summaries so a shutdown never drops one mid-flight."""
        if not self._background_tasks:
            return
        logger.info("Draining background tasks", count=len(self._background_tasks))
        await asyncio.gather(*self._background_tasks, return_exceptions=True)

    def run(self) -> None:
        """Entry point: connect, consume, run until shutdown."""
        try:
            asyncio.run(self._main())
        except KeyboardInterrupt:
            sys.exit(0)

    async def _main(self) -> None:
        logger.info("Starting agent runner", agent_name=self._config.agent_name)

        async with RabbitMQClient(self._config.rabbitmq) as client:

            async def handle_message(ctx: MessageContext) -> None:
                await self._handle_message(ctx, client)

            consumer = RabbitMQConsumer(
                client=client,
                queue_name=self._config.queue,
                handler=handle_message,
            )
            await consumer.start()
            logger.info("Runner ready", queue=self._config.queue)

            shutdown_event = asyncio.Event()
            loop = asyncio.get_running_loop()

            def _request_shutdown() -> None:
                logger.info("Shutdown signal received")
                shutdown_event.set()

            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, _request_shutdown)

            try:
                await shutdown_event.wait()
            finally:
                await self._drain_background_tasks()
                await consumer.stop()
                logger.info("Shutdown complete")

    async def _handle_message(self, ctx: MessageContext, client: RabbitMQClient) -> None:
        try:
            task = AgentTaskMessage.model_validate(ctx.payload)
        except Exception:
            logger.error("Failed to parse incoming message", payload_keys=list(ctx.payload))
            return

        # Wrap the whole turn in a root ``agent.handle`` CONSUMER span, child
        # of the inbound W3C traceparent. Being the *active* context, the
        # response fragments hang off it and a downstream consumer nests under
        # the agent. Pure no-op without OpenTelemetry installed/configured.
        with consumer_span(
            "agent.handle",
            traceparent=ctx.traceparent,
            attributes={
                "agent.name": self._config.agent_name,
                "conversation.id": task.conversation_id,
                "message.id": str(task.message_id),
                "tenant.identifier": task.tenant_identifier,
                "messaging.system": "rabbitmq",
                "messaging.destination.name": self._config.queue,
                "messaging.operation": "process",
                "agent.traceparent.incoming.present": bool(ctx.traceparent),
            },
        ) as span:
            await self._handle_task(ctx, client, task, span)

    async def _handle_task(
        self,
        ctx: MessageContext,
        client: RabbitMQClient,
        task: AgentTaskMessage,
        span: Any,
    ) -> None:
        workflow = self.create_response_workflow(client, task, traceparent=ctx.traceparent)

        try:
            resolved_config: dict[str, Any] = {}
            if self._config_source is not None:
                try:
                    resolved_config = await self._config_source.fetch(
                        self._config.agent_name, task.conversation_id
                    )
                except ConfigSourceError as exc:
                    logger.info("Config source lookup failed, using payload only", reason=str(exc))

            merged_config = {**resolved_config, **task.agent_configuration}
            known_fields = set(self.settings_class.model_fields)
            filtered_config = {k: v for k, v in merged_config.items() if k in known_fields}
            settings = self.settings_class(**filtered_config)

            role = self.resolve_role(ctx.payload)

            history = self._window_history(await self._history.get(task.conversation_id))
            history_with_user_turn = [
                *history,
                Message(role="user", content=task.content),
            ]

            with default_llm(self.create_llm(settings)):
                agent = self.build_agent(settings, workflow)
            message = self.prepare_input(task, history_with_user_turn, role)
            response = await agent.execute(message)

            answer = ""
            if response.content:
                answer = response.content.get("answer", "") or response.content.get("content", "")

            await self._history.append(
                task.conversation_id, Message(role="user", content=task.content)
            )
            await self._history.append(
                task.conversation_id, Message(role="assistant", content=answer)
            )
            self._schedule_summary(task.conversation_id)

            await self.after_execution(task, response)
            logger.info("Agent execution finished", conversation_id=task.conversation_id)

        except Exception as exc:
            span.record_error(exc)
            logger.error(
                "Message handling failed",
                conversation_id=task.conversation_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            await workflow.send_error("An internal error occurred while processing this message.")
        finally:
            await workflow.close()

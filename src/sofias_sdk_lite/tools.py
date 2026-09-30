"""Tool execution framework: BaseTool, ToolRegistry, and TaskManager.

This is the canonical home for tool *execution* concerns (running a tool,
retries, task dispatch). It is distinct from ``sofias_sdk_lite.llm``, which
only describes tools to an LLM (``ToolSpec``/``ToolCall``) without knowing
how to run them.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from sofias_sdk_lite.llm.tools import ToolSpec
from sofias_sdk_lite.observability._log import get_logger

logger = get_logger("tools")

__all__ = [
    "DEFAULT_NAME",
    "DEFAULT_TIMEOUT",
    "DEFAULT_RETRIES",
    "DEFAULT_RETRY_DELAY",
    "DEFAULT_BACKOFF_FACTOR",
    "DEFAULT_ENABLED",
    "DEFAULT_LOG_LEVEL",
    "merge_dicts",
    "EnvConfig",
    "BaseToolConfig",
    "ToolContext",
    "ExecutionResult",
    "BaseTool",
    "ToolProvider",
    "ToolRegistry",
    "NullToolProvider",
    "TaskType",
    "TaskManager",
    "resolve_task_type",
]

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_NAME: str = "sofias_tool"
DEFAULT_TIMEOUT: float = 30.0
DEFAULT_RETRIES: int = 3
DEFAULT_RETRY_DELAY: float = 1.0
DEFAULT_BACKOFF_FACTOR: float = 2.0
DEFAULT_ENABLED: bool = True
DEFAULT_LOG_LEVEL: str = "INFO"


def merge_dicts(
    base: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a shallow merge of *base* and *overrides* dictionaries."""
    result: dict[str, Any] = {}
    if base:
        result.update(base)
    if overrides:
        result.update(overrides)
    return result


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

_ENV_FIELD_MAP: dict[str, str] = {
    "api_key": "API_KEY",
    "api_base_url": "API_BASE_URL",
    "database_url": "DATABASE_URL",
    "notification_channel": "NOTIFICATION_CHANNEL",
}


@dataclass(slots=True)
class EnvConfig:
    """Environment-driven configuration shared by tools."""

    api_key: str | None = None
    api_base_url: str | None = None
    database_url: str | None = None
    notification_channel: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, prefix: str = "SOFIAS_TOOL_") -> "EnvConfig":
        """Load configuration values from environment variables."""
        values: dict[str, str | None] = {}
        for attr, suffix in _ENV_FIELD_MAP.items():
            values[attr] = os.getenv(f"{prefix}{suffix}")

        extra: dict[str, str] = {}
        prefix_len = len(prefix)
        for key, value in os.environ.items():
            if key.startswith(prefix) and key[prefix_len:] not in _ENV_FIELD_MAP.values():
                extra[key[prefix_len:]] = value

        return cls(
            api_key=values.get("api_key"),
            api_base_url=values.get("api_base_url"),
            database_url=values.get("database_url"),
            notification_channel=values.get("notification_channel"),
            extra=extra,
        )

    def as_dict(self) -> dict[str, str | None]:
        """Return a mapping with all loaded values."""
        data: dict[str, str | None] = {
            "api_key": self.api_key,
            "api_base_url": self.api_base_url,
            "database_url": self.database_url,
            "notification_channel": self.notification_channel,
        }
        data.update({f"extra_{k.lower()}": v for k, v in self.extra.items()})
        return data

    def require(self, field_name: str) -> str:
        """Return the value of *field_name* or raise a descriptive error."""
        value = getattr(self, field_name, None)
        if not value:
            raise RuntimeError(f"Missing required environment configuration: {field_name}")
        return value


# ---------------------------------------------------------------------------
# Tool configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BaseToolConfig:
    """Configuration contract that every tool must honor."""

    name: str = DEFAULT_NAME
    description: str = ""
    enabled: bool = DEFAULT_ENABLED
    timeout: float = DEFAULT_TIMEOUT
    retries: int = DEFAULT_RETRIES
    retry_delay: float = DEFAULT_RETRY_DELAY
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR
    log_level: str = DEFAULT_LOG_LEVEL
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if self.retries < 0:
            raise ValueError("retries must be zero or a positive integer")
        if self.retry_delay < 0:
            raise ValueError("retry_delay cannot be negative")
        if self.backoff_factor < 1:
            raise ValueError("backoff_factor must be greater or equal to 1")
        if not self.log_level:
            raise ValueError("log_level cannot be empty")

    def as_dict(self) -> dict[str, Any]:
        """Return a serialisable representation of the configuration."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def with_overrides(self, overrides: dict[str, Any] | None = None) -> "BaseToolConfig":
        """Produce a new config applying the provided overrides."""
        if not overrides:
            return self
        allowed = {f.name for f in fields(self)}
        update_kwargs = {k: v for k, v in overrides.items() if k in allowed}
        if not update_kwargs:
            return self
        candidate = replace(self, **update_kwargs)
        candidate._validate()
        return candidate

    def merge_metadata(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return metadata merged with additional key-value pairs."""
        merged = dict(self.metadata)
        if extra:
            merged.update(extra)
        return merged


# ---------------------------------------------------------------------------
# Execution types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolContext:
    """Container with shared context available during tool execution."""

    config: BaseToolConfig
    env: EnvConfig
    payload: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    payload_model: type[BaseModel] | None = None

    def with_payload(self, payload: Any) -> "ToolContext":
        """Create a new context with the given payload."""
        return ToolContext(
            config=self.config,
            env=self.env,
            payload=payload,
            metadata=dict(self.metadata),
            payload_model=self.payload_model,
        )


@dataclass(slots=True)
class ExecutionResult:
    """Standard result object returned by tool executions."""

    success: bool
    data: Any = None
    error: BaseException | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def unwrap(self) -> Any:
        """Return data or raise if execution failed."""
        if not self.success:
            raise RuntimeError(f"Execution failed: {self.error}")
        return self.data


class BaseTool(ABC):
    """Abstract base class that standardizes tool behavior.

    Subclasses must implement ``run``. Logging is intentionally not
    provided here — configure your own logger per tool if needed.
    """

    def __init__(
        self,
        *,
        config: BaseToolConfig | None = None,
        env: EnvConfig | None = None,
        input_model: type[BaseModel] | None = None,
    ) -> None:
        self.config = config or BaseToolConfig()
        self.env = env or EnvConfig.load()
        self.input_model = input_model

    def update_config(self, overrides: dict[str, Any] | None = None) -> None:
        """Apply configuration overrides."""
        self.config = self.config.with_overrides(overrides)

    def execute(self, context: ToolContext | None = None) -> ExecutionResult:
        """Run the tool using the provided context or defaults."""
        ctx = context or ToolContext(config=self.config, env=self.env)
        if ctx.payload_model is None:
            ctx.payload_model = self.input_model
        ctx.metadata = merge_dicts(ctx.metadata, {"tool": self.config.name})

        if not self.config.enabled:
            return ExecutionResult(
                success=False, error="Tool is disabled", metadata={"reason": "disabled"}
            )

        try:
            self.before_execute(ctx)
            ctx = self._validate_payload(ctx)
            result = self.run(ctx)
        except Exception as exc:
            result = ExecutionResult(success=False, error=exc)
        else:
            self.after_execute(ctx, result)

        return result

    @abstractmethod
    def run(self, context: ToolContext) -> ExecutionResult:
        """Concrete tools must implement their execution logic here."""

    def before_execute(self, context: ToolContext) -> None:
        """Hook executed before the main run call."""

    def after_execute(self, context: ToolContext, result: ExecutionResult) -> None:
        """Hook executed after the main run call."""

    def cleanup(self) -> None:
        """Hook executed when tearing down the tool runtime."""

    def _validate_payload(self, context: ToolContext) -> ToolContext:
        model = context.payload_model
        if model is None:
            return context
        payload = context.payload
        try:
            validated = payload if isinstance(payload, model) else model.model_validate(payload)
        except ValidationError as exc:
            # Build a compact, LLM-friendly error so the model can self-correct
            # on retry. Without these details the LLM tends to repeat the same
            # broken tool call on complex nested schemas.
            errors = []
            for err in exc.errors():
                loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
                msg = err.get("msg", "invalid")
                got = err.get("input")
                try:
                    got_repr = repr(got)
                    if len(got_repr) > 120:
                        got_repr = got_repr[:117] + "..."
                except Exception:
                    got_repr = "<unrepr>"
                errors.append(f"{loc}: {msg} (got={got_repr})")
            detail = "; ".join(errors) if errors else str(exc)
            logger.warning(
                "Payload validation failed",
                tool=self.config.name,
                model=model.__name__,
                errors=errors,
            )
            raise RuntimeError(f"Payload validation failed: {detail}") from exc
        new_context = context.with_payload(validated)
        new_context.metadata["validated_payload_model"] = model.__name__
        return new_context


# ---------------------------------------------------------------------------
# Tool registry (mediates between nodes and an external tool provider)
# ---------------------------------------------------------------------------


@runtime_checkable
class ToolProvider(Protocol):
    """Protocol defining the interface for external tool providers.

    Implement this against whatever tool system you want to expose to an
    agent (MCP servers, a plugin registry, standalone functions, ...).

    Optional member ``revision: int``: a provider whose catalog can change
    mid-turn (e.g. a tool that loads a connector's tools on demand) may expose
    a ``revision`` property that increments on every change. The node reads it
    between tool-loop iterations and only then re-fetches the specs. Providers
    that do not expose it behave exactly as before.
    """

    async def get_tools(self, node_name: str) -> list[ToolSpec]:
        """Get tools available for a specific node."""
        ...

    async def execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> Any:
        """Execute a tool by name with the given input."""
        ...


class ToolRegistry:
    """Registry that mediates between SDK nodes and an external ToolProvider.

    Wraps a ToolProvider and adds optional per-node caching, error handling,
    and a consistent interface for nodes.
    """

    def __init__(self, provider: ToolProvider, *, cache_tools: bool = True) -> None:
        self._provider = provider
        self._cache_tools = cache_tools
        self._tools_cache: dict[str, list[ToolSpec]] = {}
        self._cached_at_revision: dict[str, int] = {}
        self._revision = 0

    @property
    def provider(self) -> ToolProvider:
        """The underlying tool provider."""
        return self._provider

    @property
    def revision(self) -> int:
        """Counter that moves whenever the catalog may have changed.

        The node reads it between tool-loop iterations: while it stays put the
        node reuses the specs it already has, so an agent that never touches
        its catalog pays nothing. It moves on ``clear_cache`` and also adds the
        provider's own ``revision`` when exposed (a cacheless provider can
        change tools without going through ``clear_cache``). Only comparing
        two reads is meaningful.
        """
        provider_revision = getattr(self._provider, "revision", 0)
        return self._revision + (provider_revision if isinstance(provider_revision, int) else 0)

    async def get_tools_for_node(self, node_name: str) -> list[ToolSpec]:
        """Get tools available for a specific node, caching if enabled."""
        # A revision different from the one cached at means the provider
        # changed its catalog on its own: the cached list is stale.
        revision = self.revision
        if (
            self._cache_tools
            and node_name in self._tools_cache
            and self._cached_at_revision.get(node_name) == revision
        ):
            return self._tools_cache[node_name]

        try:
            tools = await self._provider.get_tools(node_name)
            if self._cache_tools:
                self._tools_cache[node_name] = tools
                self._cached_at_revision[node_name] = revision
            logger.debug(
                "Loaded tools for node",
                node_name=node_name,
                count=len(tools),
                tool_names=[t.name for t in tools],
            )
            return tools
        except Exception as e:
            logger.error("Failed to get tools for node", node_name=node_name, error=str(e))
            return []

    async def execute(self, tool_name: str, tool_input: dict[str, Any]) -> Any:
        """Execute a tool by name, delegating to the provider."""
        logger.debug("Executing tool", tool_name=tool_name)
        try:
            result = await self._provider.execute_tool(tool_name, tool_input)
            logger.debug("Tool execution succeeded", tool_name=tool_name)
            return result
        except Exception as e:
            logger.error("Tool execution failed", tool_name=tool_name, error=str(e))
            raise

    def clear_cache(self, node_name: str | None = None) -> None:
        """Clear the tools cache, for one node or entirely.

        Moves ``revision``, which is how the node learns the catalog changed
        and re-fetches the specs on its next iteration.
        """
        if node_name is not None:
            self._tools_cache.pop(node_name, None)
            self._cached_at_revision.pop(node_name, None)
        else:
            self._tools_cache.clear()
            self._cached_at_revision.clear()
        self._revision += 1

    def get_cached_tools(self, node_name: str) -> list[ToolSpec] | None:
        """Get cached tools for a node without fetching."""
        return self._tools_cache.get(node_name)


class NullToolProvider:
    """A no-op tool provider for agents that don't use tools."""

    async def get_tools(self, node_name: str) -> list[ToolSpec]:
        """Return an empty tool list."""
        return []

    async def execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> Any:
        """Raise since no tools are configured."""
        raise ValueError(f"No tool provider configured. Cannot execute tool '{tool_name}'.")


# ---------------------------------------------------------------------------
# Task manager (decides sync vs. background tool execution)
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    """Execution strategy for a tool call."""

    QUICK = "quick"
    BACKGROUND = "background"


def resolve_task_type(
    tool_metadata: dict[str, Any],
    task_defaults: dict[str, str] | None,
    tool_name: str,
) -> TaskType:
    """Resolve the task type for a tool call using the priority chain.

    Resolution order:
    1. ``tool.config.metadata["task_type"]`` — explicit tool-level override
    2. ``task_defaults[tool_name]`` — agent-level per-tool defaults
    3. ``TaskType.QUICK`` — fallback default
    """
    metadata_type = tool_metadata.get("task_type")
    if metadata_type is not None:
        return TaskType(metadata_type)

    if task_defaults is not None:
        default_type = task_defaults.get(tool_name)
        if default_type is not None:
            return TaskType(default_type)

    return TaskType.QUICK


@runtime_checkable
class TaskManager(Protocol):
    """Protocol defining how tool executions are managed as tasks.

    Implementations handle the execution strategy: synchronous for QUICK
    tasks, deferred/async for BACKGROUND tasks. Injected into nodes via
    ``AgentBuilder.with_task_manager()``.
    """

    async def execute_tool(
        self,
        tool: BaseTool,
        tool_call_name: str,
        tool_call_id: str | None,
        tool_input: dict[str, Any],
        task_defaults: dict[str, str] | None = None,
    ) -> Any:
        """Execute a tool call as a managed task and return a ToolResult-shaped value.

        The implementation should:
        1. Resolve the task type via ``resolve_task_type()``.
        2. Execute synchronously (QUICK) or dispatch (BACKGROUND).
        3. Return the result (see ``sofias_sdk_lite.nodes.llm_node.ToolResult``).
        """
        ...

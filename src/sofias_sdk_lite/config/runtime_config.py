"""Runtime configuration for agents.

This module provides the base class for agent-specific runtime settings
and the resolver that validates and manages them in async contexts.
"""

from contextvars import ContextVar
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from sofias_sdk_lite.errors.exceptions import AgentSDKError


class RuntimeConfigError(AgentSDKError):
    """Error when runtime configuration is invalid or missing."""

    def __init__(
        self,
        message: str,
        *,
        settings_class: str,
        validation_errors: list[Any] | None = None,
    ) -> None:
        self.settings_class = settings_class
        self.validation_errors = validation_errors or []
        super().__init__(
            message,
            settings_class=settings_class,
            validation_errors=self.validation_errors,
        )


class AgentSettings(BaseModel):
    """Base class for agent-specific runtime settings.

    Concrete agents should inherit from this class to declare their
    required and optional runtime configuration variables.

    Example:
        ```python
        class MyAgentSettings(AgentSettings):
            reasoning_model: str              # required
            fast_model: str                   # required
            max_retries: int = 3              # optional with default
            temperature: float = 0.7          # optional with default
        ```
    """

    model_config = ConfigDict(
        extra="ignore",
        validate_default=True,
    )

    trace_id: str | None = None
    """Optional trace ID for distributed tracing."""

    execution_id: str | None = None
    """Optional unique identifier for this execution."""

    task_defaults: dict[str, str] | None = None
    """Optional per-tool task type defaults (tool_name -> "quick"|"background")."""

    agent_timeout: int | None = None
    """Optional timeout override (seconds) for delegation targets.
    If set, overrides a delegation target's configured timeout for all
    targets. Typically supplied by whatever runtime configuration source
    the agent is wired to (see `sofias_sdk_lite.config.source.ConfigSource`).
    """


SettingsT = TypeVar("SettingsT", bound=AgentSettings)

# ContextVar to store the current settings in async execution context
_current_settings: ContextVar[AgentSettings | None] = ContextVar(
    "current_settings", default=None
)


class SettingsResolver(Generic[SettingsT]):
    """Resolves and manages runtime settings for agent execution.

    This class validates raw runtime data against the agent's AgentSettings
    subclass and manages the settings in a ContextVar for async access.

    The ContextVar pattern ensures settings are available throughout the
    graph execution without explicit parameter passing.

    Example:
        ```python
        resolver = SettingsResolver(MyAgentSettings)

        # Resolve from incoming message
        settings = resolver.resolve({"reasoning_model": "gpt-4o", "fast_model": "gpt-4o-mini"})

        # Set in context for async access
        resolver.set_current(settings)

        # Access from anywhere in the execution
        current = resolver.get_current()

        # Clear after execution
        resolver.clear()
        ```
    """

    def __init__(self, settings_class: type[SettingsT]) -> None:
        """Initialize the resolver with the agent's settings class.

        Args:
            settings_class: The AgentSettings subclass to validate against.
        """
        self._settings_class = settings_class

    @property
    def settings_class(self) -> type[SettingsT]:
        """The AgentSettings subclass used for validation."""
        return self._settings_class

    def resolve(self, data: dict[str, Any]) -> SettingsT:
        """Validate and resolve runtime data into settings.

        Args:
            data: Raw configuration dictionary (typically from an incoming
                task message or a `ConfigSource`).

        Returns:
            Validated settings instance.

        Raises:
            RuntimeConfigError: If validation fails (missing required fields,
                wrong types, etc.).
        """
        try:
            return self._settings_class.model_validate(data)
        except ValidationError as e:
            raise RuntimeConfigError(
                f"Runtime configuration validation failed for '{self._settings_class.__name__}'",
                settings_class=self._settings_class.__name__,
                validation_errors=e.errors(),
            ) from e

    def set_current(self, settings: SettingsT) -> None:
        """Store settings in the current execution context.

        Args:
            settings: The resolved settings to store.
        """
        _current_settings.set(settings)

    def get_current(self) -> SettingsT | None:
        """Retrieve settings from the current execution context.

        Returns:
            The current settings, or None if not set.
        """
        settings = _current_settings.get()
        if settings is not None and isinstance(settings, self._settings_class):
            return settings
        return None

    def get_current_or_raise(self) -> SettingsT:
        """Retrieve settings from context, raising if not available.

        Returns:
            The current settings.

        Raises:
            RuntimeConfigError: If settings are not set in the current context.
        """
        settings = self.get_current()
        if settings is None:
            raise RuntimeConfigError(
                f"No settings found in context for '{self._settings_class.__name__}'",
                settings_class=self._settings_class.__name__,
            )
        return settings

    def clear(self) -> None:
        """Clear settings from the current execution context."""
        _current_settings.set(None)

    def resolve_and_set(self, data: dict[str, Any]) -> SettingsT:
        """Convenience method to resolve and set settings in one call.

        Args:
            data: Raw configuration dictionary.

        Returns:
            Validated settings instance (also stored in context).

        Raises:
            RuntimeConfigError: If validation fails.
        """
        settings = self.resolve(data)
        self.set_current(settings)
        return settings


__all__ = [
    "AgentSettings",
    "RuntimeConfigError",
    "SettingsResolver",
    "SettingsT",
]

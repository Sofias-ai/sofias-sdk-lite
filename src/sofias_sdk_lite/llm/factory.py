"""Build an LLM client without the agent author having to think about it.

Resolution order for `create_llm`:

1. An explicit `LLMGatewayConfig` (or keyword overrides).
2. The agent's runtime settings (`BaseAgentSettings.model_name`,
   ``router_url``, ``router_api_key``), which the platform injects per task.
3. Environment variables: ``SOFIAS_LLM_MODEL``, ``SOFIAS_LLM_BASE_URL``,
   ``SOFIAS_LLM_API_KEY``, ``SOFIAS_LLM_PROVIDER`` (what the Sofias platform
   injects into every agent container); then ``MODEL_NAME`` / ``ROUTER_URL`` /
   ``ROUTER_API_KEY``, the same keys the agent settings use.

`AgentBuilder.build()` calls `resolve_default_llm` when no `with_llm()` was
given, and `AgentRunner` scopes the per-task settings-derived client with
`default_llm`, so an agent module never has to construct a client itself.
"""

from __future__ import annotations

import os
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, SecretStr

from sofias_sdk_lite.errors.exceptions import LLMConfigurationError
from sofias_sdk_lite.llm.gateway import PROVIDER_GATEWAY, GatewayLLM
from sofias_sdk_lite.llm.openai_compatible import OpenAICompatibleLLM
from sofias_sdk_lite.llm.protocol import LLMCallable, StreamableLLMCallable
from sofias_sdk_lite.observability._log import get_logger

logger = get_logger("llm.factory")

__all__ = [
    "ENV_LLM_API_KEY",
    "ENV_LLM_BASE_URL",
    "ENV_LLM_MODEL",
    "ENV_LLM_PROVIDER",
    "LLMGatewayConfig",
    "create_llm",
    "default_llm",
    "gateway_config_from_env",
    "gateway_config_from_settings",
    "get_default_llm",
    "resolve_default_llm",
]

ENV_LLM_MODEL: Final[str] = "SOFIAS_LLM_MODEL"
ENV_LLM_BASE_URL: Final[str] = "SOFIAS_LLM_BASE_URL"
ENV_LLM_API_KEY: Final[str] = "SOFIAS_LLM_API_KEY"
ENV_LLM_PROVIDER: Final[str] = "SOFIAS_LLM_PROVIDER"

# Fallback names, tried after the SOFIAS_LLM_* variables: the same keys the
# agent settings use, so a container that already exports them needs no renaming.
_ENV_FALLBACK_MODEL: Final[tuple[str, ...]] = ("MODEL_NAME",)
_ENV_FALLBACK_BASE_URL: Final[tuple[str, ...]] = ("ROUTER_URL",)
_ENV_FALLBACK_API_KEY: Final[tuple[str, ...]] = ("ROUTER_API_KEY",)

_OPENAI_COMPATIBLE_PROVIDERS: Final[frozenset[str]] = frozenset(
    {"openai", "openai-compatible", "generic", "litellm", "openrouter", "vllm"}
)


class LLMGatewayConfig(BaseModel):
    """Everything needed to build an LLM client.

    ``model`` names the model; ``base_url`` must be set before `create_llm`
    can build a client (the platform injects it). ``provider`` defaults to
    ``"sofias"``, the Sofias gateway.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    """Model identifier or gateway alias (e.g. ``"default"``)."""

    base_url: str = ""
    """OpenAI-compatible API root. Injected by the platform; required to build a client."""

    api_key: SecretStr = SecretStr("")
    """Bearer token. Empty sends no ``Authorization`` header."""

    provider: str = PROVIDER_GATEWAY
    """``"sofias"`` or any OpenAI-compatible label (``"openai"``, ``"litellm"``, ...)."""

    temperature: float | None = None
    """Sampling temperature; ``None`` keeps the adapter default."""

    max_tokens: int | None = None
    """Output budget; ``None`` keeps the adapter default."""

    timeout: float = 120.0
    """Per-request timeout in seconds."""

    max_retries: int = 3
    """Retries on transient failures."""

    reasoning_effort: str | None = None
    """Sofias gateway only: overrides the per-model-family default."""

    def with_overrides(self, **overrides: Any) -> LLMGatewayConfig:
        """Return a copy with the non-``None`` overrides applied."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        if "api_key" in clean and isinstance(clean["api_key"], str):
            clean["api_key"] = SecretStr(clean["api_key"])
        return self.model_copy(update=clean) if clean else self


def _first_env(env: Mapping[str, str], primary: str, fallbacks: tuple[str, ...]) -> str:
    for name in (primary, *fallbacks):
        value = env.get(name)
        if value:
            return value
    return ""


def gateway_config_from_env(environ: Mapping[str, str] | None = None) -> LLMGatewayConfig | None:
    """Read gateway configuration from the environment; ``None`` if no model is set.

    ``SOFIAS_LLM_*`` wins, then ``MODEL_NAME`` / ``ROUTER_URL`` /
    ``ROUTER_API_KEY``.
    """
    env = os.environ if environ is None else environ
    model = _first_env(env, ENV_LLM_MODEL, _ENV_FALLBACK_MODEL)
    if not model:
        return None
    base_url = _first_env(env, ENV_LLM_BASE_URL, _ENV_FALLBACK_BASE_URL)
    api_key = _first_env(env, ENV_LLM_API_KEY, _ENV_FALLBACK_API_KEY)
    provider = env.get(ENV_LLM_PROVIDER) or PROVIDER_GATEWAY
    return LLMGatewayConfig(
        model=model, base_url=base_url, api_key=SecretStr(api_key), provider=provider
    )


def gateway_config_from_settings(settings: Any) -> LLMGatewayConfig | None:
    """Read gateway configuration from agent settings; ``None`` if no model is set.

    Duck-typed on purpose: any object exposing ``model_name`` (and optionally
    ``router_url`` / ``router_api_key`` / ``router_provider``) works, so
    custom settings classes that predate `BaseAgentSettings` gaining these
    fields still resolve.
    """
    model = getattr(settings, "model_name", "") or ""
    if not model:
        return None
    base_url = getattr(settings, "router_url", "") or ""
    raw_key = getattr(settings, "router_api_key", "")
    api_key = raw_key.get_secret_value() if isinstance(raw_key, SecretStr) else str(raw_key or "")
    provider = getattr(settings, "router_provider", "") or PROVIDER_GATEWAY
    return LLMGatewayConfig(
        model=model, base_url=base_url, api_key=SecretStr(api_key), provider=provider
    )


def _resolve_config(
    source: LLMGatewayConfig | Any | None, overrides: dict[str, Any]
) -> LLMGatewayConfig:
    if isinstance(source, LLMGatewayConfig):
        config: LLMGatewayConfig | None = source
    elif source is not None:
        config = gateway_config_from_settings(source)
    else:
        config = None

    if config is None and overrides.get("model"):
        config = LLMGatewayConfig(model=str(overrides["model"]))
    if config is None:
        config = gateway_config_from_env()
    if config is None:
        raise LLMConfigurationError(
            "No LLM gateway configuration found. Pass model/base_url/api_key to "
            "create_llm(), set model_name/router_url/router_api_key on your agent "
            f"settings, or export {ENV_LLM_MODEL} (plus {ENV_LLM_BASE_URL} and "
            f"{ENV_LLM_API_KEY}).",
            missing=["model"],
        )
    return config.with_overrides(**overrides)


def create_llm(
    source: LLMGatewayConfig | Any | None = None,
    /,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    reasoning_effort: str | None = None,
    **adapter_kwargs: Any,
) -> StreamableLLMCallable:
    """Build a ready-to-use LLM client.

    Args:
        source: An `LLMGatewayConfig`, an agent settings object exposing
            ``model_name`` / ``router_url`` / ``router_api_key``, or ``None``
            to fall back to keyword arguments and then the environment.
        model: Overrides the resolved model identifier or gateway alias.
        base_url: Overrides the resolved gateway URL.
        api_key: Overrides the resolved bearer token.
        provider: Overrides the adapter flavour (``"sofias"`` or an
            OpenAI-compatible label).
        temperature: Overrides the sampling temperature.
        max_tokens: Overrides the output budget.
        timeout: Overrides the per-request timeout in seconds.
        max_retries: Overrides the number of retries on transient failures.
        reasoning_effort: Overrides the per-model-family default (Sofias
            gateway only).
        **adapter_kwargs: Forwarded to the adapter constructor (e.g.
            ``transport=`` for tests, ``default_headers=``).

    Raises:
        LLMConfigurationError: If no model or gateway URL can be resolved, or
            the provider is unknown.
    """
    config = _resolve_config(
        source,
        {
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "provider": provider,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "max_retries": max_retries,
            "reasoning_effort": reasoning_effort,
        },
    )
    if not config.base_url.strip():
        raise LLMConfigurationError(
            f"No LLM gateway URL configured for model '{config.model}'. On the Sofias "
            f"platform {ENV_LLM_BASE_URL} is injected automatically; to run locally, "
            "export it or install a fake LLM with default_llm(...).",
            missing=["base_url"],
        )

    if not config.api_key.get_secret_value():
        logger.warning(
            "LLM gateway configured without an API key; requests will be unauthenticated",
            provider=config.provider,
            base_url=config.base_url,
        )

    common: dict[str, Any] = {
        "api_key": config.api_key.get_secret_value(),
        "base_url": config.base_url,
        "model": config.model,
        "timeout": config.timeout,
        "max_retries": config.max_retries,
        "max_tokens": config.max_tokens,
        **adapter_kwargs,
    }
    if config.temperature is not None:
        common["temperature"] = config.temperature

    provider_name = config.provider.lower()
    if provider_name == PROVIDER_GATEWAY:
        return GatewayLLM(reasoning_effort=config.reasoning_effort, **common)
    if provider_name in _OPENAI_COMPATIBLE_PROVIDERS:
        return OpenAICompatibleLLM(provider=provider_name, **common)
    supported = ", ".join(sorted({PROVIDER_GATEWAY, *_OPENAI_COMPATIBLE_PROVIDERS}))
    raise LLMConfigurationError(
        f"Unknown LLM provider '{config.provider}'. Supported: {supported}",
        missing=["provider"],
    )


# --------------------------------------------------------------------------
# Default LLM in the current execution context
# --------------------------------------------------------------------------

_default_llm: ContextVar[LLMCallable | None] = ContextVar("sofias_default_llm", default=None)


def get_default_llm() -> LLMCallable | None:
    """The LLM installed by the innermost active `default_llm` scope, if any."""
    return _default_llm.get()


@contextmanager
def default_llm(llm: LLMCallable | None) -> Generator[LLMCallable | None, None, None]:
    """Make ``llm`` the default for `AgentBuilder.build()` calls inside the block.

    Passing ``None`` leaves any outer default untouched.
    """
    if llm is None:
        yield get_default_llm()
        return
    token = _default_llm.set(llm)
    try:
        yield llm
    finally:
        _default_llm.reset(token)


def resolve_default_llm() -> LLMCallable | None:
    """Return the context default LLM, else one built from the environment, else ``None``."""
    current = get_default_llm()
    if current is not None:
        return current
    config = gateway_config_from_env()
    if config is None:
        return None
    return create_llm(config)

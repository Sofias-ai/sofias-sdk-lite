"""Sofias gateway client.

The Sofias platform gives every agent access to its models through an
OpenAI-compatible gateway. This adapter is `OpenAICompatibleLLM` with the
gateway's defaults baked in:

- a per-model-family ``reasoning_effort`` so thinking channels are switched
  off (or minimised) unless the caller asks otherwise;
- a generous ``max_tokens`` budget, because structured output is paid out
  of that budget and a truncated JSON document does not parse.

There is no default address: when the agent runs on the platform the gateway
URL is injected for it (``SOFIAS_LLM_BASE_URL``), and to run locally against a
real gateway you set it yourself.
"""

from __future__ import annotations

from typing import Any, Final
from urllib.parse import urlsplit

import httpx

from sofias_sdk_lite.errors.exceptions import LLMConfigurationError
from sofias_sdk_lite.llm.openai_compatible import OpenAICompatibleLLM

__all__ = [
    "DEFAULT_GATEWAY_MAX_TOKENS",
    "GatewayLLM",
    "PROVIDER_GATEWAY",
    "normalize_gateway_base_url",
]

PROVIDER_GATEWAY: Final[str] = "sofias"
"""Provider label that selects this adapter in `create_llm`."""

DEFAULT_GATEWAY_MAX_TOKENS: Final[int] = 16_384
"""Output budget sent when neither the constructor nor the caller sets one.

A single budget on purpose: splitting it by reasoning mode produced a family
of truncation bugs in structured-output nodes. Only generated tokens are
billed, so a generous cap costs nothing on short completions.
"""

# Per-family default ``reasoning_effort`` when the caller passes none.
# ``"none"`` turns the thinking channel off, ``"low"`` keeps it minimal.
# - gpt-oss reasons by default and "none" is not accepted everywhere -> "low".
# - deepseek: "none" is not accepted everywhere -> "low".
# - qwen: thinking behaves inconsistently unless disabled -> "none" explicitly.
# - glm: "none" makes the model emit its monologue inside content -> "low".
_REASONING_EFFORT_BY_FAMILY: Final[dict[str, str]] = {
    "gpt-oss": "low",
    "deepseek": "low",
    "qwen": "none",
    "glm": "low",
}

# Fallback for models outside the table, including the ``"default"`` gateway
# alias most agents send: ``"low"`` is accepted by every model family.
_REASONING_EFFORT_FALLBACK: Final[str] = "low"


def normalize_gateway_base_url(base_url: str) -> str:
    """Return the OpenAI-compatible API root for a gateway host.

    The gateway can be configured either as a bare host
    (``https://gateway.example``) or as the API root
    (``https://gateway.example/v1``). The OpenAI surface lives under ``/v1``,
    so a URL with no path gets it appended; any URL that already has a path is
    trusted as-is.

    Raises:
        LLMConfigurationError: If ``base_url`` is empty.
    """
    url = (base_url or "").strip().rstrip("/")
    if not url:
        raise LLMConfigurationError(
            "No LLM gateway URL configured. On the Sofias platform "
            "SOFIAS_LLM_BASE_URL is injected automatically; to run locally, "
            "export it or install a fake LLM with default_llm(...).",
            missing=["base_url"],
        )
    if urlsplit(url).path in ("", "/"):
        return f"{url}/v1"
    return url


class GatewayLLM(OpenAICompatibleLLM):
    """`StreamableLLMCallable` for the Sofias gateway.

    Args:
        api_key: Gateway API key (empty string sends no ``Authorization``).
        model: Model identifier or gateway alias (e.g. ``"default"``).
        base_url: Gateway host or API root. Required; a bare host gets
            ``/v1`` appended (see `normalize_gateway_base_url`).
        temperature: Sampling temperature.
        max_retries: Retries on transient failures.
        timeout: Per-request timeout in seconds.
        reasoning_effort: ``None`` applies the per-family default; a non-empty
            string is sent as-is on every request; ``""`` disables injection
            so the model default applies.
        max_tokens: Output budget; ``None`` applies `DEFAULT_GATEWAY_MAX_TOKENS`.
        default_headers: Extra headers sent on every request.
        transport: Optional ``httpx`` transport (for tests).

    Raises:
        LLMConfigurationError: If ``base_url`` is empty.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        temperature: float = 0.1,
        max_retries: int = 3,
        timeout: float = 120.0,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        default_headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=normalize_gateway_base_url(base_url),
            model=model,
            provider=PROVIDER_GATEWAY,
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
            max_tokens=max_tokens,
            default_headers=default_headers,
            transport=transport,
        )
        self._reasoning_effort = reasoning_effort

    def _default_reasoning_effort(self, model: str) -> str:
        model_lower = model.lower()
        for family, effort in _REASONING_EFFORT_BY_FAMILY.items():
            if family in model_lower:
                return effort
        return _REASONING_EFFORT_FALLBACK

    def _augment_request_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Inject gateway defaults; values already present in the request win."""
        if self._reasoning_effort is None:
            model = kwargs.get("model") or self._model
            kwargs.setdefault("reasoning_effort", self._default_reasoning_effort(str(model)))
        elif self._reasoning_effort:
            kwargs.setdefault("reasoning_effort", self._reasoning_effort)
        kwargs.setdefault("max_tokens", DEFAULT_GATEWAY_MAX_TOKENS)

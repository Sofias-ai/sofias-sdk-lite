"""Base class for agent runtime settings.

`BaseAgentSettings` carries the platform's LLM gateway keys (``model_name``,
``router_url``, ``router_api_key``) so `AgentRunner` can build the LLM
client for each task without the agent author wiring anything. Every other
concern (embeddings, vector stores, object storage, search backends) is left
to the concrete agent, which subclasses and declares what it needs:

    class MyAgentSettings(BaseAgentSettings):
        custom_field: str = "default"

The gateway fields are optional: an agent made only of function nodes never
touches an LLM, and `AgentBuilder` falls back to the environment
(``SOFIAS_LLM_MODEL`` & co.) when settings do not name a model.
"""

from __future__ import annotations

import difflib
from typing import Any, ClassVar

from pydantic import ConfigDict, SecretStr, model_validator

from sofias_sdk_lite.config.runtime_config import AgentSettings
from sofias_sdk_lite.observability._log import get_logger

logger = get_logger("config.agent_settings")


class BaseAgentSettings(AgentSettings):
    """Runtime settings shared by all agents.

    Carries the LLM gateway keys the Sofias platform injects per task
    (``model_name``, ``router_url``, ``router_api_key``). `AgentRunner` reads
    them through `sofias_sdk_lite.llm.create_llm` and installs the resulting
    client as the default for `AgentBuilder.build()`, so agent code only
    declares LLM nodes and never constructs a client.

    All three are optional here so function-only agents validate with an
    empty payload; `has_llm_gateway` tells whether a model was supplied.

    Subclass to add agent-specific fields — feature flags, downstream
    service URLs, or anything else a given agent depends on.
    """

    model_config = ConfigDict(extra="ignore", validate_default=True)

    CREDENTIAL_FIELDS: ClassVar[frozenset[str]] = frozenset({"router_api_key"})
    """Names of fields that hold credentials (API keys, tokens, secrets).

    Extend on your subclass (``BaseAgentSettings.CREDENTIAL_FIELDS | {...}``),
    typing the fields as ``pydantic.SecretStr`` so ``repr``/``model_dump``
    never leak the raw value. The SDK excludes these fields wherever settings
    are dumped into a dict that reaches an LLM prompt (``LLMNode`` template
    variables), so a ``{api_key}`` placeholder can never inline a secret. Read
    them at the point of use with ``.get_secret_value()``.

    Example:
        ```python
        class MySettings(BaseAgentSettings):
            CREDENTIAL_FIELDS = BaseAgentSettings.CREDENTIAL_FIELDS | {"search_api_key"}

            search_api_key: SecretStr = SecretStr("")
        ```
    """

    # --- LLM gateway (canonical platform keys) ---
    model_name: str = ""
    """Model identifier or gateway alias. Empty means "no LLM configured here"."""

    router_url: str = ""
    """OpenAI-compatible gateway root, injected by the platform. Required to build an LLM."""

    router_api_key: SecretStr = SecretStr("")
    """Gateway bearer token. Empty sends unauthenticated requests."""

    router_provider: str = ""
    """Adapter flavour: ``"sofias"`` (default when empty) or an OpenAI-compatible label."""

    @property
    def has_llm_gateway(self) -> bool:
        """Whether these settings name a model (and therefore can build an LLM)."""
        return bool(self.model_name)

    @model_validator(mode="before")
    @classmethod
    def _warn_on_typo_keys(cls, data: Any) -> Any:
        """Emit a warning for unknown keys that look like typos of declared fields.

        `extra="ignore"` silently drops misspelled keys (e.g. `mdoel_name`
        instead of `model_name`), which can lead to confusing "missing
        configuration" errors later on. `extra="ignore"` is kept for
        forward-compatibility with unrelated payload keys, but a warning is
        logged whenever an unknown key is suspiciously close to a field
        declared on the concrete subclass.
        """
        if not isinstance(data, dict):
            return data
        known = set(cls.model_fields.keys())
        for key in data:
            if not isinstance(key, str) or key in known:
                continue
            close = difflib.get_close_matches(key, known, n=1, cutoff=0.85)
            if close:
                logger.warning(
                    "Unknown settings key is being ignored; possible typo",
                    key=key,
                    suggestion=close[0],
                )
        return data


__all__ = ["BaseAgentSettings"]

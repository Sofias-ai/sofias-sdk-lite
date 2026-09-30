"""LLM protocol, gateway clients, and the zero-config factory.

Two layers:

- **Protocol** (`LLMCallable`, `StreamableLLMCallable`, `LLMResponse`,
  `TokenUsage`, `ToolSpec`, ...): the contract `LLMNode` drives. Wrap any
  client in it if you need something the bundled adapters do not cover.
- **Batteries** (`create_llm`, `GatewayLLM`, `OpenAICompatibleLLM`,
  `LLMGatewayConfig`): a ready-made client for the Sofias gateway (or
  any OpenAI-compatible endpoint) resolved from settings or the environment,
  so agent code never constructs an LLM client by hand.
"""

from __future__ import annotations

from sofias_sdk_lite.llm.gateway import (
    DEFAULT_GATEWAY_MAX_TOKENS,
    PROVIDER_GATEWAY,
    GatewayLLM,
    normalize_gateway_base_url,
)
from sofias_sdk_lite.llm.factory import (
    ENV_LLM_API_KEY,
    ENV_LLM_BASE_URL,
    ENV_LLM_MODEL,
    ENV_LLM_PROVIDER,
    LLMGatewayConfig,
    create_llm,
    default_llm,
    gateway_config_from_env,
    gateway_config_from_settings,
    get_default_llm,
    resolve_default_llm,
)
from sofias_sdk_lite.llm.openai_compatible import OpenAICompatibleLLM
from sofias_sdk_lite.llm.protocol import (
    LLMCallable,
    LLMResponse,
    LLMStreamEvent,
    StreamableLLMCallable,
)
from sofias_sdk_lite.llm.tokens import TokenUsage
from sofias_sdk_lite.llm.tools import ToolCall, ToolSpec

__all__ = [
    # Protocol
    "LLMCallable",
    "LLMResponse",
    "LLMStreamEvent",
    "StreamableLLMCallable",
    "TokenUsage",
    "ToolCall",
    "ToolSpec",
    # Clients
    "GatewayLLM",
    "OpenAICompatibleLLM",
    "DEFAULT_GATEWAY_MAX_TOKENS",
    "PROVIDER_GATEWAY",
    "normalize_gateway_base_url",
    # Factory
    "LLMGatewayConfig",
    "create_llm",
    "default_llm",
    "get_default_llm",
    "resolve_default_llm",
    "gateway_config_from_env",
    "gateway_config_from_settings",
    "ENV_LLM_MODEL",
    "ENV_LLM_BASE_URL",
    "ENV_LLM_API_KEY",
    "ENV_LLM_PROVIDER",
]

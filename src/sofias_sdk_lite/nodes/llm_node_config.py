"""Configuration for LLM and Function nodes.

This module defines configuration structures for nodes that use an LLM
(LLMNodeConfig) and deterministic function nodes (FunctionNodeConfig).
NodeRetryConfig is also defined here as it is shared with delegation configs.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from sofias_sdk_lite.contracts import StrictContract

__all__ = [
    "FunctionNodeConfig",
    "LLMNodeConfig",
    "LLMSettings",
    "NodePromptConfig",
    "NodeRetryConfig",
    "NodeToolLoopConfig",
]


def _dereference_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline all $ref pointers and add additionalProperties: false.

    Produces a self-contained JSON schema suitable for provider-level
    structured output (e.g. OpenAI's response_format).
    """
    defs = schema.get("$defs", {})

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_name = node["$ref"].rsplit("/", 1)[-1]
                return _resolve(defs[ref_name])
            resolved = {k: _resolve(v) for k, v in node.items() if k != "$defs"}
            if resolved.get("type") == "object" and "additionalProperties" not in resolved:
                resolved["additionalProperties"] = False
            return resolved
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    result = _resolve(schema)
    result.pop("$defs", None)
    return result


class LLMSettings(BaseModel):
    """LLM settings for inference calls.

    All fields are optional. When None, values are resolved from
    the agent-level config, then SDK-level config, then defaults.
    """

    model_config = ConfigDict(frozen=True)

    model: str | None = None
    """LLM model identifier. None means inherit from parent config."""

    temperature: float | None = None
    """Temperature for LLM calls. None means inherit from parent config."""

    max_tokens: int | None = None
    """Maximum tokens in LLM response. None means inherit from parent config."""

    top_p: float | None = None
    """Top-p (nucleus sampling) value. None means inherit from parent config."""


class NodeToolLoopConfig(BaseModel):
    """Tool loop configuration specific to a node.

    All fields are optional. When None, values are resolved from
    the agent-level config, then SDK-level config, then defaults.
    """

    model_config = ConfigDict(frozen=True)

    max_iterations: int | None = None
    """Maximum tool loop iterations. None means inherit from parent config."""

    iteration_timeout_seconds: float | None = None
    """Timeout per iteration in seconds. None means inherit from parent config."""

    parallel_tool_calls: bool | None = None
    """Whether the LLM can request multiple tool calls in a single response.
    None uses the provider's default behavior. Requires an LLM client
    (an `LLMCallable` implementation) that supports parallel tool calls."""

    max_tool_call_repeats: int = 3
    """Maximum times the same tool call (same name + same arguments) can be
    executed within a single node run. After this limit, the call is skipped
    and the LLM receives a message indicating the query was already attempted."""

    max_tool_loop_tokens: int | None = None
    """Token budget for the whole request of a tool-loop iteration (fixed prefix
    plus accumulated history). When the estimate exceeds ``budget - reserve``,
    the node first tries to consolidate the oldest iterations into a compact
    working memory; if that does not free enough room, the loop stops and makes
    a graceful final call. ``None`` inherits the SDK default (128k)."""

    tool_loop_token_reserve: int | None = None
    """Tokens reserved within the budget for the final LLM response.
    ``None`` inherits the SDK default (16 384)."""


class NodeRetryConfig(BaseModel):
    """Retry configuration specific to a node.

    All fields are optional. When None, values are resolved from
    the agent-level config, then SDK-level config, then defaults.
    """

    model_config = ConfigDict(frozen=True)

    max_retries: int | None = None
    """Maximum retry attempts. None means inherit from parent config."""

    delay_seconds: float | None = None
    """Initial retry delay in seconds. None means inherit from parent config."""

    backoff_multiplier: float | None = None
    """Exponential backoff multiplier. None means inherit from parent config."""

    max_delay_seconds: float | None = None
    """Maximum retry delay in seconds. None means inherit from parent config."""


class NodePromptConfig(BaseModel):
    """Prompt configuration for a node."""

    model_config = ConfigDict(frozen=True)

    prompt: str | None = None
    """Fixed prompt content as a string."""

    prompt_path: Path | None = None
    """Path to a prompt file. Mutually exclusive with prompt."""

    template_variables: dict[str, Any] | None = None
    """Variables to be substituted into the prompt template at runtime."""

    mcp_config_path: Path | None = None
    """Path to a JSON file describing available MCP servers for this node.
    Purely descriptive metadata used to render a prompt section; the SDK
    does not perform MCP discovery or connect to any MCP server itself."""


class LLMNodeConfig(BaseModel):
    """Configuration for an LLM node in the agent graph.

    This defines all configurable aspects of a node that uses an LLM.
    Fields that participate in the three-level resolution (SDK > agent > node)
    are optional and default to None, meaning "inherit from parent level".

    Example:
        ```python
        config = LLMNodeConfig(
            name="classifier",
            description="Classifies incoming emails",
            input_contract=EmailInput,
            output_contract=ClassificationOutput,
            llm=LLMSettings(model="gpt-4o-mini"),  # Override model for this node
            prompt=NodePromptConfig(
                prompt_path=Path("prompts/classifier.md"),
                template_variables={"categories": ["spam", "urgent", "normal"]},
            ),
        )
        ```
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    name: str
    """Unique name identifying this node within the agent."""

    description: str | None = None
    """Human-readable description of what this node does."""

    # Contracts
    input_contract: type[StrictContract]
    """Pydantic model class defining the expected input schema.
    Accepts InputContract, DelegationRequest, or any StrictContract subclass."""

    output_contract: type[StrictContract]
    """Pydantic model class defining the expected output schema.
    Accepts OutputContract, DelegationResponse, or any StrictContract subclass."""

    # LLM configuration (optional, inherits from agent/SDK if not set)
    llm: LLMSettings | None = None
    """LLM settings for this node. None means use agent/SDK defaults."""

    # Tool loop configuration (optional, inherits from agent/SDK if not set)
    tool_loop: NodeToolLoopConfig | None = None
    """Tool loop settings for this node. None means use agent/SDK defaults."""

    # Retry configuration (optional, inherits from agent/SDK if not set)
    retry: NodeRetryConfig | None = None
    """Retry settings for this node. None means use agent/SDK defaults."""

    # Prompt configuration
    prompt: NodePromptConfig | None = None
    """Prompt configuration for this node."""

    # Raw text mode
    raw_text_field: str | None = None
    """When set, the LLM responds in plain text (no JSON schema in prompt).
    The raw text is wrapped as {raw_text_field: content} before output validation.
    The output_contract must have a field matching this name of type str."""

    # Node-level system prompt
    system_prompt: str | None = None
    """System prompt for this node. Overrides the agent-level system prompt when set."""

    # JSON parse retries
    max_parse_retries: int = 2
    """Maximum attempts to correct malformed JSON by feeding the error back to the LLM."""

    # Structured output
    response_format: dict[str, Any] | None = None
    """Provider-level response_format passed to the LLM invoke call.
    Example: {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}
    When set, the provider enforces the output schema, reducing JSON parse retries.
    Auto-derived from output_contract when not provided and raw_text_field is None."""

    # Pipeline passthrough
    output_field: str | None = None
    """When set, the LLM parsed output is stored under this key and merged with
    the original input_data, preserving pipeline context for downstream nodes.
    When None (default), the LLM output replaces current_data entirely."""

    @model_validator(mode="before")
    @classmethod
    def _derive_response_format(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("response_format") is None and data.get("raw_text_field") is None:
                contract = data.get("output_contract")
                if contract is not None and hasattr(contract, "model_json_schema"):
                    schema = _dereference_schema(contract.model_json_schema())
                    data["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": contract.__name__.lower(),
                            "schema": schema,
                        },
                    }
        return data


class FunctionNodeConfig(BaseModel):
    """Configuration for a deterministic FunctionNode.

    This is a simplified configuration for nodes that execute Python logic
    without an LLM. No prompt, tool loop, or LLM settings are needed.

    Example:
        ```python
        config = FunctionNodeConfig(
            name="data_merger",
            description="Merges outputs from multiple upstream nodes",
            retry_policy=RetryPolicy(max_retries=2),
        )
        ```
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    name: str
    """Unique name identifying this node within the agent."""

    description: str | None = None
    """Human-readable description of what this node does."""

    # Retry configuration (optional)
    retry: NodeRetryConfig | None = None
    """Retry settings for this node. Useful if the function calls external APIs."""

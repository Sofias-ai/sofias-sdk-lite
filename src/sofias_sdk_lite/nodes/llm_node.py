"""LLMNode implementation for the Sofias Agents SDK.

This module provides the LLMNode class, a node type that uses an LLM for
execution. It supports tool loops, prompt assembly, and configurable hooks
for observability.

LLMCallable, LLMResponse, LLMStreamEvent, StreamableLLMCallable, ToolCall and
ToolSpec are the shared LLM protocol types and live in `sofias_sdk_lite.llm`.
ToolResult and the hook type aliases below are node-execution-loop specific
(they were originally colocated with LLMNode in the internal SDK's node
package) and are defined here.

MCP tool provisioning note: this module never imports or hardcodes any MCP
client. Tool availability is entirely caller-injected — either as a static
`tools` list or via the generic `ToolRegistry` protocol referenced under
TYPE_CHECKING below. MCP is fully optional; there is no hard dependency on
it here.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import Counter
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, ConfigDict

from sofias_sdk_lite.config import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_TOOL_LOOP_TOKENS,
    DEFAULT_TOOL_LOOP_TOKEN_RESERVE,
    AgentSettings,
    SettingsResolver,
)
from sofias_sdk_lite.errors import MaxIterationsError, NodeExecutionError
from sofias_sdk_lite.llm import (
    LLMCallable,
    LLMResponse,
    LLMStreamEvent,
    StreamableLLMCallable,
    TokenUsage,
    ToolCall,
    ToolSpec,
)
from sofias_sdk_lite.nodes.base_node import BaseNode, NodeType
from sofias_sdk_lite.nodes.prompt_assembler import PromptAssembler
from sofias_sdk_lite.observability._log import get_logger

if TYPE_CHECKING:
    from sofias_sdk_lite.contracts import NodeContract
    from sofias_sdk_lite.nodes.llm_node_config import LLMNodeConfig

    # BaseTool / ExecutionResult / ToolContext describe a static, caller-owned
    # tool implementation. ToolRegistry is a caller-injected protocol for
    # dynamic tool provisioning (any backend, MCP included, is opaque to
    # this module). Both are optional dependencies of the tool loop.
    from sofias_sdk_lite.tools import BaseTool, ExecutionResult, ToolContext, ToolRegistry

logger = get_logger("nodes.llm_node")

__all__ = [
    "LLMCallable",
    "LLMNode",
    "LLMResponse",
    "OnChunkHook",
    "OnLLMResponseHook",
    "OnToolCallHook",
    "OnToolResultHook",
    "ShouldContinueHook",
    "ToolCall",
    "ToolResult",
]


# =============================================================================
# Tool Result and Hook Types (node-execution-loop specific)
# =============================================================================


class ToolResult(BaseModel):
    """Result from executing a tool within the LLM tool loop."""

    model_config = ConfigDict(frozen=True)

    success: bool
    """Whether the tool execution succeeded."""

    output: Any = None
    """The output from the tool (if successful)."""

    error: str | None = None
    """Error message (if failed)."""

    tool_call_id: str | None = None
    """ID of the tool call this result corresponds to."""


OnToolCallHook = Callable[[str, dict[str, Any]], dict[str, Any] | None]
"""Hook called before tool execution. Returns modified input or None for original."""

OnToolResultHook = Callable[[str, dict[str, Any], Any], Any | None]
"""Hook called after tool execution. Returns modified output or None for original."""

OnLLMResponseHook = Callable[[LLMResponse, int], None]
"""Hook called after each LLM response. For observability/logging."""

ShouldContinueHook = Callable[[int, LLMResponse], bool]
"""Hook to control loop continuation. Returns False to stop early."""

OnChunkHook = Callable[[str, int], None]
"""Called for each text chunk during streaming. Args: (content, iteration)."""


# =============================================================================
# LLMNode Implementation
# =============================================================================


class LLMNode(BaseNode):
    """A node that uses an LLM for execution.

    LLMNode encapsulates:
    - Prompt assembly with template resolution
    - LLM invocation with optional tool loop
    - Configurable hooks for observability and intervention

    The node inherits input/output validation from BaseNode. Internally,
    it may execute multiple LLM <-> tool cycles before producing the final
    result.

    Example:
        ```python
        node = LLMNode(
            config=node_config,
            contract=NodeContract(input_schema=MyInput, output_schema=MyOutput),
            llm=my_llm_callable,
            tools=[tool1, tool2],
        )
        result = await node.execute({"query": "Hello"})
        ```
    """

    def __init__(
        self,
        config: LLMNodeConfig,
        contract: NodeContract,
        llm: LLMCallable,
        tools: list[BaseTool] | None = None,
        prompt_assembler: PromptAssembler | None = None,
        *,
        system_prompt: str | None = None,
        settings_resolver: SettingsResolver[AgentSettings] | None = None,
        tool_registry: ToolRegistry | None = None,
        on_tool_call: OnToolCallHook | None = None,
        on_tool_result: OnToolResultHook | None = None,
        on_llm_response: OnLLMResponseHook | None = None,
        should_continue: ShouldContinueHook | None = None,
        on_chunk: OnChunkHook | None = None,
    ) -> None:
        """Initialize the LLM node.

        Args:
            config: Node configuration (name, prompt, LLM settings, etc.).
            contract: Input/output contract for validation.
            llm: LLM callable for making completions.
            tools: Optional list of BaseTool instances available to this node.
            prompt_assembler: Custom prompt assembler. If None, one is created
                from config.
            settings_resolver: Optional resolver for runtime settings access.
            tool_registry: Optional registry for dynamic tool provisioning.
                Any backend may implement this protocol; the node has no
                knowledge of or dependency on a specific tool provider.
            on_tool_call: Hook called before each tool execution.
            on_tool_result: Hook called after each tool execution.
            on_llm_response: Hook called after each LLM response.
            should_continue: Hook to control loop continuation.
            on_chunk: Hook called for each text chunk during streaming.
        """
        super().__init__(
            name=config.name,
            contract=contract,
            description=config.description,
        )

        self._config = config
        self._llm = llm
        # Node-level system_prompt overrides agent-level one
        self._system_prompt = config.system_prompt if config.system_prompt is not None else system_prompt
        self._tools = tools or []
        self._tools_by_name = {t.config.name: t for t in self._tools}
        self._settings_resolver = settings_resolver
        self._tool_registry = tool_registry

        # Raw text mode
        self._raw_text_field = config.raw_text_field

        # Build prompt assembler if not provided.
        # A caller-supplied assembler is AUTHORITATIVE: the runtime never
        # rebuilds it with the registry's tools (see _prompt_with_registry_tools).
        # Without this distinction a caller who deliberately passes `tools=[]`
        # to keep tool schemas OUT of the prompt text (they already travel in
        # the request's `tools` field) saw them re-injected anyway, paying for
        # every schema twice per turn.
        self._owns_prompt_assembler = prompt_assembler is None
        if prompt_assembler is not None:
            self._prompt_assembler = prompt_assembler
        else:
            tool_specs = [self._build_tool_spec(t) for t in self._tools]
            self._prompt_assembler = PromptAssembler(
                config=config,
                tools=tool_specs,
                output_contract=config.output_contract,
                raw_text=self._raw_text_field is not None,
            )

        # Hooks
        self._on_tool_call = on_tool_call
        self._on_tool_result = on_tool_result
        self._on_llm_response = on_llm_response
        self._should_continue = should_continue
        self._on_chunk = on_chunk

        # Holds the final output from _stream_run() since async generators
        # cannot return values in Python
        self._last_stream_output: dict[str, Any] | None = None

    @property
    def node_type(self) -> NodeType:
        return NodeType.LLM

    @property
    def config(self) -> LLMNodeConfig:
        """The node's configuration."""
        return self._config

    def _get_max_iterations(self) -> int:
        """Resolve max iterations from config hierarchy.

        Returns:
            Maximum iterations for the tool loop.
        """
        if self._config.tool_loop and self._config.tool_loop.max_iterations is not None:
            return self._config.tool_loop.max_iterations
        return DEFAULT_MAX_ITERATIONS

    def _get_parallel_tool_calls(self) -> bool | None:
        """Resolve parallel_tool_calls from config hierarchy.

        Returns:
            Whether the LLM can request multiple tool calls per response,
            or None to use the provider's default.
        """
        if self._config.tool_loop and self._config.tool_loop.parallel_tool_calls is not None:
            return self._config.tool_loop.parallel_tool_calls
        return None

    def _get_max_tool_call_repeats(self) -> int:
        """Resolve max_tool_call_repeats from config.

        Returns:
            Maximum times the same tool call can repeat before being skipped.
        """
        if self._config.tool_loop and self._config.tool_loop.max_tool_call_repeats is not None:
            return self._config.tool_loop.max_tool_call_repeats
        return 3

    def _get_max_tool_loop_tokens(self) -> int:
        """Resolve the tool-loop token budget (whole request, prefix included)."""
        if self._config.tool_loop and self._config.tool_loop.max_tool_loop_tokens is not None:
            return self._config.tool_loop.max_tool_loop_tokens
        return DEFAULT_MAX_TOOL_LOOP_TOKENS

    def _get_tool_loop_token_reserve(self) -> int:
        """Resolve the tokens kept free for the final response."""
        if self._config.tool_loop and self._config.tool_loop.tool_loop_token_reserve is not None:
            return self._config.tool_loop.tool_loop_token_reserve
        return DEFAULT_TOOL_LOOP_TOKEN_RESERVE

    # Rough chars-per-token ratio, consistent across the Sofias ecosystem.
    _CHARS_PER_TOKEN: float = 3.2

    def _estimate_history_tokens(
        self,
        history: list[tuple[LLMResponse, list[ToolResult]]],
    ) -> int:
        """Estimate the tokens of the accumulated tool-loop history."""
        chars = 0
        for response, results in history:
            if response.tool_calls:
                for tc in response.tool_calls:
                    chars += len(json.dumps(tc.input, ensure_ascii=False))
                for r in results:
                    chars += len(str(r.output) if r.success else (r.error or ""))
            elif response.content:
                chars += len(response.content)
        return int(chars / self._CHARS_PER_TOKEN)

    def _estimate_fixed_overhead_tokens(
        self,
        prompt: str,
        tool_specs: list[ToolSpec] | None,
    ) -> int:
        """Estimate the FIXED prefix that travels with every loop iteration.

        Measuring only the history under-counts what is actually sent: each
        call carries the system prompt, the user prompt and every tool schema.
        With many tool connectors that prefix is most of the spend, so ignoring
        it makes the budget look far roomier than it is.
        """
        chars = len(prompt or "") + len(self._system_prompt or "")
        for spec in tool_specs or []:
            chars += len(spec.name or "") + len(spec.description or "")
            chars += len(json.dumps(spec.parameters_schema or {}, ensure_ascii=False))
        return int(chars / self._CHARS_PER_TOKEN)

    # -- Closing the turn -------------------------------------------------------
    #
    # All four exits (with and without streaming, by exhausting iterations and
    # by a repeated-call stall) share this text. Earlier wording claimed
    # "You have all the data you need" when work remained, which led models to
    # tell users a perfectly working connector was missing, then to invent a
    # permissions problem. Forbidding that excuse explicitly is what stops it.

    _FINAL_TURN_GUIDANCE: str = (
        "You can no longer call tools in this turn. Using everything you have "
        "gathered so far, give the user the best possible answer. "
        "If your work is INCOMPLETE, say so plainly: summarise what you did, "
        "state exactly what remains, and tell the user they can ask you to "
        "continue. "
        "Do NOT attribute this to missing tools, access, permissions or "
        "connectors: none of those is the reason you are being asked to finish "
        "now, and saying so would send the user to chase a problem that does "
        "not exist."
    )

    _FINAL_TURN_MAX_ITERATIONS: str = (
        "You have reached the maximum number of tool-call iterations allowed "
        "for this turn."
    )

    _FINAL_TURN_ALL_SKIPPED: str = (
        "Your last tool calls repeated calls already executed with the same "
        "arguments, so they were skipped and the tool loop was stopped."
    )

    def _final_turn_message(self, reason: str) -> dict[str, str]:
        """User message that closes the turn: the real reason plus shared guidance."""
        return {"role": "user", "content": f"{reason} {self._FINAL_TURN_GUIDANCE}"}

    # How many CONSECUTIVE batches with every call skipped are needed to close
    # the turn. With one, the loop cut out at the first stall: the model had
    # just received the skip notices and had no chance to react, and with
    # parallel calls a whole batch being repeats is an easy accident.
    _ALL_SKIPPED_BATCHES_BEFORE_FINAL: int = 2

    # -- Working-memory consolidation ---------------------------------------------

    # How many recent iterations are kept VERBATIM when consolidating. The most
    # recent material is what the model is reasoning over right now.
    _CONSOLIDATION_KEEP_RECENT: int = 2

    _CONSOLIDATION_MARKER: str = "[CONSOLIDATED WORKING MEMORY]"

    _CONSOLIDATION_REQUEST: str = (
        "Your context is full. Rewrite EVERYTHING above into a compact working "
        "memory for the task you are executing. This replaces the original "
        "material permanently: anything you leave out is LOST and you will not "
        "be able to ask for it again.\n"
        "Keep every identifier, name, figure, date, URL and status you still "
        "need. Keep the record of what you have already DONE, so you do not "
        "redo it. Drop only phrasing and duplication.\n"
        "Reply with the working memory itself — no preamble, no commentary."
    )

    @classmethod
    def _is_consolidated(cls, response: LLMResponse) -> bool:
        return (response.content or "").startswith(cls._CONSOLIDATION_MARKER)

    async def _consolidate_history(
        self,
        prompt: str,
        history: list[tuple[LLMResponse, list[ToolResult]]],
    ) -> list[tuple[LLMResponse, list[ToolResult]]] | None:
        """Free room for real: summarise the oldest iterations and REPLACE them.

        Nothing is truncated blindly. The model writes a working memory of the
        old material and that summary takes its place. It is irreversible on
        purpose: returning the original alongside the summary would be back to
        having no room, i.e. a loop.

        Returns ``None`` when there is nothing old to fold or the attempt fails;
        the caller then closes the turn through the honest exit.
        """
        if not history:
            return None

        # How many are kept verbatim ADAPTS to the history: if the pressure
        # comes from a single huge result, a fixed minimum of two would never
        # allow consolidating and the turn would die with the budget exhausted.
        # At least one entry is always left to fold.
        keep = max(0, min(self._CONSOLIDATION_KEEP_RECENT, len(history) - 1))
        cut = len(history) - keep
        old, recent = history[:cut], history[cut:]

        # Summarising a summary only degrades it: no room to gain there.
        if len(old) == 1 and self._is_consolidated(old[0][0]):
            return None

        messages = self._build_messages_from_history(prompt, old)
        messages.append({"role": "user", "content": self._CONSOLIDATION_REQUEST})

        try:
            summary = await self._llm.invoke(
                prompt=prompt,
                tools=None,
                messages=messages,
                system_prompt=self._system_prompt,
            )
        except Exception as exc:
            logger.warning(
                "tool_loop_consolidation_failed",
                node_name=self.name,
                iterations_folded=len(old),
                error=str(exc),
            )
            return None

        self._accumulate_usage(summary)
        text = (summary.content or "").strip()
        if not text:
            logger.warning(
                "tool_loop_consolidation_empty",
                node_name=self.name,
                iterations_folded=len(old),
            )
            return None

        logger.info(
            "tool_loop_history_consolidated",
            node_name=self.name,
            iterations_folded=len(old),
            chars_before=sum(
                len(str(r.output) if r.success else (r.error or ""))
                for _, results in old
                for r in results
            ),
            chars_after=len(text),
        )
        # The marker is for the MODEL: without it, it would read the summary as
        # its own words and might ask for the original material again.
        consolidated = LLMResponse(content=f"{self._CONSOLIDATION_MARKER}\n{text}")
        return [(consolidated, []), *recent]

    async def _fit_history_to_budget(
        self,
        prompt: str,
        tool_specs: list[ToolSpec] | None,
        history: list[tuple[LLMResponse, list[ToolResult]]],
        token_budget: int,
        token_reserve: int,
    ) -> tuple[list[tuple[LLMResponse, list[ToolResult]]], bool]:
        """Make room for the next iteration, consolidating old history if needed.

        Returns ``(history, fits)``. When ``fits`` is ``False`` the caller must
        stop the loop and close the turn through the graceful exit.
        """
        overhead = self._estimate_fixed_overhead_tokens(prompt, tool_specs)
        used = self._estimate_history_tokens(history) + overhead
        if token_budget - used >= token_reserve:
            return history, True

        # If not even the prefix fits, the missing room is NOT in the history:
        # consolidating would just burn an LLM call.
        prefix_alone_exceeds = token_budget - overhead < token_reserve
        consolidated = (
            None if prefix_alone_exceeds else await self._consolidate_history(prompt, history)
        )
        if consolidated is not None:
            history = consolidated
            used = self._estimate_history_tokens(history) + overhead
        if token_budget - used < token_reserve:
            logger.warning(
                "tool_loop_token_budget_exhausted",
                node_name=self.name,
                tokens_used=used,
                overhead_tokens=overhead,
                budget=token_budget,
                consolidated=consolidated is not None,
                prefix_alone_exceeds_budget=prefix_alone_exceeds,
            )
            return history, False
        return history, True

    @staticmethod
    def _tool_call_key(tool_call: ToolCall) -> str:
        """Create a stable, hashable key for a tool call (name + args).

        Args:
            tool_call: The tool call to fingerprint.

        Returns:
            A string key uniquely identifying this tool name + arguments combo.
        """
        args_str = json.dumps(tool_call.input, sort_keys=True, default=str)
        digest = hashlib.md5(args_str.encode()).hexdigest()  # noqa: S324
        return f"{tool_call.name}:{digest}"

    def _apply_output_field(
        self, output_data: dict[str, Any], input_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge LLM output into input_data under output_field if configured.

        When ``output_field`` is set, the parsed LLM output is stored under
        that key and the rest of ``input_data`` passes through, preserving
        pipeline context for downstream nodes.
        """
        if self._config.output_field:
            return {**input_data, self._config.output_field: output_data}
        return output_data

    def _get_runtime_vars(self) -> dict[str, Any]:
        """Get runtime variables from settings if available.

        Returns:
            Dictionary of runtime variables for prompt template resolution.
        """
        if self._settings_resolver is None:
            return {}

        settings = self._settings_resolver.get_current()
        if settings is None:
            return {}

        # Convert settings to dict for template resolution. Credential fields
        # (``BaseAgentSettings.CREDENTIAL_FIELDS``) never reach a prompt template.
        exclude: frozenset[str] = getattr(settings, "CREDENTIAL_FIELDS", frozenset())
        return settings.model_dump(exclude_none=True, exclude=set(exclude))

    def _build_tool_spec(self, tool: BaseTool) -> ToolSpec:
        """Build a ToolSpec from a BaseTool instance.

        Args:
            tool: The BaseTool to extract specification from.

        Returns:
            ToolSpec with name, description, and parameters schema.
        """
        name = tool.config.name
        # Priority: config.description > config.metadata["description"] > class docstring
        description = tool.config.description or tool.config.metadata.get("description", "")
        if not description and tool.__doc__:
            description = tool.__doc__.strip().split("\n\n")[0].strip()
        if not description:
            description = f"Tool: {name}"

        parameters_schema: dict[str, Any] = {"type": "object", "properties": {}}
        if tool.input_model is not None:
            parameters_schema = tool.input_model.model_json_schema()

        return ToolSpec(
            name=name,
            description=description,
            parameters_schema=parameters_schema,
        )

    async def _get_merged_tool_specs(self) -> list[ToolSpec] | None:
        """Combine static tool specs with registry tool specs.

        Returns:
            Merged list of ToolSpec, or None if no tools available.
        """
        static_specs = [self._build_tool_spec(t) for t in self._tools]
        registry_specs: list[ToolSpec] = []

        if self._tool_registry is not None:
            registry_specs = await self._tool_registry.get_tools_for_node(self.name)

        all_specs = static_specs + registry_specs
        return all_specs if all_specs else None

    def _tool_catalog_revision(self) -> int:
        """Cheap read of the tool catalog's state.

        Returns:
            An integer that moves when the catalog may have changed; 0 when
            the registry does not keep count (custom registries, test doubles).
        """
        rev = getattr(self._tool_registry, "revision", 0)
        return rev if isinstance(rev, int) else 0

    async def _refresh_tool_catalog(
        self,
        seen_revision: int,
        tool_specs: list[ToolSpec] | None,
        base_prompt: str,
        prompt: str,
        runtime_vars: dict[str, Any],
    ) -> tuple[int, list[ToolSpec] | None, str]:
        """Re-read the tools when the catalog changed mid-turn.

        Specs used to be resolved once, before the loop, so a tool could never
        widen the catalog for later iterations: the model never saw what it
        had just loaded. For callers that do not use this the cost is one
        integer comparison per iteration.

        Args:
            seen_revision: Revision the current specs were resolved at.
            tool_specs: Current specs.
            base_prompt: Prompt as the node's assembler produced it, before
                the registry rebuild. Re-assembling the already rebuilt one
                would duplicate the schemas inside the text.
            prompt: Current prompt.
            runtime_vars: Template variables for this turn.

        Returns:
            Updated ``(revision, specs, prompt)``, or the ones received when
            nothing changed.
        """
        revision = self._tool_catalog_revision()
        if revision == seen_revision:
            return seen_revision, tool_specs, prompt

        new_specs = await self._get_merged_tool_specs()
        logger.info(
            "Tool catalog refreshed",
            node_name=self.name,
            tools_before=len(tool_specs) if tool_specs else 0,
            tools_after=len(new_specs) if new_specs else 0,
        )
        return (
            revision,
            new_specs,
            self._prompt_with_registry_tools(base_prompt, new_specs, runtime_vars),
        )

    def _prompt_with_registry_tools(
        self,
        prompt: str,
        tool_specs: list[ToolSpec] | None,
        runtime_vars: dict[str, Any],
    ) -> str:
        """Re-assemble the prompt including the registry's tools.

        Only when the assembler is OURS. A caller-supplied assembler is
        respected as-is: it is the only way a consumer can decide that tool
        schemas are NOT repeated inside the prompt text (they already travel in
        the API's ``tools`` field). Rebuilding here doubled their token cost
        with no way to opt out.

        Args:
            prompt: Prompt already assembled with the node's assembler.
            tool_specs: Static plus registry tools, or None.
            runtime_vars: Template variables for this turn.

        Returns:
            The re-assembled prompt, or the one received when no rebuild applies.
        """
        if not self._owns_prompt_assembler:
            return prompt
        if self._tool_registry is None or not tool_specs:
            return prompt

        assembler = PromptAssembler(
            config=self._config,
            tools=tool_specs,
            output_contract=self._config.output_contract,
            raw_text=self._raw_text_field is not None,
        )
        return assembler.assemble(runtime_vars=runtime_vars)

    async def _execute_registry_tool(self, tool_call: ToolCall, tool_start: float) -> ToolResult:
        """Execute a tool call via the ToolRegistry.

        Args:
            tool_call: The tool call to execute.
            tool_start: perf_counter timestamp when tool execution started.

        Returns:
            ToolResult with success/failure and output.
        """
        # Apply on_tool_call hook
        tool_input = tool_call.input
        if self._on_tool_call is not None:
            modified_input = self._on_tool_call(tool_call.name, tool_input)
            if modified_input is not None:
                tool_input = modified_input

        try:
            output = await self._tool_registry.execute(tool_call.name, tool_input)  # type: ignore[union-attr]

            tool_duration_ms = (time.perf_counter() - tool_start) * 1000
            logger.debug(
                "Tool execution via ToolRegistry completed",
                tool_name=tool_call.name,
                node_name=self.name,
                duration_ms=round(tool_duration_ms, 2),
            )
        except Exception as e:
            tool_duration_ms = (time.perf_counter() - tool_start) * 1000
            logger.warning(
                "Tool execution via ToolRegistry failed, feeding error back to LLM",
                tool_name=tool_call.name,
                node_name=self.name,
                duration_ms=round(tool_duration_ms, 2),
                error=str(e),
            )
            return ToolResult(
                success=False,
                error=f"Tool '{tool_call.name}' execution via registry failed: {e}",
                tool_call_id=tool_call.id,
            )

        # Apply on_tool_result hook
        if self._on_tool_result is not None:
            modified_output = self._on_tool_result(tool_call.name, tool_input, output)
            if modified_output is not None:
                output = modified_output

        return ToolResult(
            success=True,
            output=output,
            tool_call_id=tool_call.id,
        )

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call.

        Args:
            tool_call: The tool call to execute.

        Returns:
            ToolResult with success/failure and output/error.
        """
        tool_start = time.perf_counter()
        tool = self._tools_by_name.get(tool_call.name)
        if tool is None:
            # Fallback to ToolRegistry if available
            if self._tool_registry is not None:
                return await self._execute_registry_tool(tool_call, tool_start)
            logger.error(
                "Tool not found",
                tool_name=tool_call.name,
                node_name=self.name,
                available_tools=list(self._tools_by_name.keys()),
            )
            return ToolResult(
                success=False,
                error=f"Tool '{tool_call.name}' not found. Available tools: {list(self._tools_by_name.keys())}",
                tool_call_id=tool_call.id,
            )

        # Apply on_tool_call hook
        tool_input = tool_call.input
        if self._on_tool_call is not None:
            modified_input = self._on_tool_call(tool_call.name, tool_input)
            if modified_input is not None:
                tool_input = modified_input

        # Execute the tool using BaseTool.execute() in a worker thread (async, non-blocking)
        try:
            context = ToolContext(
                config=tool.config,
                env=tool.env,
                payload=tool_input,
            )
            exec_result: ExecutionResult = await asyncio.to_thread(tool.execute, context)

            tool_duration_ms = (time.perf_counter() - tool_start) * 1000

            if not exec_result.success:
                logger.warning(
                    "Tool execution failed, feeding error back to LLM",
                    tool_name=tool_call.name,
                    node_name=self.name,
                    duration_ms=round(tool_duration_ms, 2),
                    error=str(exec_result.error),
                )
                return ToolResult(
                    success=False,
                    error=f"Tool '{tool_call.name}' execution failed: {exec_result.error}",
                    tool_call_id=tool_call.id,
                )

            output = exec_result.data
            logger.debug(
                "Tool execution completed",
                tool_name=tool_call.name,
                node_name=self.name,
                duration_ms=round(tool_duration_ms, 2),
            )
        except Exception as e:
            tool_duration_ms = (time.perf_counter() - tool_start) * 1000
            logger.warning(
                "Tool execution raised exception, feeding error back to LLM",
                tool_name=tool_call.name,
                node_name=self.name,
                duration_ms=round(tool_duration_ms, 2),
                error=str(e),
            )
            return ToolResult(
                success=False,
                error=f"Tool '{tool_call.name}' execution failed: {e}",
                tool_call_id=tool_call.id,
            )

        # Apply on_tool_result hook
        if self._on_tool_result is not None:
            modified_output = self._on_tool_result(tool_call.name, tool_input, output)
            if modified_output is not None:
                output = modified_output

        return ToolResult(
            success=True,
            output=output,
            tool_call_id=tool_call.id,
        )

    def _accumulate_usage(self, response: LLMResponse) -> None:
        """Add token usage from an LLM response to the ExecutionContext accumulator."""
        if response.usage is None:
            return
        from sofias_sdk_lite.agent import get_execution_context

        exec_ctx = get_execution_context()
        if exec_ctx is not None:
            model_requested = getattr(response, "model_requested", "") or ""
            provider = getattr(response, "provider", "") or ""
            exec_ctx.token_accumulator.add(
                response.usage,
                model_requested=model_requested,
                provider=provider,
            )

    async def _parse_json_with_retries(
        self,
        response: LLMResponse,
        messages: list[dict[str, Any]] | None,
        prompt: str,
        tool_specs: list[ToolSpec] | None,
        max_retries: int,
        iteration: int,
    ) -> dict[str, Any]:
        """Parse LLM JSON response, retrying with error feedback on failure.

        On JSONDecodeError, sends the error back to the LLM and asks it to
        correct the response. Avoids hard failures on minor formatting issues.

        Raises:
            NodeExecutionError: If parsing fails after all retries.
        """
        content = response.content or ""

        logger.info(
            "Parsing LLM final response",
            node_name=self.name,
            content_length=len(content),
            content_preview=content[:500] if content else "<empty>",
        )

        for parse_attempt in range(max_retries + 1):
            try:
                return json.loads(content) if content else {}
            except json.JSONDecodeError as e:
                if parse_attempt >= max_retries:
                    logger.error(
                        "LLM response not valid JSON after retries",
                        node_name=self.name,
                        parse_attempts=parse_attempt + 1,
                        error=str(e),
                        content_preview=content[:500] if content else "<empty>",
                    )
                    raise NodeExecutionError(
                        f"LLM response is not valid JSON in node '{self.name}' "
                        f"after {parse_attempt + 1} attempt(s): {e}",
                        node_name=self.name,
                        cause=e,
                    ) from e

                logger.warning(
                    "LLM response not valid JSON, requesting correction",
                    node_name=self.name,
                    parse_attempt=parse_attempt + 1,
                    error=str(e),
                )
                correction_messages = (messages or []) + [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response was not valid JSON: {e}\n"
                            "Please respond with valid JSON only, no explanation."
                        ),
                    },
                ]
                correction_response = await self._llm.invoke(
                    prompt=prompt,
                    tools=tool_specs,
                    messages=correction_messages,
                    system_prompt=self._system_prompt,
                )
                self._accumulate_usage(correction_response)
                content = correction_response.content or ""

        return {}  # unreachable, kept for type checker

    def _build_messages_from_history(
        self,
        prompt: str,
        history: list[tuple[LLMResponse, list[ToolResult]]],
    ) -> list[dict[str, Any]]:
        """Build message history for multi-turn LLM calls.

        Args:
            prompt: The system prompt.
            history: List of (llm_response, tool_results) tuples.

        Returns:
            List of messages for the LLM.
        """
        messages: list[dict[str, Any]] = []

        for response, tool_results in history:
            # Add assistant message with tool calls
            if response.tool_calls:
                messages.append({
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tc.id or f"call_{i}",
                            "name": tc.name,
                            "arguments": tc.input,
                        }
                        for i, tc in enumerate(response.tool_calls)
                    ],
                })
                # Add tool results
                for result in tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": str(result.output) if result.success else result.error,
                    })
            elif response.content:
                messages.append({
                    "role": "assistant",
                    "content": response.content,
                })

        return messages

    async def _stream_run(
        self,
        input_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[Any, None]:
        """Execute the LLM node with streaming.

        Same logic as _run() but yields StreamEvent instances as they occur.
        The final parsed output is stored in self._last_stream_output.

        Args:
            input_data: Validated input dictionary.
            context: Optional context dictionary.

        Yields:
            StreamEvent instances (TextChunkEvent, ToolCallStartEvent, etc.).

        Raises:
            MaxIterationsError: If tool loop exceeds max iterations.
            NodeExecutionError: If LLM call fails.
        """
        from sofias_sdk_lite.agent import TextChunkEvent, ToolCallResultEvent, ToolCallStartEvent

        start_time = time.perf_counter()
        self._last_stream_output = None

        runtime_vars = self._get_runtime_vars()
        runtime_vars.update(input_data)
        prompt = self._prompt_assembler.assemble(runtime_vars=runtime_vars)

        max_iterations = self._get_max_iterations()
        parallel_tool_calls = self._get_parallel_tool_calls()
        tool_specs = await self._get_merged_tool_specs()
        tool_catalog_revision = self._tool_catalog_revision()

        base_prompt = prompt
        prompt = self._prompt_with_registry_tools(prompt, tool_specs, runtime_vars)

        history: list[tuple[LLMResponse, list[ToolResult]]] = []
        iteration = 0
        tool_call_counts: Counter[str] = Counter()
        max_tool_call_repeats = self._get_max_tool_call_repeats()
        token_budget = self._get_max_tool_loop_tokens()
        token_reserve = self._get_tool_loop_token_reserve()
        all_skipped_batches = 0

        while iteration < max_iterations:
            tool_catalog_revision, tool_specs, prompt = await self._refresh_tool_catalog(
                tool_catalog_revision, tool_specs, base_prompt, prompt, runtime_vars
            )

            if token_budget and history:
                history, fits = await self._fit_history_to_budget(
                    prompt, tool_specs, history, token_budget, token_reserve
                )
                if not fits:
                    break

            # Check should_continue hook
            if iteration > 0 and history:
                last_response = history[-1][0]
                if self._should_continue is not None:
                    if not self._should_continue(iteration, last_response):
                        logger.debug(
                            "Tool loop stopped by should_continue hook",
                            node_name=self.name,
                            iteration=iteration,
                        )
                        break

            messages = self._build_messages_from_history(prompt, history) if history else None

            # Streaming path: use stream_invoke if LLM supports it
            if isinstance(self._llm, StreamableLLMCallable):
                accumulated_content = ""
                accumulated_tool_calls: list[ToolCall] = []
                accumulated_usage: TokenUsage | None = None
                stream_model_requested = ""
                stream_provider = ""

                max_stream_retries = 2
                last_stream_error: Exception | None = None

                for stream_attempt in range(max_stream_retries + 1):
                    try:
                        async for event in self._llm.stream_invoke(
                            prompt=prompt,
                            tools=tool_specs,
                            messages=messages,
                            system_prompt=self._system_prompt,
                            response_format=self._config.response_format,
                            parallel_tool_calls=parallel_tool_calls,  # type: ignore[call-arg]
                        ):
                            if event.delta:
                                accumulated_content += event.delta
                                yield TextChunkEvent(
                                    node_name=self.name,
                                    content=event.delta,
                                )
                                if self._on_chunk is not None:
                                    self._on_chunk(event.delta, iteration)
                            if event.tool_call:
                                accumulated_tool_calls.append(event.tool_call)
                            if event.usage:
                                accumulated_usage = event.usage
                            # Capture model/provider from stream events (forward-compat)
                            if not stream_model_requested:
                                stream_model_requested = getattr(event, "model_requested", "") or ""
                            if not stream_provider:
                                stream_provider = getattr(event, "provider", "") or ""
                        # Stream completed successfully
                        last_stream_error = None
                        break
                    except Exception as e:
                        last_stream_error = e
                        # Only retry if no text content was yielded to the user yet
                        # (tool-call-only streams or connection failures before any text)
                        can_retry = (
                            not accumulated_content
                            and stream_attempt < max_stream_retries
                        )
                        if can_retry:
                            logger.warning(
                                "LLM stream failed, retrying",
                                node_name=self.name,
                                iteration=iteration,
                                attempt=stream_attempt + 1,
                                max_retries=max_stream_retries,
                                error=str(e),
                            )
                            # Reset accumulators for the retry
                            accumulated_tool_calls = []
                            accumulated_usage = None
                            stream_model_requested = ""
                            stream_provider = ""
                            continue
                        else:
                            logger.error(
                                "LLM stream invocation failed",
                                node_name=self.name,
                                iteration=iteration,
                                attempt=stream_attempt + 1,
                                error=str(e),
                            )
                            raise NodeExecutionError(
                                f"LLM stream invocation failed in node '{self.name}': {e}",
                                node_name=self.name,
                                cause=e,
                            ) from e

                if last_stream_error is not None:
                    raise NodeExecutionError(
                        f"LLM stream invocation failed in node '{self.name}': {last_stream_error}",
                        node_name=self.name,
                        cause=last_stream_error,
                    ) from last_stream_error

                # Build the complete LLMResponse from accumulated data
                response = LLMResponse(
                    content=accumulated_content or None,
                    tool_calls=accumulated_tool_calls or None,
                    usage=accumulated_usage,
                    model_requested=stream_model_requested,
                    provider=stream_provider,
                )
                self._accumulate_usage(response)
            else:
                # Non-streaming fallback: invoke() and yield content as single chunk
                max_invoke_retries = 2
                last_invoke_error: Exception | None = None
                response = None  # type: ignore[assignment]
                for invoke_attempt in range(max_invoke_retries + 1):
                    try:
                        response = await self._llm.invoke(
                            prompt=prompt,
                            tools=tool_specs,
                            messages=messages,
                            system_prompt=self._system_prompt,
                            response_format=self._config.response_format,
                            parallel_tool_calls=parallel_tool_calls,  # type: ignore[call-arg]
                        )
                        last_invoke_error = None
                        break
                    except Exception as e:
                        last_invoke_error = e
                        if invoke_attempt < max_invoke_retries:
                            logger.warning(
                                "LLM invocation failed, retrying",
                                node_name=self.name,
                                iteration=iteration,
                                attempt=invoke_attempt + 1,
                                max_retries=max_invoke_retries,
                                error=str(e),
                            )
                            continue
                        logger.error(
                            "LLM invocation failed",
                            node_name=self.name,
                            iteration=iteration,
                            attempt=invoke_attempt + 1,
                            error=str(e),
                        )
                        raise NodeExecutionError(
                            f"LLM invocation failed in node '{self.name}': {e}",
                            node_name=self.name,
                            cause=e,
                        ) from e

                if last_invoke_error is not None or response is None:
                    raise NodeExecutionError(
                        f"LLM invocation failed in node '{self.name}': {last_invoke_error}",
                        node_name=self.name,
                        cause=last_invoke_error,
                    )

                self._accumulate_usage(response)

                if response.content:
                    yield TextChunkEvent(
                        node_name=self.name,
                        content=response.content,
                    )
                    if self._on_chunk is not None:
                        self._on_chunk(response.content, iteration)

            # Fire on_llm_response hook
            if self._on_llm_response is not None:
                self._on_llm_response(response, iteration)

            # Check if final response
            if response.is_final:
                if self._raw_text_field is not None:
                    output_data = {self._raw_text_field: response.content or ""}
                else:
                    messages_so_far = self._build_messages_from_history(prompt, history) if history else None
                    output_data = await self._parse_json_with_retries(
                        response=response,
                        messages=messages_so_far,
                        prompt=prompt,
                        tool_specs=tool_specs,
                        max_retries=self._config.max_parse_retries,
                        iteration=iteration,
                    )

                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.info(
                    "Node streaming execution completed",
                    node_name=self.name,
                    duration_ms=round(duration_ms, 2),
                    iterations=iteration + 1,
                    tool_calls_total=sum(
                        len(r.tool_calls or []) for r, _ in history
                    ),
                )
                self._last_stream_output = self._apply_output_field(output_data, input_data)
                return

            # Iteration trace. NAMES only: arguments carry user content and logs
            # are no place for that. This event used to be emitted only on the
            # non-streaming path, which is not the one production agents use,
            # so their turns left no record of WHICH tools had been called.
            if response.tool_calls:
                logger.info(
                    "Tool calls in iteration",
                    node_name=self.name,
                    iteration=iteration,
                    tool_calls_count=len(response.tool_calls),
                    tool_names=[tc.name for tc in response.tool_calls],
                )

            # Execute tool calls (with repeat guardrail)
            tool_results: list[ToolResult] = []
            pending_calls: list[ToolCall] = []
            pending_indices: list[int] = []
            skipped_count = 0
            for tool_call in response.tool_calls or []:
                call_key = self._tool_call_key(tool_call)
                tool_call_counts[call_key] += 1

                if tool_call_counts[call_key] > max_tool_call_repeats:
                    skipped_count += 1
                    logger.warning(
                        "Skipping repeated tool call",
                        node_name=self.name,
                        tool_name=tool_call.name,
                        iteration=iteration,
                        repeat_count=tool_call_counts[call_key],
                    )
                    skip_result = ToolResult(
                        success=True,
                        output=(
                            f"This exact call to '{tool_call.name}' with the same arguments "
                            f"has already been executed {max_tool_call_repeats} times. "
                            "The result will not change. Do NOT retry this call. "
                            "Move on to the next field or produce your final answer."
                        ),
                        tool_call_id=tool_call.id,
                    )
                    tool_results.append(skip_result)
                    yield ToolCallResultEvent(
                        node_name=self.name,
                        tool_name=tool_call.name,
                        success=True,
                        output=skip_result.output,
                        error=None,
                    )
                    continue

                yield ToolCallStartEvent(
                    node_name=self.name,
                    tool_name=tool_call.name,
                    tool_input=tool_call.input,
                    tool_call_id=tool_call.id,
                )

                logger.debug(
                    "Executing tool",
                    node_name=self.name,
                    tool_name=tool_call.name,
                    iteration=iteration,
                )
                pending_indices.append(len(tool_results))
                tool_results.append(None)  # type: ignore[arg-type]
                pending_calls.append(tool_call)

            # Execute pending tool calls in parallel
            if pending_calls:
                executed = await asyncio.gather(
                    *(self._execute_tool(tc) for tc in pending_calls)
                )
                for idx, tc, result in zip(pending_indices, pending_calls, executed):
                    tool_results[idx] = result
                    yield ToolCallResultEvent(
                        node_name=self.name,
                        tool_name=tc.name,
                        success=result.success,
                        output=result.output,
                        error=result.error,
                    )

            history.append((response, tool_results))
            iteration += 1

            # A whole batch skipped as repeats may be an accident, so it gets
            # another round. Only when the stall repeats is the turn closed.
            total_calls = len(response.tool_calls or [])
            all_skipped = total_calls > 0 and skipped_count == total_calls
            all_skipped_batches = all_skipped_batches + 1 if all_skipped else 0
            if all_skipped_batches >= self._ALL_SKIPPED_BATCHES_BEFORE_FINAL:
                logger.warning(
                    "All tool calls skipped, forcing final LLM response",
                    node_name=self.name,
                    iteration=iteration,
                    consecutive_batches=all_skipped_batches,
                )
                force_messages = self._build_messages_from_history(prompt, history)
                force_messages.append(self._final_turn_message(self._FINAL_TURN_ALL_SKIPPED))
                final_response = await self._llm.invoke(
                    prompt=prompt,
                    tools=None,
                    messages=force_messages,
                    system_prompt=self._system_prompt,
                    response_format=self._config.response_format,
                )
                self._accumulate_usage(final_response)
                if final_response.content:
                    yield TextChunkEvent(
                        node_name=self.name,
                        content=final_response.content,
                    )
                if self._raw_text_field is not None:
                    output_data = {self._raw_text_field: final_response.content or ""}
                else:
                    output_data = await self._parse_json_with_retries(
                        response=final_response,
                        messages=force_messages,
                        prompt=prompt,
                        tool_specs=None,
                        max_retries=self._config.max_parse_retries,
                        iteration=iteration,
                    )

                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.info(
                    "Node streaming execution completed (forced final after all-skipped)",
                    node_name=self.name,
                    duration_ms=round(duration_ms, 2),
                    iterations=iteration,
                    tool_calls_total=sum(
                        len(r.tool_calls or []) for r, _ in history
                    ),
                )
                self._last_stream_output = self._apply_output_field(output_data, input_data)
                return

        # Max iterations reached — graceful exit: one final LLM call WITHOUT
        # tools so the model can synthesise an answer, ask the user whether to
        # continue, or explain what it has done so far.
        logger.warning(
            "Max iterations reached, making final graceful response",
            node_name=self.name,
            max_iterations=max_iterations,
        )
        graceful_messages = self._build_messages_from_history(prompt, history)
        graceful_messages.append(self._final_turn_message(self._FINAL_TURN_MAX_ITERATIONS))

        try:
            if isinstance(self._llm, StreamableLLMCallable):
                graceful_content = ""
                graceful_usage: TokenUsage | None = None
                graceful_model = ""
                graceful_provider = ""
                async for event in self._llm.stream_invoke(
                    prompt=prompt,
                    tools=None,
                    messages=graceful_messages,
                    system_prompt=self._system_prompt,
                ):
                    if event.delta:
                        graceful_content += event.delta
                        yield TextChunkEvent(
                            node_name=self.name,
                            content=event.delta,
                        )
                    if event.usage:
                        graceful_usage = event.usage
                    if not graceful_model:
                        graceful_model = getattr(event, "model_requested", "") or ""
                    if not graceful_provider:
                        graceful_provider = getattr(event, "provider", "") or ""
                graceful_response = LLMResponse(
                    content=graceful_content or None,
                    tool_calls=None,
                    usage=graceful_usage,
                    model_requested=graceful_model,
                    provider=graceful_provider,
                )
            else:
                graceful_response = await self._llm.invoke(
                    prompt=prompt,
                    tools=None,
                    messages=graceful_messages,
                    system_prompt=self._system_prompt,
                )
                if graceful_response.content:
                    yield TextChunkEvent(
                        node_name=self.name,
                        content=graceful_response.content,
                    )
            self._accumulate_usage(graceful_response)
        except Exception:
            logger.error(
                "Graceful final response failed, raising MaxIterationsError",
                node_name=self.name,
            )
            raise MaxIterationsError(
                node_name=self.name,
                max_iterations=max_iterations,
                iterations_completed=iteration,
            )

        if self._raw_text_field is not None:
            output_data = {self._raw_text_field: graceful_response.content or ""}
        else:
            output_data = await self._parse_json_with_retries(
                response=graceful_response,
                messages=graceful_messages,
                prompt=prompt,
                tool_specs=None,
                max_retries=self._config.max_parse_retries,
                iteration=iteration,
            )

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Node streaming execution completed (graceful exit after max iterations)",
            node_name=self.name,
            duration_ms=round(duration_ms, 2),
            iterations=iteration,
            tool_calls_total=sum(
                len(r.tool_calls or []) for r, _ in history
            ),
        )
        self._last_stream_output = self._apply_output_field(output_data, input_data)
        return

    async def _run(
        self,
        input_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the LLM node logic.

        This method handles prompt assembly, LLM invocation, and tool loop.
        Input/output validation is handled by the parent BaseNode class.

        Args:
            input_data: Validated input dictionary.
            context: Optional context dictionary.

        Returns:
            Output dictionary from LLM execution.

        Raises:
            MaxIterationsError: If tool loop exceeds max iterations.
            NodeExecutionError: If LLM call fails.
        """
        start_time = time.perf_counter()
        node_type = getattr(self._config, "node_type", "llm")

        logger.info(
            "Node execution started",
            node_name=self.name,
            node_type=node_type,
            tools_available=len(self._tools),
        )

        try:
            # Get runtime variables and assemble prompt
            runtime_vars = self._get_runtime_vars()
            # Include validated input in runtime vars for prompt templates
            runtime_vars.update(input_data)
            prompt = self._prompt_assembler.assemble(runtime_vars=runtime_vars)

            # Tool loop
            max_iterations = self._get_max_iterations()
            parallel_tool_calls = self._get_parallel_tool_calls()
            tool_specs = await self._get_merged_tool_specs()
            tool_catalog_revision = self._tool_catalog_revision()

            base_prompt = prompt
            prompt = self._prompt_with_registry_tools(prompt, tool_specs, runtime_vars)

            history: list[tuple[LLMResponse, list[ToolResult]]] = []
            iteration = 0
            tool_call_counts: Counter[str] = Counter()
            max_tool_call_repeats = self._get_max_tool_call_repeats()
            token_budget = self._get_max_tool_loop_tokens()
            token_reserve = self._get_tool_loop_token_reserve()
            all_skipped_batches = 0

            logger.info(
                "Tool loop config",
                node_name=self.name,
                max_iterations=max_iterations,
                parallel_tool_calls=parallel_tool_calls,
                max_tool_call_repeats=max_tool_call_repeats,
                max_tool_loop_tokens=token_budget,
                tool_loop_token_reserve=token_reserve,
                tools_count=len(tool_specs) if tool_specs else 0,
            )

            while iteration < max_iterations:
                tool_catalog_revision, tool_specs, prompt = await self._refresh_tool_catalog(
                    tool_catalog_revision, tool_specs, base_prompt, prompt, runtime_vars
                )

                if token_budget and history:
                    history, fits = await self._fit_history_to_budget(
                        prompt, tool_specs, history, token_budget, token_reserve
                    )
                    if not fits:
                        break

                # Check should_continue hook
                if iteration > 0 and history:
                    last_response = history[-1][0]
                    if self._should_continue is not None:
                        if not self._should_continue(iteration, last_response):
                            logger.debug(
                                "Tool loop stopped by should_continue hook",
                                node_name=self.name,
                                iteration=iteration,
                            )
                            break

                # Build messages from history
                messages = self._build_messages_from_history(prompt, history) if history else None

                # Call LLM (with retry on failure)
                max_invoke_retries = 2
                last_invoke_error: Exception | None = None
                response = None
                for invoke_attempt in range(max_invoke_retries + 1):
                    try:
                        response = await self._llm.invoke(
                            prompt=prompt,
                            tools=tool_specs,
                            messages=messages,
                            system_prompt=self._system_prompt,
                            response_format=self._config.response_format,
                            parallel_tool_calls=parallel_tool_calls,  # type: ignore[call-arg]
                        )
                        last_invoke_error = None
                        break
                    except Exception as e:
                        last_invoke_error = e
                        if invoke_attempt < max_invoke_retries:
                            logger.warning(
                                "LLM invocation failed, retrying",
                                node_name=self.name,
                                iteration=iteration,
                                attempt=invoke_attempt + 1,
                                max_retries=max_invoke_retries,
                                error=str(e),
                            )
                            continue
                        logger.error(
                            "LLM invocation failed",
                            node_name=self.name,
                            iteration=iteration,
                            attempt=invoke_attempt + 1,
                            error=str(e),
                        )
                        raise NodeExecutionError(
                            f"LLM invocation failed in node '{self.name}': {e}",
                            node_name=self.name,
                            cause=e,
                        ) from e

                if last_invoke_error is not None or response is None:
                    raise NodeExecutionError(
                        f"LLM invocation failed in node '{self.name}': {last_invoke_error}",
                        node_name=self.name,
                        cause=last_invoke_error,
                    )

                self._accumulate_usage(response)

                # Invoke on_llm_response hook
                if self._on_llm_response is not None:
                    self._on_llm_response(response, iteration)

                # Log tool calls per iteration
                if response.tool_calls:
                    tool_names = [tc.name for tc in response.tool_calls]
                    logger.info(
                        "Tool calls in iteration",
                        node_name=self.name,
                        iteration=iteration,
                        tool_calls_count=len(response.tool_calls),
                        tool_names=tool_names,
                    )

                # Check if final response
                if response.is_final:
                    if self._raw_text_field is not None:
                        output_data = {self._raw_text_field: response.content or ""}
                    else:
                        output_data = await self._parse_json_with_retries(
                            response=response,
                            messages=messages,
                            prompt=prompt,
                            tool_specs=tool_specs,
                            max_retries=self._config.max_parse_retries,
                            iteration=iteration,
                        )

                    duration_ms = (time.perf_counter() - start_time) * 1000
                    logger.info(
                        "Node execution completed",
                        node_name=self.name,
                        node_type=node_type,
                        duration_ms=round(duration_ms, 2),
                        iterations=iteration + 1,
                        tool_calls_total=sum(
                            len(r.tool_calls or []) for r, _ in history
                        ),
                    )
                    return self._apply_output_field(output_data, input_data)

                # Execute tool calls (with repeat guardrail)
                tool_results: list[ToolResult] = []
                pending_calls: list[ToolCall] = []
                pending_indices: list[int] = []
                skipped_count = 0
                for tool_call in response.tool_calls or []:
                    call_key = self._tool_call_key(tool_call)
                    tool_call_counts[call_key] += 1

                    if tool_call_counts[call_key] > max_tool_call_repeats:
                        skipped_count += 1
                        logger.warning(
                            "Skipping repeated tool call",
                            node_name=self.name,
                            tool_name=tool_call.name,
                            iteration=iteration,
                            repeat_count=tool_call_counts[call_key],
                        )
                        tool_results.append(ToolResult(
                            success=True,
                            output=(
                                f"This exact call to '{tool_call.name}' with the same arguments "
                                f"has already been executed {max_tool_call_repeats} times. "
                                "The result will not change. Do NOT retry this call. "
                                "Move on to the next field or produce your final answer."
                            ),
                            tool_call_id=tool_call.id,
                        ))
                        continue

                    logger.debug(
                        "Executing tool",
                        node_name=self.name,
                        tool_name=tool_call.name,
                        iteration=iteration,
                    )
                    pending_indices.append(len(tool_results))
                    tool_results.append(None)  # type: ignore[arg-type]
                    pending_calls.append(tool_call)

                # Execute pending tool calls in parallel
                if pending_calls:
                    executed = await asyncio.gather(
                        *(self._execute_tool(tc) for tc in pending_calls)
                    )
                    for idx, result in zip(pending_indices, executed):
                        tool_results[idx] = result

                # Add to history
                history.append((response, tool_results))
                iteration += 1

                # A whole batch skipped as repeats may be an accident, so it
                # gets another round. Only when the stall repeats is the turn closed.
                total_calls = len(response.tool_calls or [])
                all_skipped = total_calls > 0 and skipped_count == total_calls
                all_skipped_batches = all_skipped_batches + 1 if all_skipped else 0
                if all_skipped_batches >= self._ALL_SKIPPED_BATCHES_BEFORE_FINAL:
                    logger.warning(
                        "All tool calls skipped, forcing final LLM response",
                        node_name=self.name,
                        iteration=iteration,
                        consecutive_batches=all_skipped_batches,
                    )
                    messages = self._build_messages_from_history(prompt, history)
                    messages.append(self._final_turn_message(self._FINAL_TURN_ALL_SKIPPED))
                    final_response = await self._llm.invoke(
                        prompt=prompt,
                        tools=None,
                        messages=messages,
                        system_prompt=self._system_prompt,
                        response_format=self._config.response_format,
                    )
                    self._accumulate_usage(final_response)
                    if self._raw_text_field is not None:
                        output_data = {self._raw_text_field: final_response.content or ""}
                    else:
                        output_data = await self._parse_json_with_retries(
                            response=final_response,
                            messages=messages,
                            prompt=prompt,
                            tool_specs=None,
                            max_retries=self._config.max_parse_retries,
                            iteration=iteration,
                        )

                    duration_ms = (time.perf_counter() - start_time) * 1000
                    logger.info(
                        "Node execution completed (forced final after all-skipped)",
                        node_name=self.name,
                        node_type=node_type,
                        duration_ms=round(duration_ms, 2),
                        iterations=iteration,
                        tool_calls_total=sum(
                            len(r.tool_calls or []) for r, _ in history
                        ),
                    )
                    return self._apply_output_field(output_data, input_data)

            # Max iterations reached — graceful exit: one final LLM call
            # WITHOUT tools so the model can synthesise an answer or explain
            # what remains to be done.
            logger.warning(
                "Max iterations reached, making final graceful response",
                node_name=self.name,
                max_iterations=max_iterations,
            )
            graceful_messages = self._build_messages_from_history(prompt, history)
            graceful_messages.append(self._final_turn_message(self._FINAL_TURN_MAX_ITERATIONS))

            try:
                graceful_response = await self._llm.invoke(
                    prompt=prompt,
                    tools=None,
                    messages=graceful_messages,
                    system_prompt=self._system_prompt,
                )
                self._accumulate_usage(graceful_response)
            except Exception:
                logger.error(
                    "Graceful final response failed, raising MaxIterationsError",
                    node_name=self.name,
                )
                raise MaxIterationsError(
                    node_name=self.name,
                    max_iterations=max_iterations,
                    iterations_completed=iteration,
                )

            if self._raw_text_field is not None:
                output_data = {self._raw_text_field: graceful_response.content or ""}
            else:
                output_data = await self._parse_json_with_retries(
                    response=graceful_response,
                    messages=graceful_messages,
                    prompt=prompt,
                    tool_specs=None,
                    max_retries=self._config.max_parse_retries,
                    iteration=iteration,
                )

            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "Node execution completed (graceful exit after max iterations)",
                node_name=self.name,
                node_type=node_type,
                duration_ms=round(duration_ms, 2),
                iterations=iteration,
                tool_calls_total=sum(
                    len(r.tool_calls or []) for r, _ in history
                ),
            )
            return self._apply_output_field(output_data, input_data)

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Node execution failed",
                node_name=self.name,
                node_type=node_type,
                duration_ms=round(duration_ms, 2),
                error_type=type(e).__name__,
                error=str(e),
            )
            raise

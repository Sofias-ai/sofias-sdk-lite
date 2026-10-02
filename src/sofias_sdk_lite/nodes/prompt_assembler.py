"""Prompt assembly for node execution.

This module provides the PromptAssembler class that constructs the final
prompt sent to the LLM by combining fixed and variable parts.

ToolSpec is re-exported from sofias_sdk_lite.llm for backward compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sofias_sdk_lite.llm import ToolSpec

if TYPE_CHECKING:
    from sofias_sdk_lite.contracts import StrictContract
    from sofias_sdk_lite.nodes.llm_node_config import LLMNodeConfig as NodeConfig

# Re-export so existing importers keep working
__all__ = ["PromptAssembler", "ToolSpec"]


class PromptAssembler:
    """Assembles the final prompt for LLM calls.

    The prompt has two parts:
    - Fixed: The base instruction from NodeConfig (string or file).
    - Variable: Generated at runtime based on available tools and output contract.

    Template variables in the fixed prompt are resolved using NodeConfig
    variables merged with runtime variables (runtime takes precedence).

    Example:
        ```python
        assembler = PromptAssembler(
            config=node_config,
            tools=[tool_spec_1, tool_spec_2],
            output_contract=MyOutputContract,
        )
        prompt = assembler.assemble(runtime_vars={"user_name": "Alice"})
        ```
    """

    def __init__(
        self,
        config: NodeConfig,
        tools: list[ToolSpec] | None = None,
        output_contract: type[StrictContract] | None = None,
        raw_text: bool = False,
    ) -> None:
        """Initialize the prompt assembler.

        Args:
            config: Node configuration containing prompt settings.
            tools: List of tool specifications available to this node.
            output_contract: Output contract class for schema formatting.
            raw_text: If True, skip JSON schema output section in the prompt.
        """
        self._config = config
        self._tools = tools or []
        self._output_contract = output_contract
        self._raw_text = raw_text

    @property
    def config(self) -> NodeConfig:
        """The node configuration."""
        return self._config

    @property
    def tools(self) -> list[ToolSpec]:
        """The tools available to this node."""
        return self._tools

    def _load_fixed_prompt(self) -> str:
        """Load the fixed prompt from config.

        Returns:
            The fixed prompt string, or empty string if not configured.
        """
        prompt_config = self._config.prompt
        if prompt_config is None:
            return ""

        # Direct string prompt takes precedence
        if prompt_config.prompt is not None:
            return prompt_config.prompt

        # Load from file if path is specified
        if prompt_config.prompt_path is not None:
            path = prompt_config.prompt_path
            if not isinstance(path, Path):
                path = Path(path)
            if path.exists():
                return path.read_text(encoding="utf-8")
            return ""

        return ""

    def _get_template_variables(
        self, runtime_vars: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Merge config template variables with runtime variables.

        Runtime variables take precedence over config variables.

        Args:
            runtime_vars: Optional runtime variables to merge.

        Returns:
            Merged dictionary of template variables.
        """
        variables: dict[str, Any] = {}

        # Start with config variables
        if self._config.prompt and self._config.prompt.template_variables:
            variables.update(self._config.prompt.template_variables)

        # Runtime vars override config vars
        if runtime_vars:
            variables.update(runtime_vars)

        return variables

    def _resolve_template(self, template: str, variables: dict[str, Any]) -> str:
        """Resolve template variables in a string.

        Uses {variable_name} syntax for substitution.

        Args:
            template: The template string with {placeholders}.
            variables: Dictionary of variable names to values.

        Returns:
            The resolved string with placeholders replaced.
        """
        if not variables:
            return template

        result = template
        for key, value in variables.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))

        return result

    def _load_mcp_config(self) -> dict[str, str]:
        """Load MCP configuration from the JSON file specified in config.

        This only reads a user-supplied JSON file describing MCP server
        names and descriptions for prompt rendering. It does not perform
        MCP discovery or connect to any MCP server.

        Returns:
            Dictionary mapping MCP names to their descriptions,
            or empty dict if not configured or file doesn't exist.
        """
        prompt_config = self._config.prompt
        if prompt_config is None or prompt_config.mcp_config_path is None:
            return {}

        path = prompt_config.mcp_config_path
        if not isinstance(path, Path):
            path = Path(path)
        if not path.exists():
            return {}

        return json.loads(path.read_text(encoding="utf-8"))

    def _format_mcp_section(self) -> str:
        """Format the MCP section for the prompt.

        Returns:
            Formatted markdown string listing available MCPs,
            or empty string if none.
        """
        mcp_config = self._load_mcp_config()
        if not mcp_config:
            return ""

        lines = ["## Available MCPs", ""]
        for name, description in mcp_config.items():
            lines.append(f"- **{name}**: {description}")
        lines.append("")

        return "\n".join(lines)

    def _format_tools_section(self) -> str:
        """Format the tools section for the prompt.

        Returns:
            Formatted string describing available tools, or empty string if none.
        """
        if not self._tools:
            return ""

        lines = ["## Available Tools", ""]
        for tool in self._tools:
            lines.append(f"### {tool.name}")
            lines.append(tool.description)
            lines.append("")
            lines.append("Parameters:")
            lines.append(f"```json\n{self._format_schema(tool.parameters_schema)}\n```")
            lines.append("")

        return "\n".join(lines)

    def _format_schema(self, schema: dict[str, Any]) -> str:
        """Format a JSON schema for display.

        Args:
            schema: The JSON schema dictionary.

        Returns:
            Formatted string representation of the schema.
        """
        return json.dumps(schema, indent=2)

    def _format_output_section(self) -> str:
        """Format the expected output section.

        Output structure is enforced via response_format at the provider API
        level, not via prompt text. Injecting the JSON schema into the prompt
        confused LLMs (especially with nested $ref/$defs) and wasted tokens.

        Returns:
            Always returns empty string.
        """
        return ""

    def assemble(self, runtime_vars: dict[str, Any] | None = None) -> str:
        """Assemble the final prompt.

        Combines the fixed prompt with variable sections (tools, output format)
        and resolves template variables.

        Args:
            runtime_vars: Optional runtime variables for template resolution.

        Returns:
            The complete assembled prompt ready for LLM invocation.
        """
        # Get template variables
        variables = self._get_template_variables(runtime_vars)

        # Load and resolve fixed prompt
        fixed_prompt = self._load_fixed_prompt()
        resolved_prompt = self._resolve_template(fixed_prompt, variables)

        # Build variable sections
        sections = [resolved_prompt] if resolved_prompt else []

        mcp_section = self._format_mcp_section()
        if mcp_section:
            sections.append(mcp_section)

        tools_section = self._format_tools_section()
        if tools_section:
            sections.append(tools_section)

        output_section = self._format_output_section()
        if output_section:
            sections.append(output_section)

        # Join sections with clear separation
        return "\n\n".join(sections).strip()

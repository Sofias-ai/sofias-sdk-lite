"""Shared fakes for LLMNode tests: a scripted in-memory LLM and tool registry."""

from __future__ import annotations

from typing import Any

from sofias_sdk_lite import InputContract, LLMNodeConfig, LLMResponse, NodeContract, OutputContract
from sofias_sdk_lite.llm import ToolCall, ToolSpec
from sofias_sdk_lite.nodes.llm_node import LLMNode
from sofias_sdk_lite.nodes.llm_node_config import NodeToolLoopConfig


class QueryInput(InputContract):
    query: str


class ResultOutput(OutputContract):
    result: str


CONTRACT = NodeContract(input_schema=QueryInput, output_schema=ResultOutput)


class FakeLLM:
    """Scripted ``LLMCallable``: returns (or raises) the next queued response."""

    def __init__(self, *responses: LLMResponse | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def invoke(
        self,
        prompt: str,
        tools: list[ToolSpec] | None = None,
        messages: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        response_format: dict[str, Any] | None = None,
        **_: Any,
    ) -> LLMResponse:
        self.calls.append(
            {"prompt": prompt, "tools": tools, "messages": messages, "system_prompt": system_prompt}
        )
        if not self.responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    def sent_text(self, index: int) -> str:
        """Everything in the ``messages`` of call number ``index`` as one string."""
        import json

        return json.dumps(self.calls[index]["messages"] or [], ensure_ascii=False, default=str)


class FakeRegistry:
    """Minimal ``ToolRegistry`` stand-in: fixed specs, scripted outputs."""

    def __init__(self, specs: list[ToolSpec], outputs: list[Any] | None = None) -> None:
        self.specs = specs
        self.outputs = list(outputs or [])
        self.executed: list[tuple[str, dict[str, Any]]] = []

    async def get_tools_for_node(self, node_name: str) -> list[ToolSpec]:
        return self.specs

    async def execute(self, name: str, tool_input: dict[str, Any]) -> Any:
        self.executed.append((name, tool_input))
        if self.outputs:
            return self.outputs.pop(0)
        return f"{name} ok"


def spec(name: str = "read", description: str = "reads a file") -> ToolSpec:
    return ToolSpec(name=name, description=description, parameters_schema={"type": "object"})


def tool_call(name: str = "read", call_id: str = "c1", **args: Any) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name=name, input=args, id=call_id)])


def final(content: str = '{"result": "ok"}') -> LLMResponse:
    return LLMResponse(content=content)


def make_config(
    *,
    system_prompt: str | None = None,
    tool_loop: NodeToolLoopConfig | None = None,
) -> LLMNodeConfig:
    return LLMNodeConfig(
        name="worker",
        input_contract=QueryInput,
        output_contract=ResultOutput,
        system_prompt=system_prompt,
        tool_loop=tool_loop or NodeToolLoopConfig(max_iterations=10),
    )


def make_node(
    llm: FakeLLM,
    *,
    config: LLMNodeConfig | None = None,
    registry: FakeRegistry | None = None,
    **kwargs: Any,
) -> LLMNode:
    return LLMNode(
        config=config or make_config(),
        contract=CONTRACT,
        llm=llm,
        tool_registry=registry,
        **kwargs,
    )

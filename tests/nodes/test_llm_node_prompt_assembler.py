"""A caller-supplied ``prompt_assembler`` is authoritative.

The runtime used to rebuild the assembler with ALL tools (static plus
registry) whenever a registry was present, overriding whatever the node's
author passed. A caller who deliberately passes ``tools=[]`` to keep schemas
out of the prompt text (they already travel in the API's ``tools`` field) saw
them re-injected anyway and paid for every schema twice.
"""

from __future__ import annotations

from sofias_sdk_lite import LLMNodeConfig
from sofias_sdk_lite.nodes.llm_node import LLMNode
from sofias_sdk_lite.nodes.prompt_assembler import PromptAssembler
from tests.nodes.llm_fakes import CONTRACT, FakeLLM, FakeRegistry, final, make_config, spec

TOOLS_SECTION = "## Available Tools"


def _assembler_without_tools(config: LLMNodeConfig) -> PromptAssembler:
    return PromptAssembler(config=config, tools=[], raw_text=False)


class TestCallerAssembler:
    async def test_schemas_are_not_reinjected_into_the_prompt(self) -> None:
        llm = FakeLLM(final())
        config = make_config()
        node = LLMNode(
            config=config,
            contract=CONTRACT,
            llm=llm,
            prompt_assembler=_assembler_without_tools(config),
            tool_registry=FakeRegistry([spec("reg_a"), spec("reg_b")]),
        )

        await node.execute({"query": "hi"})

        assert TOOLS_SECTION not in llm.calls[0]["prompt"]

    async def test_tools_still_reach_the_model_through_the_tools_field(self) -> None:
        llm = FakeLLM(final())
        config = make_config()
        node = LLMNode(
            config=config,
            contract=CONTRACT,
            llm=llm,
            prompt_assembler=_assembler_without_tools(config),
            tool_registry=FakeRegistry([spec("reg_a"), spec("reg_b")]),
        )

        await node.execute({"query": "hi"})

        assert {t.name for t in llm.calls[0]["tools"]} == {"reg_a", "reg_b"}

    def test_marks_the_assembler_as_foreign(self) -> None:
        config = make_config()
        node = LLMNode(
            config=config,
            contract=CONTRACT,
            llm=FakeLLM(),
            prompt_assembler=_assembler_without_tools(config),
        )
        assert node._owns_prompt_assembler is False


class TestOwnAssembler:
    """Without a caller assembler nothing changes: the runtime still injects."""

    async def test_still_injects_registry_tools(self) -> None:
        llm = FakeLLM(final())
        node = LLMNode(
            config=make_config(),
            contract=CONTRACT,
            llm=llm,
            tool_registry=FakeRegistry([spec("reg_a")]),
        )

        await node.execute({"query": "hi"})

        prompt = llm.calls[0]["prompt"]
        assert TOOLS_SECTION in prompt
        assert "reg_a" in prompt

    def test_marks_the_assembler_as_owned(self) -> None:
        node = LLMNode(config=make_config(), contract=CONTRACT, llm=FakeLLM())
        assert node._owns_prompt_assembler is True

    async def test_no_registry_means_no_rebuild(self) -> None:
        llm = FakeLLM(final())
        node = LLMNode(config=make_config(), contract=CONTRACT, llm=llm)

        await node.execute({"query": "hi"})

        assert TOOLS_SECTION not in llm.calls[0]["prompt"]

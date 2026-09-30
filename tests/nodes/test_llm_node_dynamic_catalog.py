"""The tool catalog can grow mid-turn.

Tool specs used to be resolved ONCE, before the loop. Any tool that loads
others on demand was useless: the agent called ``load_tools(connector)``, the
registry grew... and the model kept seeing the old catalog until the end of
the turn, because nobody asked again. The only way out was to send every tool
from the first message and pay for all of them.

The node now reads ``registry.revision`` between iterations. When it has not
moved (every agent that does not use this) there is no refresh and the cost
is comparing two integers.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from sofias_sdk_lite.llm import ToolSpec
from sofias_sdk_lite.tools import ToolRegistry
from tests.nodes.llm_fakes import FakeLLM, FakeRegistry, final, make_node, spec, tool_call


class GrowingProvider:
    """Provider that widens its catalog when ``load`` runs.

    The minimal shape of a real on-demand policy: a catalog tool that, when
    called, brings in a connector's tools.
    """

    def __init__(self) -> None:
        self.names = ["load"]
        self.registry: ToolRegistry | None = None

    async def get_tools(self, node_name: str) -> list[ToolSpec]:
        return [spec(n) for n in self.names]

    async def execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        if tool_name == "load":
            self.names = ["load", "read_mail", "send_mail"]
            assert self.registry is not None
            self.registry.clear_cache()
            return "loaded 2 tools"
        return "ok"


def _growing_registry() -> ToolRegistry:
    provider = GrowingProvider()
    registry = ToolRegistry(provider)
    provider.registry = registry
    return registry


def _tools_per_call(llm: FakeLLM) -> list[set[str]]:
    """Which tools the model saw on each call, in order."""
    return [{t.name for t in call["tools"] or []} for call in llm.calls]


class TestGrowingCatalog:
    async def test_model_sees_the_tools_it_just_loaded(self) -> None:
        llm = FakeLLM(tool_call("load"), final())
        node = make_node(llm, registry=_growing_registry())  # type: ignore[arg-type]

        await node._run({"query": "read my mail"})

        seen = _tools_per_call(llm)
        assert seen[0] == {"load"}
        assert seen[1] == {"load", "read_mail", "send_mail"}

    async def test_streaming_path_sees_them_too(self) -> None:
        llm = FakeLLM(tool_call("load"), final())
        node = make_node(llm, registry=_growing_registry())  # type: ignore[arg-type]

        async for _ in node._stream_run({"query": "read my mail"}):
            pass

        assert _tools_per_call(llm)[1] == {"load", "read_mail", "send_mail"}

    async def test_prompt_is_reassembled_from_the_original_not_stacked(self) -> None:
        llm = FakeLLM(tool_call("load"), final())
        node = make_node(llm, registry=_growing_registry())  # type: ignore[arg-type]

        await node._run({"query": "read my mail"})

        prompt = llm.calls[1]["prompt"]
        assert "### read_mail" in prompt
        assert prompt.count("### load") == 1

    async def test_unchanged_catalog_is_not_fetched_again(self) -> None:
        # The guarantee this is free for whoever does not use it: a turn of
        # three iterations that never touches the catalog asks ONCE.
        registry = _growing_registry()
        spy = AsyncMock(wraps=registry.get_tools_for_node)
        registry.get_tools_for_node = spy  # type: ignore[method-assign]
        llm = FakeLLM(tool_call("other", call_id="a"), tool_call("other", call_id="b"), final())
        node = make_node(llm, registry=registry)  # type: ignore[arg-type]

        await node._run({"query": "hi"})

        assert spy.await_count == 1

    async def test_registry_without_revision_keeps_working(self) -> None:
        # Custom registries / test doubles that do not keep count must behave
        # exactly as before.
        llm = FakeLLM(tool_call("search"), final())
        node = make_node(llm, registry=FakeRegistry([spec("search")]))

        result = await node._run({"query": "hi"})

        assert result == {"result": "ok"}
        assert _tools_per_call(llm) == [{"search"}, {"search"}]


class TestCacheHonoursProviderRevision:
    """A cache that ignores its source's invalidation is a broken cache.

    If the provider changes its catalog on its own — without going through
    ``clear_cache``, exactly the case ``revision`` covers — the per-node cache
    kept returning the stale list and the node's refresh achieved nothing.
    """

    async def test_new_provider_revision_invalidates_cached_tools(self) -> None:
        class SelfChangingProvider:
            def __init__(self) -> None:
                self.revision = 0
                self.names = ["a"]

            async def get_tools(self, node_name: str) -> list[ToolSpec]:
                return [spec(n) for n in self.names]

            async def execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
                return "ok"

        provider = SelfChangingProvider()
        registry = ToolRegistry(provider)

        first = await registry.get_tools_for_node("n")
        provider.names = ["a", "b"]
        provider.revision += 1
        second = await registry.get_tools_for_node("n")

        assert {t.name for t in first} == {"a"}
        assert {t.name for t in second} == {"a", "b"}

    async def test_same_revision_keeps_serving_the_cache(self) -> None:
        provider = AsyncMock()
        provider.get_tools = AsyncMock(return_value=[spec("a")])
        del provider.revision
        registry = ToolRegistry(provider)

        await registry.get_tools_for_node("n")
        await registry.get_tools_for_node("n")

        assert provider.get_tools.await_count == 1


class TestRegistryRevision:
    def test_starts_at_zero(self) -> None:
        assert ToolRegistry(GrowingProvider()).revision == 0

    def test_clear_cache_moves_it(self) -> None:
        registry = ToolRegistry(GrowingProvider())
        before = registry.revision
        registry.clear_cache()
        assert registry.revision != before

    def test_clearing_one_node_moves_it_too(self) -> None:
        registry = ToolRegistry(GrowingProvider())
        before = registry.revision
        registry.clear_cache("some_node")
        assert registry.revision != before

    def test_adds_the_provider_revision_when_exposed(self) -> None:
        # A cacheless provider can change tools without calling clear_cache;
        # without adding its revision the node would never notice.
        class ProviderWithRevision(GrowingProvider):
            revision = 7

        assert ToolRegistry(ProviderWithRevision()).revision == 7

    def test_non_integer_provider_revision_is_ignored(self) -> None:
        class OddProvider(GrowingProvider):
            revision = "not a number"

        assert ToolRegistry(OddProvider()).revision == 0

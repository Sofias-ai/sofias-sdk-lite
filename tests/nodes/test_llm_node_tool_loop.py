"""Tool-loop memory, token budget, and honest turn closing.

The loop keeps its history intact while it fits; when it does not, it frees
room for real by consolidating the oldest iterations into a working memory
(never by blind truncation). All exits share one closing text that never
claims the model "has all the data" and forbids blaming tools or permissions.
"""

from __future__ import annotations

import logging

import pytest

from sofias_sdk_lite import LLMResponse
from sofias_sdk_lite.config import DEFAULT_MAX_TOOL_LOOP_TOKENS, DEFAULT_TOOL_LOOP_TOKEN_RESERVE
from sofias_sdk_lite.nodes.llm_node import LLMNode
from sofias_sdk_lite.nodes.llm_node_config import NodeToolLoopConfig
from tests.nodes.llm_fakes import (
    FakeLLM,
    FakeRegistry,
    final,
    make_config,
    make_node,
    spec,
    tool_call,
)


# ---------------------------------------------------------------------------
# History is kept intact while it fits
# ---------------------------------------------------------------------------


class TestHistoryKeptWhileItFits:
    async def test_two_old_reads_reach_the_model_intact(self) -> None:
        companies = "COMPANY;" + "e" * 5000
        contacts = "CONTACT;" + "c" * 5000
        llm = FakeLLM(
            tool_call(call_id="a", f="companies"),
            tool_call(call_id="b", f="contacts"),
            final(),
        )
        node = make_node(llm, registry=FakeRegistry([spec()], [companies, contacts]))

        result = await node._run({"query": "cross both files"})

        assert result == {"result": "ok"}
        sent = llm.sent_text(2)
        assert companies in sent, "the OLD read was lost"
        assert contacts in sent
        assert "chars truncated" not in sent

    async def test_one_huge_result_survives_without_pressure(self) -> None:
        big = "x" * 20_000
        llm = FakeLLM(tool_call(call_id="a", f="a"), tool_call(call_id="b", f="b"), final())
        cfg = make_config(
            tool_loop=NodeToolLoopConfig(max_iterations=10, max_tool_loop_tokens=500_000)
        )
        node = make_node(llm, config=cfg, registry=FakeRegistry([spec()], [big, "short"]))

        await node._run({"query": "hi"})

        assert big in llm.sent_text(2)


class TestBudgetMeasurement:
    def test_overhead_includes_system_prompt_and_tool_schemas(self) -> None:
        node = make_node(FakeLLM(), config=make_config(system_prompt="S" * 3200))
        specs = [spec(description="D" * 3200)]

        overhead = node._estimate_fixed_overhead_tokens("P" * 3200, specs)

        # 3200 chars ≈ 1000 tokens at 3.2 chars/token; three blocks.
        assert overhead >= 3000

    def test_nothing_to_count_yields_zero(self) -> None:
        node = make_node(FakeLLM(), config=make_config(system_prompt=None))
        assert node._estimate_fixed_overhead_tokens("", None) == 0

    def test_budget_means_the_whole_window(self) -> None:
        assert DEFAULT_MAX_TOOL_LOOP_TOKENS >= 128_000
        room = DEFAULT_MAX_TOOL_LOOP_TOKENS - 79_000 - DEFAULT_TOOL_LOOP_TOKEN_RESERVE
        assert room > 25_000

    def test_config_fields_resolve_over_defaults(self) -> None:
        cfg = make_config(
            tool_loop=NodeToolLoopConfig(max_tool_loop_tokens=1000, tool_loop_token_reserve=100)
        )
        node = make_node(FakeLLM(), config=cfg)
        assert node._get_max_tool_loop_tokens() == 1000
        assert node._get_tool_loop_token_reserve() == 100

    def test_defaults_when_unset(self) -> None:
        node = make_node(FakeLLM())
        assert node._get_max_tool_loop_tokens() == DEFAULT_MAX_TOOL_LOOP_TOKENS
        assert node._get_tool_loop_token_reserve() == DEFAULT_TOOL_LOOP_TOKEN_RESERVE


# ---------------------------------------------------------------------------
# No room: consolidate for real, don't truncate blindly
# ---------------------------------------------------------------------------


def _tight_node(llm: FakeLLM, outputs: list[str], *, budget: int = 2000, reserve: int = 500):
    cfg = make_config(
        tool_loop=NodeToolLoopConfig(
            max_iterations=10, max_tool_loop_tokens=budget, tool_loop_token_reserve=reserve
        )
    )
    return make_node(llm, config=cfg, registry=FakeRegistry([spec()], outputs))


class TestConsolidation:
    async def test_frees_room_and_keeps_working(self) -> None:
        llm = FakeLLM(
            tool_call(call_id="a", f="a"),
            final("20 companies read: Acme, Globex, …"),  # consolidation summary
            tool_call(call_id="b", f="b"),
            final(),
        )
        node = _tight_node(llm, ["y" * 8000, "short"])

        result = await node._run({"query": "hi"})

        assert result == {"result": "ok"}
        assert llm.calls[1]["tools"] is None  # consolidation call carries no tools
        assert len(llm.calls) == 4  # and the loop kept executing tools afterwards

    async def test_replaces_old_material_and_marks_it(self) -> None:
        old = "z" * 8000
        llm = FakeLLM(
            tool_call(call_id="a", f="a"),
            final("memory: 20 companies"),
            tool_call(call_id="b", f="b"),
            final(),
        )
        node = _tight_node(llm, [old, "short"])

        await node._run({"query": "hi"})

        after = llm.sent_text(2)
        assert old not in after, "consolidated material must never be re-sent"
        assert LLMNode._CONSOLIDATION_MARKER in after
        assert "memory: 20 companies" in after

    async def test_failed_consolidation_exits_honestly(self) -> None:
        llm = FakeLLM(
            tool_call(call_id="a", f="a"),
            RuntimeError("provider down"),
            final('{"result": "partial"}'),
        )
        node = _tight_node(llm, ["w" * 8000])

        result = await node._run({"query": "hi"})

        assert result == {"result": "partial"}
        instruction = llm.sent_text(2)
        assert "You have all the data you need" not in instruction
        assert "remains" in instruction

    async def test_no_consolidation_when_prefix_alone_exceeds_budget(self) -> None:
        llm = FakeLLM(tool_call(call_id="a", f="a"), final('{"result": "partial"}'))
        cfg = make_config(
            system_prompt="S" * 320_000,  # ≈100k tokens > budget on its own
            tool_loop=NodeToolLoopConfig(
                max_iterations=10, max_tool_loop_tokens=96_000, tool_loop_token_reserve=16_384
            ),
        )
        node = make_node(llm, config=cfg, registry=FakeRegistry([spec()], ["short"]))

        result = await node._run({"query": "hi"})

        assert result == {"result": "partial"}
        assert len(llm.calls) == 2  # one round + honest exit, NO consolidation call

    async def test_empty_history_is_not_consolidated(self) -> None:
        node = _tight_node(FakeLLM(), [])
        assert await node._consolidate_history("p", []) is None

    async def test_a_summary_is_not_summarised_again(self) -> None:
        llm = FakeLLM()
        node = _tight_node(llm, [])
        already = LLMResponse(content=f"{LLMNode._CONSOLIDATION_MARKER}\n20 companies")

        assert await node._consolidate_history("p", [(already, [])]) is None
        assert llm.calls == []

    async def test_single_huge_round_can_be_folded(self) -> None:
        llm = FakeLLM(tool_call(call_id="a", f="a"), final("memory: 20 companies"), final())
        node = _tight_node(llm, ["q" * 8000], budget=1500, reserve=500)

        result = await node._run({"query": "hi"})

        assert result == {"result": "ok"}
        assert llm.calls[1]["tools"] is None

    async def test_empty_summary_counts_as_failure(self) -> None:
        llm = FakeLLM(final("   "))
        node = _tight_node(llm, [])
        history = [(tool_call(call_id="a", f="a"), [])]

        assert await node._consolidate_history("p", history) is None


# ---------------------------------------------------------------------------
# Honest turn exits
# ---------------------------------------------------------------------------


def _repeat_node(llm: FakeLLM) -> LLMNode:
    cfg = make_config(tool_loop=NodeToolLoopConfig(max_iterations=10, max_tool_call_repeats=1))
    return make_node(llm, config=cfg, registry=FakeRegistry([spec("my_tool")]))


class TestHonestExits:
    async def test_first_all_skipped_batch_does_not_close_the_turn(self) -> None:
        same = tool_call("my_tool", call_id="s", q="same")
        llm = FakeLLM(same, same, final('{"result": "recovered"}'))
        node = _repeat_node(llm)

        result = await node._run({"query": "hi"})

        assert result == {"result": "recovered"}
        assert llm.calls[-1]["tools"] is not None  # still a normal round WITH tools

    async def test_second_all_skipped_batch_closes_the_turn(self) -> None:
        same = tool_call("my_tool", call_id="s", q="same")
        llm = FakeLLM(same, same, same, final('{"result": "closed"}'))
        node = _repeat_node(llm)

        result = await node._run({"query": "hi"})

        assert result == {"result": "closed"}
        assert llm.calls[-1]["tools"] is None

    async def test_a_useful_batch_resets_the_count(self) -> None:
        same = tool_call("my_tool", call_id="s", q="same")
        other = tool_call("my_tool", call_id="n", q="other")
        llm = FakeLLM(same, same, other, same, final('{"result": "goes on"}'))
        node = _repeat_node(llm)

        result = await node._run({"query": "hi"})

        assert result == {"result": "goes on"}
        assert llm.calls[-1]["tools"] is not None

    async def test_closing_message_does_not_lie_or_blame_connectors(self) -> None:
        same = tool_call("my_tool", call_id="s", q="same")
        llm = FakeLLM(same, same, same, final('{"result": "closed"}'))
        node = _repeat_node(llm)

        await node._run({"query": "hi"})

        instruction = llm.sent_text(3)
        assert "You have all the data you need" not in instruction
        assert "permissions" in instruction
        assert "connectors" in instruction
        assert "remains" in instruction

    async def test_both_exits_share_the_guidance(self) -> None:
        llm = FakeLLM(
            tool_call(call_id="1", q="1"),
            tool_call(call_id="2", q="2"),
            final('{"result": "exhausted"}'),
        )
        cfg = make_config(tool_loop=NodeToolLoopConfig(max_iterations=2))
        node = make_node(llm, config=cfg, registry=FakeRegistry([spec()]))

        await node._run({"query": "hi"})

        instruction = llm.sent_text(2)
        assert LLMNode._FINAL_TURN_GUIDANCE in instruction
        assert "maximum number of tool-call iterations" in instruction

    async def test_streaming_path_has_the_same_contract(self) -> None:
        same = tool_call("my_tool", call_id="s", q="same")
        llm = FakeLLM(same, same, final('{"result": "recovered"}'))
        node = _repeat_node(llm)

        async for _ in node._stream_run({"query": "hi"}):
            pass

        assert node._last_stream_output == {"result": "recovered"}
        assert llm.calls[-1]["tools"] is not None

    async def test_streaming_path_closes_at_second_batch(self) -> None:
        same = tool_call("my_tool", call_id="s", q="same")
        llm = FakeLLM(same, same, same, final('{"result": "closed"}'))
        node = _repeat_node(llm)

        async for _ in node._stream_run({"query": "hi"}):
            pass

        assert node._last_stream_output == {"result": "closed"}
        instruction = llm.sent_text(3)
        assert "You have all the data you need" not in instruction
        assert "connectors" in instruction

    async def test_streaming_budget_exhaustion_exits_gracefully(self) -> None:
        llm = FakeLLM(tool_call(call_id="a", f="a"), final('{"result": "partial"}'))
        cfg = make_config(
            system_prompt="S" * 320_000,
            tool_loop=NodeToolLoopConfig(
                max_iterations=10, max_tool_loop_tokens=96_000, tool_loop_token_reserve=16_384
            ),
        )
        node = make_node(llm, config=cfg, registry=FakeRegistry([spec()], ["short"]))

        async for _ in node._stream_run({"query": "hi"}):
            pass

        assert node._last_stream_output == {"result": "partial"}
        assert LLMNode._FINAL_TURN_MAX_ITERATIONS in llm.sent_text(1)


# ---------------------------------------------------------------------------
# Tool trace on the streaming path
# ---------------------------------------------------------------------------


class TestToolTrace:
    async def test_streaming_path_logs_tool_names_but_not_arguments(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        llm = FakeLLM(tool_call("read", call_id="a", path="/secret/user-file.csv"), final())
        node = make_node(llm, registry=FakeRegistry([spec()]))

        with caplog.at_level(logging.INFO, logger="sofias_sdk_lite.nodes.llm_node"):
            async for _ in node._stream_run({"query": "hi"}):
                pass

        trace = [
            r.getMessage() for r in caplog.records if "Tool calls in iteration" in r.getMessage()
        ]
        assert len(trace) == 1
        assert "'read'" in trace[0]
        assert "user-file.csv" not in trace[0]

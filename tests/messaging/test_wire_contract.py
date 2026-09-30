"""Wire-contract pinning: these field names/shapes are the interop contract
with any platform that publishes/consumes these messages. Do not casually
rename fields — a change here is a breaking wire change.
"""

from __future__ import annotations

from sofias_sdk_lite.messaging import AgentTaskMessage, StreamFragment


def test_agent_task_message_required_fields():
    task = AgentTaskMessage(
        content="hello",
        agent="my_agent",
        tenant_identifier="tenant-1",
        conversation_id="conv-1",
        message_id="msg-1",
    )
    dumped = task.model_dump(mode="json")

    for field in (
        "content",
        "agent",
        "agent_configuration",
        "mcp_servers",
        "tenant_identifier",
        "conversation_id",
        "message_id",
        "reply_to",
        "correlation_id",
    ):
        assert field in dumped, f"AgentTaskMessage lost wire field: {field}"

    assert dumped["content"] == "hello"
    assert dumped["agent"] == "my_agent"
    assert dumped["tenant_identifier"] == "tenant-1"
    assert dumped["agent_configuration"] == {}
    assert dumped["reply_to"] is None


def test_stream_fragment_required_fields_and_defaults():
    fragment = StreamFragment(
        conversation_id="conv-1",
        message_id="msg-1",
        content="partial answer",
        agent="my_agent",
    )
    dumped = fragment.model_dump(mode="json")

    for field in (
        "conversation_id",
        "message_id",
        "content",
        "agent",
        "status",
        "complete",
        "fragment_index",
        "is_last",
    ):
        assert field in dumped, f"StreamFragment lost wire field: {field}"

    assert dumped["status"] == "streaming"
    assert dumped["complete"] is False
    assert dumped["fragment_index"] == 0
    assert dumped["is_last"] is False


def test_stream_fragment_sanitizes_html_content():
    fragment = StreamFragment(
        conversation_id="conv-1",
        message_id="msg-1",
        content="<script>alert(1)</script>hello",
        agent="my_agent",
    )
    assert "<script>" not in fragment.content

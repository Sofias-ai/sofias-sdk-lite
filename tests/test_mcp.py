"""MCP connectors delivered by the Sofias platform.

The platform hands each turn's connectors to the agent inside
``AgentTaskMessage.mcp_servers``. For marketplace agents every entry points to
the platform's MCP intermediary with a short-lived grant in its headers: the
agent never sees the user's real credentials. ``McpToolProvider`` turns those
entries into tools an ``LLMNode`` can call.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from sofias_sdk_lite.mcp import McpError, McpToolProvider, mcp_servers_from_message
from sofias_sdk_lite.messaging.models import MCPServerConfig


def _entry(name: str = "notion", **conf) -> MCPServerConfig:
    base = {
        "__mcp_url": f"http://platform-gateway:3000/mcp/{name}",
        "__mcp_headers": {"Authorization": "Bearer grant-123"},
        "__mcp_namespace": True,
    }
    return MCPServerConfig(name=name, address="platform-gateway", port=3000, configuration={**base, **conf})


class _FakeServer:
    """A Streamable HTTP MCP server, answering JSON or event streams."""

    def __init__(self, tools: list[dict], *, event_stream: bool = False) -> None:
        self.tools = tools
        self.event_stream = event_stream
        self.requests: list[tuple[dict, httpx.Headers]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((body, request.headers))
        method = body.get("method")
        if "id" not in body:
            return httpx.Response(202)
        if method == "initialize":
            result = {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {"name": "x"}}
            return self._reply(body["id"], result, session="ses-1")
        if method == "tools/list":
            return self._reply(body["id"], {"tools": self.tools})
        if method == "tools/call":
            name = body["params"]["name"]
            if name == "fails":
                error = {"jsonrpc": "2.0", "id": body["id"], "error": {"code": -32003, "message": "tool not allowed"}}
                return httpx.Response(200, json=error)
            if name == "summary_only":
                result = {"content": [{"type": "text", "text": "Found 2 project(s)."}],
                          "structuredContent": {"projects": ["Alpha", "Beta"]}}
                return self._reply(body["id"], result)
            text = f"ran {name} with {json.dumps(body['params'].get('arguments', {}))}"
            return self._reply(body["id"], {"content": [{"type": "text", "text": text}]})
        return httpx.Response(400)

    def _reply(self, id_: int, result: dict, session: str | None = None) -> httpx.Response:
        message = {"jsonrpc": "2.0", "id": id_, "result": result}
        headers = {"Mcp-Session-Id": session} if session else {}
        if self.event_stream:
            stream = 'event: message\ndata: {"jsonrpc":"2.0","method":"notifications/progress"}\n\n'
            stream += f"event: message\ndata: {json.dumps(message)}\n\n"
            return httpx.Response(200, text=stream, headers={**headers, "Content-Type": "text/event-stream"})
        return httpx.Response(200, json=message, headers=headers)


_TOOLS = [
    {"name": "notion-search", "description": "Search pages", "inputSchema": {"type": "object"}},
    {"name": "notion-fetch", "description": "Fetch a page", "inputSchema": {"type": "object"}},
]


def _provider(server: _FakeServer, *entries: MCPServerConfig) -> McpToolProvider:
    return McpToolProvider(list(entries) or [_entry()], transport=httpx.MockTransport(server))


def test_tools_carry_the_connector_in_front():
    provider = _provider(_FakeServer(_TOOLS))

    tools = asyncio.run(provider.get_tools("assistant"))

    assert [t.name for t in tools] == ["notion__notion-search", "notion__notion-fetch"]
    assert tools[0].description == "Search pages"


def test_calls_go_to_the_real_tool_with_the_grant_and_the_session():
    server = _FakeServer(_TOOLS)
    provider = _provider(server)

    async def run() -> str:
        await provider.get_tools("assistant")
        return await provider.execute_tool("notion__notion-search", {"query": "plan"})

    result = asyncio.run(run())

    assert result == 'ran notion-search with {"query": "plan"}'
    call, headers = server.requests[-1]
    assert call["params"]["name"] == "notion-search"
    assert headers["authorization"] == "Bearer grant-123"
    assert headers["mcp-session-id"] == "ses-1"


def test_event_stream_answers_are_understood():
    provider = _provider(_FakeServer(_TOOLS, event_stream=True))

    tools = asyncio.run(provider.get_tools("assistant"))

    assert len(tools) == 2


def test_the_allow_list_is_respected():
    provider = _provider(_FakeServer(_TOOLS), _entry(__mcp_tools=["notion-fetch"]))

    tools = asyncio.run(provider.get_tools("assistant"))

    assert [t.name for t in tools] == ["notion__notion-fetch"]


def test_a_server_error_is_reported_with_its_message():
    server = _FakeServer([*_TOOLS, {"name": "fails", "inputSchema": {}}])
    provider = _provider(server)

    async def run() -> None:
        await provider.get_tools("assistant")
        await provider.execute_tool("notion__fails", {})

    with pytest.raises(McpError, match="tool not allowed"):
        asyncio.run(run())


def test_an_unreachable_connector_does_not_hide_the_others():
    def handler(request: httpx.Request) -> httpx.Response:
        if "/mcp/broken" in str(request.url):
            raise httpx.ConnectError("down")
        return _FakeServer(_TOOLS)(request)

    provider = McpToolProvider([_entry("broken"), _entry("notion")], transport=httpx.MockTransport(handler))

    tools = asyncio.run(provider.get_tools("assistant"))

    assert {t.name for t in tools} == {"notion__notion-search", "notion__notion-fetch"}


def test_entries_without_platform_url_use_the_internal_address():
    [server] = mcp_servers_from_message([MCPServerConfig(name="boe", address="mcp-boe", port=22005)])

    assert server.url == "http://mcp-boe:22005/mcp"
    assert server.namespace is False


def test_structured_content_reaches_the_model_behind_a_summary():
    """Some servers put only a summary in the text and the data in
    ``structuredContent``: without it the model only sees counters."""
    server = _FakeServer([*_TOOLS, {"name": "summary_only", "inputSchema": {}}])
    provider = _provider(server)

    async def run() -> str:
        await provider.get_tools("assistant")
        return await provider.execute_tool("notion__summary_only", {})

    result = asyncio.run(run())

    assert result.startswith("Found 2 project(s).")
    assert json.loads(result.split("\n", 1)[1]) == {"projects": ["Alpha", "Beta"]}


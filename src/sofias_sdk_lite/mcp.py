"""MCP connectors delivered by the Sofias platform.

The platform sends each turn's connectors in ``AgentTaskMessage.mcp_servers``.
Inside ``configuration`` it puts the transport details under ``__mcp_*`` keys:

- ``__mcp_url``: the endpoint to call. For marketplace agents it is always the
  platform's MCP intermediary, which adds the user's real credentials itself.
- ``__mcp_headers``: headers for every request (a short-lived grant). Never put
  them in a prompt or a log.
- ``__mcp_tools``: the allow list approved for that connector.
- ``__mcp_namespace``: offer the tools as ``connector__tool`` so two connectors
  with a tool of the same name do not clash.

``McpToolProvider`` implements the ``ToolProvider`` protocol on top of those
entries with a minimal Streamable HTTP client (JSON-RPC over ``httpx``, JSON or
event-stream answers, ``Mcp-Session-Id``), so it needs no extra dependency::

    from sofias_sdk_lite.mcp import McpToolProvider
    from sofias_sdk_lite.tools import ToolRegistry

    provider = McpToolProvider(task.mcp_servers)
    builder.with_tool_registry(ToolRegistry(provider))
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from sofias_sdk_lite.llm.tools import ToolSpec
from sofias_sdk_lite.observability._log import get_logger

logger = get_logger("mcp")

__all__ = ["McpError", "McpServer", "McpToolProvider", "mcp_servers_from_message"]

PROTOCOL_VERSION = "2025-06-18"
_UNSUPPORTED = re.compile(r"[^A-Za-z0-9_-]")


class McpError(Exception):
    """An MCP server answered with an error or could not be reached."""


@dataclass(frozen=True)
class McpServer:
    """One connector of the turn, ready to call."""

    name: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    tools: tuple[str, ...] | None = None
    namespace: bool = False


def _field(raw: Any, name: str) -> Any:
    return raw.get(name) if isinstance(raw, dict) else getattr(raw, name, None)


def mcp_servers_from_message(raw_servers: list[Any]) -> list[McpServer]:
    """Read the ``mcp_servers`` of a task message (models or plain dicts)."""
    servers: list[McpServer] = []
    for raw in raw_servers:
        conf = dict(_field(raw, "conf") or _field(raw, "configuration") or {})
        tools = conf.get("__mcp_tools")
        servers.append(
            McpServer(
                name=str(_field(raw, "name")),
                url=conf.get("__mcp_url") or f"http://{_field(raw, 'address')}:{_field(raw, 'port')}/mcp",
                headers=dict(conf.get("__mcp_headers") or {}),
                tools=tuple(tools) if tools else None,
                namespace=bool(conf.get("__mcp_namespace") or conf.get("__mcp_headers")),
            )
        )
    return servers


def _offered_name(server: McpServer, tool: str) -> str:
    if not server.namespace:
        return tool
    return _UNSUPPORTED.sub("_", f"{server.name}__{tool}")[:64]


class _Session:
    """A minimal Streamable HTTP MCP session against one server."""

    def __init__(self, server: McpServer, client: httpx.AsyncClient) -> None:
        self._server = server
        self._client = client
        self._session_id: str | None = None
        self._next_id = 0
        self._initialized = False

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        headers = {
            **self._server.headers,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": PROTOCOL_VERSION,
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        try:
            response = await self._client.post(self._server.url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise McpError(f"{self._server.name}: unreachable ({type(exc).__name__})") from exc
        self._session_id = response.headers.get("Mcp-Session-Id") or self._session_id
        if response.status_code == 202 or "id" not in payload:
            return None
        if response.status_code >= 400:
            raise McpError(f"{self._server.name}: HTTP {response.status_code}")
        message = self._message(response, payload["id"])
        if message.get("error"):
            raise McpError(f"{self._server.name}: {message['error'].get('message', message['error'])}")
        return message.get("result") or {}

    @staticmethod
    def _message(response: httpx.Response, id_: int) -> dict[str, Any]:
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            return response.json()
        messages = []
        for event in re.split(r"\r?\n\r?\n", response.text):
            data = "\n".join(line[5:].strip() for line in event.splitlines() if line.startswith("data:"))
            if data:
                try:
                    messages.append(json.loads(data))
                except json.JSONDecodeError:
                    continue
        for message in messages:
            if message.get("id") == id_:
                return message
        raise McpError("event stream without an answer")

    async def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._initialized:
            await self._initialize()
        return await self._call(method, params) or {}

    async def _call(self, method: str, params: dict[str, Any] | None) -> dict[str, Any] | None:
        self._next_id += 1
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            payload["params"] = params
        return await self._post(payload)

    async def _initialize(self) -> None:
        await self._call(
            "initialize",
            {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
             "clientInfo": {"name": "sofias-sdk-lite", "version": "1"}},
        )
        await self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self._initialized = True


class McpToolProvider:
    """``ToolProvider`` over the connectors the platform sent for this turn.

    Connects lazily on the first ``get_tools``. A connector that cannot be
    reached is skipped (and logged) so the others keep working.
    """

    def __init__(self, raw_servers: list[Any], *, transport: httpx.AsyncBaseTransport | None = None,
                 timeout: float = 60.0) -> None:
        self._servers = mcp_servers_from_message(raw_servers)
        self._client = httpx.AsyncClient(transport=transport, timeout=timeout)
        self._sessions = {server.name: _Session(server, self._client) for server in self._servers}
        self._routes: dict[str, tuple[str, str]] = {}
        self._specs: list[ToolSpec] | None = None

    async def get_tools(self, node_name: str) -> list[ToolSpec]:
        if self._specs is None:
            self._specs = []
            for server in self._servers:
                try:
                    listed = await self._list(server)
                except McpError as exc:
                    logger.warning("mcp_server_unavailable", server=server.name, error=str(exc))
                    continue
                self._specs.extend(listed)
        return list(self._specs)

    async def _list(self, server: McpServer) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        cursor: str | None = None
        while True:
            result = await self._sessions[server.name].request("tools/list", {"cursor": cursor} if cursor else None)
            for tool in result.get("tools", []):
                name = tool.get("name")
                if not name or (server.tools is not None and name not in server.tools):
                    continue
                offered = _offered_name(server, name)
                self._routes[offered] = (server.name, name)
                specs.append(ToolSpec(name=offered, description=tool.get("description") or name,
                                      parameters_schema=tool.get("inputSchema") or {"type": "object"}))
            cursor = result.get("nextCursor")
            if not cursor:
                return specs

    async def execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> Any:
        if self._specs is None:
            await self.get_tools("")
        route = self._routes.get(tool_name)
        if route is None:
            raise McpError(f"unknown MCP tool '{tool_name}'")
        server_name, real_name = route
        result = await self._sessions[server_name].request(
            "tools/call", {"name": real_name, "arguments": tool_input or {}}
        )
        if result.get("isError"):
            raise McpError(_text(result) or f"{tool_name} failed")
        return _with_structured(result)

    async def aclose(self) -> None:
        await self._client.aclose()


def _text(result: dict[str, Any]) -> str:
    return "\n".join(block.get("text", "") for block in result.get("content", []) if block.get("type") == "text")


def _with_structured(result: dict[str, Any]) -> str:
    """The text of a tool result plus its ``structuredContent``, unless the text
    already is that same JSON. Some servers only put a summary in the text."""
    text = _text(result)
    structured = result.get("structuredContent")
    if not structured:
        return text if result.get("content") else json.dumps(result)
    for block in result.get("content", []):
        try:
            if json.loads(block.get("text", "")) == structured:
                return text
        except (TypeError, ValueError):
            continue
    serialized = json.dumps(structured)
    return f"{text}\n{serialized}" if text else serialized

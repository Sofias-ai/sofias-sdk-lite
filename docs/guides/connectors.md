# Using the user's connectors (MCP)

When a user who has connected their tools (mail, documents, CRM, their company's
own MCP servers…) talks to your agent, the platform sends those connectors with
every turn in `AgentTaskMessage.mcp_servers`.

Your agent **never receives the user's credentials**. Each entry points to the
platform's MCP intermediary with a short-lived grant (about 20 minutes, one
turn). The intermediary adds the real credential on every call, only exposes the
tools approved for third-party agents on that connector, and records each call
(never its content). If the user disconnects an account, the grant stops working
for it at once.

Install version 0.3.0 or later: `pip install "sofias-sdk-lite>=0.3.0"`.

## Which connectors your agent receives

On every turn, `mcp_servers` carries:

- the connectors **the user talking to your agent** has connected in Sofias
  (mail, documents, CRM, project management…);
- the MCP servers **of the user's company** ("Own MCPs"): those the company
  connects once for everyone, and those each person connects with their own
  account or key.

Of each connector, your agent only gets the tools that the person who manages
it (the Sofias catalogue, or the company administrator for its own MCPs) marked
for third-party agents. The list can be empty: your agent must still work
without connectors.

## Exposing the connectors to an LLM node

```python
from sofias_sdk_lite.mcp import McpToolProvider
from sofias_sdk_lite.tools import ToolRegistry

provider = McpToolProvider(task.mcp_servers)   # connects on first use
builder.with_tool_registry(ToolRegistry(provider))
```

Tools are offered as `connector__tool` (for example `notion__notion-search`), so
two connectors with a tool of the same name never clash. A connector that cannot
be reached is skipped and the others keep working.

## What each entry contains

| Key in `configuration` | Meaning |
| --- | --- |
| `__mcp_url` | Endpoint to call (the intermediary). |
| `__mcp_headers` | Headers for every request: the grant. Never log them or put them in a prompt. |
| `__mcp_tools` | Allow list of tool names approved for this connector. |
| `__mcp_namespace` | The tools are offered with the connector name in front. |

## Testing with connectors in the developer portal

The sandbox gives your agent the connectors you choose, exactly as in
production: through the intermediary, never your credentials.

1. In Sofias, open **My connectors** and, under **Developer portal**, press
   **Use my connectors in tests**. Your developer account and your Sofias
   account must use the same email address.
2. In the developer portal, open your agent version, press **Test in Sandbox**
   and start the sandbox.
3. Under **MCPs to test with**, tick the connectors to use and press **Save**.
   They are used from the next message.

## Adding your own MCP servers

If your company uses Sofias, its administrator adds the company's MCP servers
in Sofias, under **Configuration** > **Own MCPs** > **New connector**. They
then reach every agent of the company, yours included, as described above.

What the server needs:

- a public HTTPS endpoint (private and internal addresses are rejected);
- the MCP Streamable HTTP transport;
- one way to authenticate: OAuth (with dynamic client registration, or with
  your own client ID and secret), a key for the whole company, a key or a URL
  per user, or none.

Before anything reaches an agent, the platform measures every tool and reviews
its description. The administrator approves which tools agents may use, blocks
the ones they must not (destructive tools are blocked by default) and marks the
ones third-party agents receive. If the server later changes an approved tool,
the connector goes back to review and the approved version keeps being used.

## Network access

Marketplace agents reach the internet only through the egress proxy and only
the hosts declared in `egress_hosts`. The intermediary is reached directly: you
do not need to declare it, nor the servers behind your connectors.

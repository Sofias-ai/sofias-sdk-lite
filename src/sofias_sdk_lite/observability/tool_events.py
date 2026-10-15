"""One line per tool call, for the developer portal's sandbox logs.

Every tool the model calls -- your own or a connector's -- leaves one line on
standard output, whatever your logging configuration::

    SOFIAS_EVENT {"type":"tool_call","tool":"search","server":"notion","ok":true,"duration_ms":412,...}

The portal reads these lines to show which tools your agent called. They carry
names, timings and outcome only: never the arguments or the results, which may
hold your users' data. Set ``SOFIAS_EVENTS=off`` to silence them.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

__all__ = ["EVENT_PREFIX", "emit_tool_call"]

#: Marks the lines the portal parses; everything else in the log is shown raw.
EVENT_PREFIX = "SOFIAS_EVENT"

_logger = logging.getLogger("sofias_sdk_lite.events")


class _StdoutHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Writes to whatever ``sys.stdout`` is at emit time (it may be swapped later)."""

    @property  # type: ignore[override]
    def stream(self) -> Any:
        return sys.stdout

    @stream.setter
    def stream(self, _value: Any) -> None:
        pass


def _ensure_handler() -> None:
    """Own stdout handler, so the line appears even without logging configured."""
    if _logger.handlers:
        return
    handler = _StdoutHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def emit_tool_call(
    tool_name: str,
    *,
    ok: bool,
    duration_ms: float,
    node_name: str | None = None,
    error_type: str | None = None,
) -> None:
    """Write the event line for one tool call (no arguments, no results)."""
    if os.environ.get("SOFIAS_EVENTS", "").lower() in {"off", "0", "false"}:
        return
    server, separator, tool = tool_name.partition("__")
    if not separator:
        server, tool = "", tool_name
    event: dict[str, Any] = {
        "type": "tool_call",
        "tool": tool,
        "server": server or None,
        "ok": ok,
        "duration_ms": round(duration_ms),
        "node": node_name,
        "error_type": error_type,
        "ts": round(time.time(), 3),
    }
    _ensure_handler()
    _logger.info(
        "%s %s", EVENT_PREFIX, json.dumps(event, separators=(",", ":"), ensure_ascii=False)
    )

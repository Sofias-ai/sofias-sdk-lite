"""Minimal structured-logging shim over the stdlib ``logging`` module.

This SDK has no dependency on any third-party structured-logging library.
This module provides a small adapter with a keyword-argument call shape
(``logger.info("message", key=value, ...)``) backed by the stdlib logger.
Bring your own ``logging.Handler`` / formatter to route these anywhere
(stdout, a file, an OpenTelemetry log exporter, etc.).
"""

from __future__ import annotations

import logging as _logging
from typing import Any


class _StructuredLogger:
    """Adapts stdlib ``logging.Logger`` to accept structured keyword context."""

    def __init__(self, name: str) -> None:
        self._logger = _logging.getLogger(name)

    def _emit(self, level: int, message: str, **context: Any) -> None:
        if context:
            context_str = " ".join(f"{k}={v!r}" for k, v in context.items())
            message = f"{message} ({context_str})"
        self._logger.log(level, message)

    def debug(self, message: str, **context: Any) -> None:
        self._emit(_logging.DEBUG, message, **context)

    def info(self, message: str, **context: Any) -> None:
        self._emit(_logging.INFO, message, **context)

    def warning(self, message: str, **context: Any) -> None:
        self._emit(_logging.WARNING, message, **context)

    def error(self, message: str, **context: Any) -> None:
        self._emit(_logging.ERROR, message, **context)


def get_logger(name: str) -> _StructuredLogger:
    """Return a structured logger namespaced under ``sofias_sdk_lite.<name>``."""
    return _StructuredLogger(f"sofias_sdk_lite.{name}")

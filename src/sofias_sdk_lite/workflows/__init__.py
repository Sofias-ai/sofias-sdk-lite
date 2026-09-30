"""Response workflows: how an agent delivers its output.

``ResponseWorkflow``/``StreamingResponseWorkflow``/``BaseDelegationResponseWorkflow``
are protocol-level and live in ``sofias_sdk_lite.agent`` (they're part of the
agent execution contract). This package adds concrete, transport-specific
implementations: ``ChatResponseWorkflow`` (RabbitMQ streams, chunks the final
answer), ``BaseStreamingResponseWorkflow`` (RabbitMQ streams, token-by-token
via ``on_stream_event``; subclass and override its hooks) and ``NullWorkflow``
(tests/local dev).
"""

from __future__ import annotations

from sofias_sdk_lite.agent.response_workflow import (
    BaseDelegationResponseWorkflow,
    PublishFn,
    ResponseWorkflow,
    StreamingResponseWorkflow,
)
from sofias_sdk_lite.workflows.chat import EMPTY_RESPONSE_FALLBACK, ChatResponseWorkflow
from sofias_sdk_lite.workflows.null import NullWorkflow
from sofias_sdk_lite.workflows.streaming_chat import BaseStreamingResponseWorkflow

__all__ = [
    "ResponseWorkflow",
    "StreamingResponseWorkflow",
    "BaseDelegationResponseWorkflow",
    "PublishFn",
    "ChatResponseWorkflow",
    "BaseStreamingResponseWorkflow",
    "EMPTY_RESPONSE_FALLBACK",
    "NullWorkflow",
]

"""RabbitMQ transport layer: client, consumer, publisher, RPC, and delegation."""

from sofias_sdk_lite.nodes.delegation_transport import NullDelegationTransport

from .client import RabbitMQClient
from .config import RabbitMQConfig
from .consumer import MessageHandler, RabbitMQConsumer
from .delegation_transport import DelegationTransportError, RabbitMQDelegationTransport
from .publisher import Exchange, Producer, PublishOptions, RabbitMQPublisher, publish_reply
from .rpc_client import RabbitMQRPCClient, RPCError, RPCTimeoutError
from .types import AckPolicy, MessageContext, with_traceparent

__all__ = [
    "RabbitMQClient",
    "RabbitMQConfig",
    "RabbitMQConsumer",
    "MessageHandler",
    "RabbitMQPublisher",
    "publish_reply",
    "Exchange",
    "Producer",
    "RabbitMQRPCClient",
    "RPCError",
    "RPCTimeoutError",
    # Delegation
    "RabbitMQDelegationTransport",
    "DelegationTransportError",
    "NullDelegationTransport",
    # Types
    "AckPolicy",
    "MessageContext",
    "PublishOptions",
    "with_traceparent",
]

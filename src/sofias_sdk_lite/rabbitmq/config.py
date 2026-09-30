"""RabbitMQ connection configuration."""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RabbitMQConfig"]


@dataclass
class RabbitMQConfig:
    """Configuration for RabbitMQ connections (AMQP and Streams).

    Only connection-level settings belong here.
    Topology parameters (queue_name, exchange_name, routing_key, stream_name)
    belong in the constructors of Consumer / Publisher.
    """

    host: str = "127.0.0.1"
    amqp_port: int = 5672
    stream_port: int = 5552
    username: str = "guest"
    password: str = "guest"
    virtual_host: str = "/"
    prefetch_count: int = 1
    heartbeat: int = 60
    reconnect_interval: float = 5.0
    shutdown_timeout: float = 5.0

    @property
    def amqp_url(self) -> str:
        """Build an AMQP URL from the individual fields."""
        vhost = self.virtual_host if self.virtual_host.startswith("/") else f"/{self.virtual_host}"
        return f"amqp://{self.username}:{self.password}@{self.host}:{self.amqp_port}{vhost}"

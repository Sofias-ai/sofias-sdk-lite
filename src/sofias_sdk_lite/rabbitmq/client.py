"""RabbitMQ client helper built on top of aio-pika."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import aio_pika

from sofias_sdk_lite.observability._log import get_logger

from .config import RabbitMQConfig

__all__ = ["RabbitMQClient"]

logger = get_logger("rabbitmq.client")


class RabbitMQClient:
    """Manages the connection and channel lifecycle for RabbitMQ.

    Accepts a :class:`RabbitMQConfig` and supports usage as an async
    context manager::

        async with RabbitMQClient(config) as client:
            ...
    """

    def __init__(self, config: RabbitMQConfig) -> None:
        self._config = config
        self._url = config.amqp_url
        self._reconnect_interval = config.reconnect_interval
        self._prefetch_count = config.prefetch_count
        self._connection: Optional[aio_pika.RobustConnection] = None
        self._channel: Optional[aio_pika.RobustChannel] = None
        self._lock = asyncio.Lock()

    @property
    def config(self) -> RabbitMQConfig:
        """The configuration used to create this client."""
        return self._config

    async def connect(self) -> None:
        async with self._lock:
            if self._connection and not self._connection.is_closed:
                return

            logger.info("rabbitmq_connect_start", url=self._url)
            self._connection = await aio_pika.connect_robust(self._url)
            self._channel = await self._connection.channel()
            await self._channel.set_qos(prefetch_count=self._prefetch_count)
            logger.info("rabbitmq_connected", url=self._url)

    async def disconnect(self) -> None:
        async with self._lock:
            if self._channel and not self._channel.is_closed:
                await self._channel.close()
            if self._connection and not self._connection.is_closed:
                await self._connection.close()
            self._channel = None
            self._connection = None
            logger.info("rabbitmq_disconnected", url=self._url)

    async def ensure_connected(self) -> None:
        if self._connection and not self._connection.is_closed:
            return

        attempt = 0
        while True:
            attempt += 1
            try:
                await self.connect()
                return
            except Exception as exc:  # pragma: no cover - retry loop
                logger.warning(
                    "rabbitmq_connect_failed",
                    url=self._url,
                    attempt=attempt,
                    error=str(exc),
                )
                await asyncio.sleep(self._reconnect_interval)

    @property
    def channel(self) -> aio_pika.RobustChannel:
        if not self._channel or self._channel.is_closed:
            raise RuntimeError("RabbitMQ channel is not available")
        return self._channel

    @asynccontextmanager
    async def acquire_channel(self) -> AsyncIterator[aio_pika.RobustChannel]:
        await self.ensure_connected()
        try:
            yield self.channel
        finally:
            # Channels are persistent; we keep them open for reuse.
            pass

    # -- Async context manager ------------------------------------------------

    async def __aenter__(self) -> "RabbitMQClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

"""AggregatorNode for collecting delegation responses (fan-in).

This module provides the AggregatorNode class, which waits for responses
from previously dispatched delegations and resolves them according to
configurable policies (all, any, majority).

The node receives a list of correlation IDs via its input and uses the
DelegationTransport to collect responses without sending any new requests.
"""

from __future__ import annotations

import math
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from sofias_sdk_lite.errors import AggregationTimeoutError
from sofias_sdk_lite.nodes.aggregator_config import AggregatorNodeConfig
from sofias_sdk_lite.nodes.base_node import BaseNode, NodeType
from sofias_sdk_lite.observability._log import get_logger

if TYPE_CHECKING:
    from sofias_sdk_lite.agent import StreamEvent
    from sofias_sdk_lite.contracts import NodeContract
    from sofias_sdk_lite.nodes.delegation_transport import DelegationTransport

logger = get_logger("nodes.aggregator_node")

__all__ = ["AggregatorNode", "RetryHandler"]

RetryHandler = Callable[[list[str]], Awaitable[list[str]]]
"""Async callable that receives missing correlation IDs and re-dispatches work.
Returns the new correlation IDs to wait for (may be the same or different)."""


class AggregatorNode(BaseNode):
    """Node that collects responses from previously dispatched delegations.

    AggregatorNode implements a fan-in pattern: given a list of correlation IDs
    (from prior DelegationNode sends or any other source), it waits for responses
    via the DelegationTransport and resolves according to a configurable policy.

    Resolution policies:
    - 'all': Wait until every expected response arrives.
    - 'any': Resolve as soon as at least one response arrives.
    - 'majority': Resolve when more than floor(expected * threshold) responses arrive.

    Retry support:
    When on_timeout='retry_missing', the node calls the retry_handler with the
    list of unresponsive correlation IDs. The handler re-dispatches work and
    returns the new IDs to wait for. Each retry gets its own timeout window
    controlled by retry_timeout_seconds (or timeout_seconds if not set).

    Example:
        config = AggregatorNodeConfig(
            name="collect_results",
            resolution_policy="all",
            timeout_seconds=60.0,
        )

        node = AggregatorNode(
            config=config,
            contract=my_contract,
            transport=my_transport,
        )

        result = await node.execute({"pending_ids": ["corr-1", "corr-2"]})
        # result = {
        #     "results": {"corr-1": {...}, "corr-2": {...}},
        #     "pending": [],
        #     "is_partial": False,
        #     "total_expected": 2,
        #     "total_received": 2,
        # }
    """

    def __init__(
        self,
        config: AggregatorNodeConfig,
        contract: NodeContract,
        transport: DelegationTransport,
        retry_handler: RetryHandler | None = None,
    ) -> None:
        """Initialize the AggregatorNode.

        Args:
            config: Configuration for the aggregator node.
            contract: Input/output contract for validation.
            transport: Transport for receiving responses.
            retry_handler: Async callable for retry_missing policy. Receives
                list of missing correlation IDs, returns new IDs to wait for.
                Required when config.on_timeout='retry_missing'.
        """
        super().__init__(
            name=config.name,
            contract=contract,
            description=config.description,
        )
        self._config = config
        self._transport = transport
        self._retry_handler = retry_handler
        self._last_stream_output: dict[str, Any] | None = None

    @property
    def node_type(self) -> NodeType:
        return NodeType.AGGREGATOR

    @property
    def config(self) -> AggregatorNodeConfig:
        """The node's configuration."""
        return self._config

    def _is_resolved(self, total_expected: int, total_received: int) -> bool:
        """Check if the resolution policy is satisfied."""
        policy = self._config.resolution_policy

        if policy == "all":
            return total_received >= total_expected
        elif policy == "any":
            return total_received >= 1
        elif policy == "majority":
            threshold = math.floor(total_expected * self._config.majority_threshold)
            return total_received > threshold
        else:
            return total_received >= total_expected

    def _build_output(
        self,
        results: dict[str, dict[str, Any]],
        remaining_ids: set[str],
        total_expected: int,
    ) -> dict[str, Any]:
        """Build the standard output dictionary."""
        return {
            "results": results,
            "pending": sorted(remaining_ids),
            "is_partial": len(remaining_ids) > 0,
            "total_expected": total_expected,
            "total_received": len(results),
        }

    async def _collect_responses(
        self,
        results: dict[str, dict[str, Any]],
        remaining_ids: set[str],
        total_expected: int,
        timeout: float,
    ) -> bool:
        """Poll transport for responses until resolved or timeout.

        Args:
            results: Dict to populate with responses (mutated in place).
            remaining_ids: Set of IDs still pending (mutated in place).
            total_expected: Total expected for resolution check.
            timeout: Maximum seconds to poll in this phase.

        Returns:
            True if the resolution policy is satisfied.
        """
        start_time = time.time()
        poll_interval = self._config.poll_interval_seconds

        while not self._is_resolved(total_expected, len(results)):
            elapsed = time.time() - start_time
            time_left = timeout - elapsed

            if time_left <= 0:
                return False

            if not remaining_ids:
                break

            result = await self._transport.wait_any_response(
                correlation_ids=remaining_ids,
                timeout=min(time_left, poll_interval),
            )
            if result is None:
                continue

            correlation_id, response = result
            remaining_ids.discard(correlation_id)
            results[correlation_id] = response

            logger.debug(
                f"AggregatorNode '{self._name}' received response for "
                f"{correlation_id} ({len(results)}/{total_expected})"
            )

        return True

    async def _run(
        self,
        input_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the aggregation logic."""
        pending_ids_raw = input_data.get(self._config.pending_ids_field, [])
        pending_ids: list[str] = [str(pid) for pid in pending_ids_raw]

        remaining_ids = set(pending_ids)
        total_expected = len(remaining_ids)
        results: dict[str, dict[str, Any]] = {}

        if not remaining_ids:
            return self._build_output(results, remaining_ids, total_expected)

        retries_remaining = (
            self._config.max_retries
            if self._config.on_timeout == "retry_missing" and self._retry_handler
            else 0
        )

        # Initial collection phase
        resolved = await self._collect_responses(
            results, remaining_ids, total_expected, self._config.timeout_seconds
        )

        # Retry loop
        while not resolved and retries_remaining > 0 and remaining_ids:
            retries_remaining -= 1
            logger.info(
                f"AggregatorNode '{self._name}' retrying {len(remaining_ids)} "
                f"missing IDs (retries left: {retries_remaining})"
            )

            new_ids = await self._retry_handler(sorted(remaining_ids))  # type: ignore[misc]
            remaining_ids = set(new_ids)
            total_expected = len(results) + len(remaining_ids)

            retry_timeout = (
                self._config.retry_timeout_seconds
                if self._config.retry_timeout_seconds is not None
                else self._config.timeout_seconds
            )

            resolved = await self._collect_responses(
                results, remaining_ids, total_expected, retry_timeout
            )

        if resolved:
            return self._build_output(results, remaining_ids, total_expected)

        return self._handle_timeout(results, remaining_ids, total_expected)

    async def _stream_run(
        self,
        input_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute aggregation with streaming events."""
        from sofias_sdk_lite.agent import AggregationResponseEvent  # noqa: F401 (see _stream_collect)

        pending_ids_raw = input_data.get(self._config.pending_ids_field, [])
        pending_ids: list[str] = [str(pid) for pid in pending_ids_raw]

        remaining_ids = set(pending_ids)
        total_expected = len(remaining_ids)
        results: dict[str, dict[str, Any]] = {}

        if not remaining_ids:
            self._last_stream_output = self._build_output(
                results, remaining_ids, total_expected
            )
            return

        retries_remaining = (
            self._config.max_retries
            if self._config.on_timeout == "retry_missing" and self._retry_handler
            else 0
        )

        # Initial + retry phases with streaming
        phase_timeout = self._config.timeout_seconds
        resolved = False

        while True:
            async for event in self._stream_collect(
                results, remaining_ids, total_expected, phase_timeout
            ):
                yield event

            resolved = self._is_resolved(total_expected, len(results))

            if resolved:
                break

            if retries_remaining > 0 and remaining_ids and self._retry_handler:
                retries_remaining -= 1
                logger.info(
                    f"AggregatorNode '{self._name}' retrying {len(remaining_ids)} "
                    f"missing IDs (retries left: {retries_remaining})"
                )

                new_ids = await self._retry_handler(sorted(remaining_ids))
                remaining_ids = set(new_ids)
                total_expected = len(results) + len(remaining_ids)

                phase_timeout = (
                    self._config.retry_timeout_seconds
                    if self._config.retry_timeout_seconds is not None
                    else self._config.timeout_seconds
                )
            else:
                break

        if resolved:
            self._last_stream_output = self._build_output(
                results, remaining_ids, total_expected
            )
        else:
            self._last_stream_output = self._handle_timeout(
                results, remaining_ids, total_expected
            )

    async def _stream_collect(
        self,
        results: dict[str, dict[str, Any]],
        remaining_ids: set[str],
        total_expected: int,
        timeout: float,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Poll with streaming events for a single collection phase."""
        from sofias_sdk_lite.agent import AggregationResponseEvent

        start_time = time.time()
        poll_interval = self._config.poll_interval_seconds

        while not self._is_resolved(total_expected, len(results)):
            elapsed = time.time() - start_time
            time_left = timeout - elapsed

            if time_left <= 0:
                return

            if not remaining_ids:
                break

            result = await self._transport.wait_any_response(
                correlation_ids=remaining_ids,
                timeout=min(time_left, poll_interval),
            )
            if result is None:
                continue

            correlation_id, response = result
            remaining_ids.discard(correlation_id)
            results[correlation_id] = response

            is_resolved = self._is_resolved(total_expected, len(results))

            yield AggregationResponseEvent(
                node_name=self._name,
                correlation_id=correlation_id,
                total_expected=total_expected,
                total_received=len(results),
                is_resolved=is_resolved,
            )

    def _handle_timeout(
        self,
        results: dict[str, dict[str, Any]],
        remaining_ids: set[str],
        total_expected: int,
    ) -> dict[str, Any]:
        """Handle timeout according to on_timeout policy.

        For 'retry_missing', this is only called after all retries are exhausted,
        and it falls back to 'fail' behavior.
        """
        if self._config.on_timeout == "partial_result":
            logger.warning(
                f"AggregatorNode '{self._name}' timed out after "
                f"{self._config.timeout_seconds}s, returning partial result "
                f"({len(results)}/{total_expected})"
            )
            return self._build_output(results, remaining_ids, total_expected)

        # Both 'fail' and 'retry_missing' (after exhausting retries) raise
        raise AggregationTimeoutError(
            f"AggregatorNode '{self._name}' timed out after "
            f"{self._config.timeout_seconds}s waiting for "
            f"{self._config.resolution_policy} policy "
            f"({len(results)}/{total_expected} received)",
            timeout_seconds=self._config.timeout_seconds,
            total_expected=total_expected,
            total_received=len(results),
            pending_ids=sorted(remaining_ids),
            received_ids=sorted(results.keys()),
        )

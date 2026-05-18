# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import re
import time

from lmdeploy.serve.proxy.config import ProxyConfig
from lmdeploy.serve.proxy.node import Node, NodeRegistry
from lmdeploy.serve.proxy.routing.base import BaseStrategy
from lmdeploy.serve.proxy.routing.min_expected import MinExpectedLatencyStrategy
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

# Prometheus metric name: lmdeploy uses colon in name but Prometheus export
# replaces colons with underscores
_METRIC_RE = re.compile(
    r'^lmdeploy[:_]gpu_cache_usage_perc(?:\{[^}]*})?\s+([0-9.eE+-]+)',
    re.MULTILINE,
)


def parse_prometheus_cache_usage(text: str) -> float | None:
    """Parse lmdeploy:gpu_cache_usage_perc from Prometheus text exposition
    format.

    Returns the float value, or None if the metric is not found.
    """
    match = _METRIC_RE.search(text)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


class MinCacheUsageStrategy(BaseStrategy):
    """Route to the node with the lowest KV cache usage.

    Polls backend /metrics endpoints in a background task. Falls back to MinExpectedLatencyStrategy when no metrics data
    is available.
    """

    def __init__(self, registry: NodeRegistry, config: ProxyConfig):
        super().__init__(registry, config)
        self._fallback = MinExpectedLatencyStrategy(registry, config)
        self._poll_task: asyncio.Task | None = None

    async def select_node(self, model_name: str, role=None) -> Node:
        nodes = await self.registry.get(model_name, role=role)
        if not nodes:
            raise ValueError(f'No available node for model {model_name!r}')

        stale_threshold = time.time() - 3 * self.config.metrics_poll_interval
        valid_nodes = [
            n for n in nodes
            if n.cache_usage is not None
            and n.last_metrics_poll is not None
            and n.last_metrics_poll > stale_threshold
        ]

        if not valid_nodes:
            return await self._fallback.select_node(model_name, role=role)

        return min(valid_nodes, key=lambda n: n.cache_usage)

    async def start(self) -> None:
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

    async def _poll_loop(self) -> None:
        while True:
            try:
                await self._poll_all_nodes()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f'Error in metrics poll loop: {e}')
            await asyncio.sleep(self.config.metrics_poll_interval)

    async def _poll_all_nodes(self) -> None:
        if self.client is None or self.client.closed:
            return
        nodes = await self.registry.all_nodes()
        for node in nodes:
            try:
                async with self.client.get(f'{node.url}/metrics') as response:
                    if response.status == 200:
                        text = await response.text()
                        usage = parse_prometheus_cache_usage(text)
                        if usage is not None:
                            await self.registry.update_cache_usage(node.url, usage)
            except Exception as e:
                logger.debug(f'Failed to poll metrics from {node.url}: {e}')

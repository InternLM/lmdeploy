# Copyright (c) OpenMMLab. All rights reserved.

import random

from lmdeploy.serve.proxy.node import Node
from lmdeploy.serve.proxy.routing.base import BaseStrategy


class MinExpectedLatencyStrategy(BaseStrategy):
    """Select node with lowest expected latency (unfinished / speed)."""

    async def select_node(self, model_name: str, role=None) -> Node:
        nodes = await self.registry.get(model_name, role=role)
        if not nodes:
            raise ValueError(f"No available node for model {model_name!r}")

        speeds = [n.speed for n in nodes]
        with_speed = [s for s in speeds if s is not None]
        average_speed = sum(with_speed) / len(with_speed) if with_speed else 1
        resolved_speeds = [s if s is not None else average_speed for s in speeds]

        # Random traversal for load spreading in low-concurrency scenarios
        indices = list(range(len(nodes)))
        random.shuffle(indices)

        min_latency = float('inf')
        min_index = 0
        for i in indices:
            latency = nodes[i].unfinished / resolved_speeds[i]
            if min_latency > latency:
                min_latency = latency
                min_index = i
        return nodes[min_index]

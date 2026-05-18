# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np

from lmdeploy.serve.proxy.node import Node
from lmdeploy.serve.proxy.routing.base import BaseStrategy


class MinObservedLatencyStrategy(BaseStrategy):
    """Select node with lowest mean observed latency."""

    async def select_node(self, model_name: str, role=None) -> Node:
        nodes = await self.registry.get(model_name, role=role)
        if not nodes:
            raise ValueError(f"No available node for model {model_name!r}")

        latencies = []
        for node in nodes:
            if len(node.latency):
                latencies.append(float(np.mean(list(node.latency))))
            else:
                latencies.append(float('inf'))

        index = int(np.argmin(np.array(latencies)))
        return nodes[index]

# Copyright (c) OpenMMLab. All rights reserved.

import random

from lmdeploy.serve.proxy.node import Node
from lmdeploy.serve.proxy.routing.base import BaseStrategy


class RandomStrategy(BaseStrategy):
    """Weighted random selection based on node speed."""

    async def select_node(self, model_name: str, role=None) -> Node:
        nodes = await self.registry.get(model_name, role=role)
        if not nodes:
            raise ValueError(f"No available node for model {model_name!r}")

        urls_with_speeds, speeds, urls_without_speeds = [], [], []
        for node in nodes:
            if node.speed is not None:
                urls_with_speeds.append(node)
                speeds.append(node.speed)
            else:
                urls_without_speeds.append(node)

        average_speed = sum(speeds) / len(speeds) if speeds else 1
        all_nodes = urls_with_speeds + urls_without_speeds
        all_speeds = speeds + [average_speed] * len(urls_without_speeds)

        speed_sum = sum(all_speeds)
        weights = [s / speed_sum for s in all_speeds]
        index = random.choices(range(len(all_nodes)), weights=weights)[0]
        return all_nodes[index]

# Copyright (c) OpenMMLab. All rights reserved.

from abc import ABC, abstractmethod

from lmdeploy.serve.proxy.config import ProxyConfig
from lmdeploy.serve.proxy.node import Node, NodeRegistry


class BaseStrategy(ABC):
    """Abstract base class for routing strategies."""

    def __init__(self, registry: NodeRegistry, config: ProxyConfig):
        self.registry = registry
        self.config = config

    @abstractmethod
    async def select_node(self, model_name: str, role=None) -> Node:
        """Select the best node for a request to the given model."""

    async def on_request_start(self, node: Node) -> None:
        """Hook called before forwarding a request."""
        node.unfinished += 1

    async def on_request_end(self, node: Node, latency: float) -> None:
        """Hook called after a request completes."""
        node.unfinished -= 1
        node.latency.append(latency)

    async def start(self) -> None:
        """Start any background tasks.

        Override if needed.
        """

    async def stop(self) -> None:
        """Stop background tasks.

        Override if needed.
        """

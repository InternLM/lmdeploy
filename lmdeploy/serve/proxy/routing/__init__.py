# Copyright (c) OpenMMLab. All rights reserved.

from lmdeploy.serve.proxy.config import ProxyConfig, RoutingStrategy
from lmdeploy.serve.proxy.node import NodeRegistry
from lmdeploy.serve.proxy.routing.base import BaseStrategy


def get_strategy(strategy: RoutingStrategy, registry: NodeRegistry, config: ProxyConfig) -> BaseStrategy:
    """Factory function to create a routing strategy instance."""
    if strategy == RoutingStrategy.RANDOM:
        from lmdeploy.serve.proxy.routing.random import RandomStrategy
        return RandomStrategy(registry, config)
    elif strategy == RoutingStrategy.MIN_EXPECTED_LATENCY:
        from lmdeploy.serve.proxy.routing.min_expected import MinExpectedLatencyStrategy
        return MinExpectedLatencyStrategy(registry, config)
    elif strategy == RoutingStrategy.MIN_OBSERVED_LATENCY:
        from lmdeploy.serve.proxy.routing.min_observed import MinObservedLatencyStrategy
        return MinObservedLatencyStrategy(registry, config)
    elif strategy == RoutingStrategy.MIN_CACHE_USAGE:
        from lmdeploy.serve.proxy.routing.min_cache import MinCacheUsageStrategy
        return MinCacheUsageStrategy(registry, config)
    else:
        raise ValueError(f"Unknown routing strategy: {strategy}")

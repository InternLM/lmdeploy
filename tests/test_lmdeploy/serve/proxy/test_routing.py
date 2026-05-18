# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import collections

import pytest

from lmdeploy.serve.proxy.config import ProxyConfig, RoutingStrategy
from lmdeploy.serve.proxy.node import Node, NodeRegistry
from lmdeploy.serve.proxy.routing import get_strategy
from lmdeploy.serve.proxy.routing.min_expected import MinExpectedLatencyStrategy
from lmdeploy.serve.proxy.routing.min_observed import MinObservedLatencyStrategy
from lmdeploy.serve.proxy.routing.random import RandomStrategy


def _make_registry(*nodes):
    """Create a NodeRegistry and populate its _nodes dict directly."""
    registry = NodeRegistry(cache_status=False)
    registry._nodes = {n.url: n for n in nodes}
    return registry


def _make_node(url='http://localhost:8000', models=None, speed=None,
               unfinished=0, latency=None, cache_usage=None):
    """Create a Node with given attributes."""
    node = Node(url=url, models=models or ['test-model'], speed=speed,
                unfinished=unfinished, cache_usage=cache_usage)
    if latency is not None:
        node.latency = collections.deque(latency, maxlen=15)
    return node


def run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


# --- RandomStrategy tests ---

class TestRandomStrategy:

    def test_select_node_returns_matching_node(self):
        node_a = _make_node(url='http://a', models=['llama'], speed=10)
        node_b = _make_node(url='http://b', models=['llama'], speed=20)
        registry = _make_registry(node_a, node_b)
        config = ProxyConfig()
        strategy = RandomStrategy(registry, config)
        result = run(strategy.select_node('llama'))
        assert result.url in ('http://a', 'http://b')

    def test_select_node_no_matching_model_raises(self):
        node_a = _make_node(url='http://a', models=['qwen'])
        registry = _make_registry(node_a)
        config = ProxyConfig()
        strategy = RandomStrategy(registry, config)
        with pytest.raises(ValueError, match='No available node'):
            run(strategy.select_node('llama'))

    def test_select_node_without_speed_uses_average(self):
        node_a = _make_node(url='http://a', models=['llama'], speed=10)
        node_b = _make_node(url='http://b', models=['llama'], speed=None)
        registry = _make_registry(node_a, node_b)
        config = ProxyConfig()
        strategy = RandomStrategy(registry, config)
        # Run many times to ensure both nodes can be selected
        urls = set()
        for _ in range(50):
            result = run(strategy.select_node('llama'))
            urls.add(result.url)
        assert 'http://a' in urls
        assert 'http://b' in urls


# --- MinExpectedLatencyStrategy tests ---

class TestMinExpectedLatencyStrategy:

    def test_selects_node_with_lowest_unfinished_over_speed(self):
        # node_a: 2/10 = 0.2, node_b: 1/10 = 0.1 -> node_b wins
        node_a = _make_node(url='http://a', models=['llama'], speed=10, unfinished=2)
        node_b = _make_node(url='http://b', models=['llama'], speed=10, unfinished=1)
        registry = _make_registry(node_a, node_b)
        config = ProxyConfig()
        strategy = MinExpectedLatencyStrategy(registry, config)
        result = run(strategy.select_node('llama'))
        assert result.url == 'http://b'

    def test_handles_nodes_with_no_speed(self):
        # node_a: 1/None -> uses avg speed (10), 1/10 = 0.1
        # node_b: 1/10 = 0.1 -> tie, either could win
        node_a = _make_node(url='http://a', models=['llama'], speed=None, unfinished=1)
        node_b = _make_node(url='http://b', models=['llama'], speed=10, unfinished=1)
        registry = _make_registry(node_a, node_b)
        config = ProxyConfig()
        strategy = MinExpectedLatencyStrategy(registry, config)
        result = run(strategy.select_node('llama'))
        assert result.url in ('http://a', 'http://b')

    def test_select_node_no_matching_model_raises(self):
        node_a = _make_node(url='http://a', models=['qwen'])
        registry = _make_registry(node_a)
        config = ProxyConfig()
        strategy = MinExpectedLatencyStrategy(registry, config)
        with pytest.raises(ValueError, match='No available node'):
            run(strategy.select_node('llama'))


# --- MinObservedLatencyStrategy tests ---

class TestMinObservedLatencyStrategy:

    def test_selects_node_with_lowest_mean_latency(self):
        node_a = _make_node(url='http://a', models=['llama'], latency=[1.0, 2.0, 3.0])
        node_b = _make_node(url='http://b', models=['llama'], latency=[0.5, 0.6, 0.7])
        registry = _make_registry(node_a, node_b)
        config = ProxyConfig()
        strategy = MinObservedLatencyStrategy(registry, config)
        result = run(strategy.select_node('llama'))
        assert result.url == 'http://b'

    def test_handles_nodes_with_no_latency_history(self):
        # node_a has latency data, node_b has none (inf)
        node_a = _make_node(url='http://a', models=['llama'], latency=[1.0, 2.0])
        node_b = _make_node(url='http://b', models=['llama'], latency=[])
        registry = _make_registry(node_a, node_b)
        config = ProxyConfig()
        strategy = MinObservedLatencyStrategy(registry, config)
        result = run(strategy.select_node('llama'))
        assert result.url == 'http://a'

    def test_select_node_no_matching_model_raises(self):
        node_a = _make_node(url='http://a', models=['qwen'])
        registry = _make_registry(node_a)
        config = ProxyConfig()
        strategy = MinObservedLatencyStrategy(registry, config)
        with pytest.raises(ValueError, match='No available node'):
            run(strategy.select_node('llama'))


# --- BaseStrategy hook tests ---

class TestBaseStrategyHooks:

    def test_on_request_start_increments_unfinished(self):
        node = _make_node(url='http://a', models=['llama'], unfinished=0)
        registry = _make_registry(node)
        config = ProxyConfig()
        strategy = RandomStrategy(registry, config)
        run(strategy.on_request_start(node))
        assert node.unfinished == 1

    def test_on_request_end_decrements_unfinished_and_records_latency(self):
        node = _make_node(url='http://a', models=['llama'], unfinished=1)
        registry = _make_registry(node)
        config = ProxyConfig()
        strategy = RandomStrategy(registry, config)
        run(strategy.on_request_end(node, 0.5))
        assert node.unfinished == 0
        assert list(node.latency) == [0.5]

    def test_on_request_start_and_end_combined(self):
        node = _make_node(url='http://a', models=['llama'], unfinished=0)
        registry = _make_registry(node)
        config = ProxyConfig()
        strategy = RandomStrategy(registry, config)
        run(strategy.on_request_start(node))
        assert node.unfinished == 1
        run(strategy.on_request_end(node, 1.2))
        assert node.unfinished == 0
        assert list(node.latency) == [1.2]


# --- Factory tests ---

class TestGetStrategyFactory:

    def test_random_strategy(self):
        registry = _make_registry()
        config = ProxyConfig()
        strategy = get_strategy(RoutingStrategy.RANDOM, registry, config)
        assert isinstance(strategy, RandomStrategy)

    def test_min_expected_latency_strategy(self):
        registry = _make_registry()
        config = ProxyConfig()
        strategy = get_strategy(RoutingStrategy.MIN_EXPECTED_LATENCY, registry, config)
        assert isinstance(strategy, MinExpectedLatencyStrategy)

    def test_min_observed_latency_strategy(self):
        registry = _make_registry()
        config = ProxyConfig()
        strategy = get_strategy(RoutingStrategy.MIN_OBSERVED_LATENCY, registry, config)
        assert isinstance(strategy, MinObservedLatencyStrategy)

    def test_min_cache_usage_strategy_imports_correctly(self):
        # MIN_CACHE_USAGE is defined but the module doesn't exist yet;
        # just verify the factory references it correctly by checking
        # that it attempts the import (will raise ImportError, not ValueError)
        registry = _make_registry()
        config = ProxyConfig()
        with pytest.raises(ImportError):
            get_strategy(RoutingStrategy.MIN_CACHE_USAGE, registry, config)

    def test_unknown_strategy_raises(self):
        registry = _make_registry()
        config = ProxyConfig()
        with pytest.raises(ValueError, match='Unknown routing strategy'):
            get_strategy('not_a_strategy', registry, config)

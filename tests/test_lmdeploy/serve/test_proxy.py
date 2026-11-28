import pytest

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.serve.proxy.proxy import NodeManager, Status
from lmdeploy.serve.proxy.utils import RoutingStrategy


def _make_node_manager(routing_strategy: RoutingStrategy, nodes: dict) -> NodeManager:
    """Build a NodeManager without running ``__init__`` side effects.

    ``NodeManager.__init__`` starts a heartbeat thread and creates a PD
    connection pool, neither of which is needed to exercise ``get_node_url``.
    """
    manager = NodeManager.__new__(NodeManager)
    manager.routing_strategy = routing_strategy
    manager.nodes = nodes
    return manager


@pytest.mark.parametrize('routing_strategy', [
    RoutingStrategy.RANDOM,
    RoutingStrategy.MIN_EXPECTED_LATENCY,
    RoutingStrategy.MIN_OBSERVED_LATENCY,
])
def test_get_node_url_returns_none_when_no_node_matches_role(routing_strategy):
    # The model is only registered on a Decode node, but queried with the
    # default Hybrid role. ``get_node_url`` should return None (so callers can
    # reply with an "unavailable model" error) instead of crashing.
    manager = _make_node_manager(routing_strategy, {
        'http://decode-node:23333': Status(role=EngineRole.Decode, models=['test-model']),
    })
    assert manager.get_node_url('test-model') is None


@pytest.mark.parametrize('routing_strategy', [
    RoutingStrategy.RANDOM,
    RoutingStrategy.MIN_EXPECTED_LATENCY,
    RoutingStrategy.MIN_OBSERVED_LATENCY,
])
def test_get_node_url_returns_url_when_node_matches(routing_strategy):
    manager = _make_node_manager(routing_strategy, {
        'http://hybrid-node:23333': Status(role=EngineRole.Hybrid, models=['test-model'], speed=100),
    })
    assert manager.get_node_url('test-model') == 'http://hybrid-node:23333'

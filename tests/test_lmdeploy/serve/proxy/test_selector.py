# Copyright (c) OpenMMLab. All rights reserved.

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.disagg.conn.proxy_conn import PDConnectionPool
from lmdeploy.serve.proxy.core.config import RoutingStrategy
from lmdeploy.serve.proxy.core.replica import ReplicaLoad, SelectedReplica
from lmdeploy.serve.proxy.registry.pool import ReplicaPool
from lmdeploy.serve.proxy.routing.selector import ReplicaSelector


def test_acquire_picks_replica_with_model():
    pool = ReplicaPool(PDConnectionPool())
    pool.add('http://a:1', ReplicaLoad(models=['m1'], speed=10))
    pool.add('http://b:1', ReplicaLoad(models=['m2'], speed=10))
    selector = ReplicaSelector(pool, RoutingStrategy.MIN_EXPECTED_LATENCY)

    selected = selector.acquire('m1', EngineRole.Hybrid)
    assert selected.url == 'http://a:1'
    pool.inflight_finish(selected)

    selected = selector.acquire('m2', EngineRole.Hybrid)
    assert selected.url == 'http://b:1'
    pool.inflight_finish(selected)

    assert selector.acquire('missing', EngineRole.Hybrid) is None


def test_acquire_reserves_inflight_atomically():
    pool = ReplicaPool(PDConnectionPool())
    pool.add('http://a:1', ReplicaLoad(models=['m1'], speed=10))
    selector = ReplicaSelector(pool, RoutingStrategy.MIN_EXPECTED_LATENCY)
    selected = selector.acquire('m1', EngineRole.Hybrid)
    assert isinstance(selected, SelectedReplica)
    assert selected.url == 'http://a:1'
    assert pool.snapshot()['http://a:1'].unfinished == 1
    pool.inflight_finish(selected)
    assert pool.snapshot()['http://a:1'].unfinished == 0

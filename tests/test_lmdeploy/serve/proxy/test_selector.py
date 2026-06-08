# Copyright (c) OpenMMLab. All rights reserved.

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.disagg.conn.proxy_conn import PDConnectionPool
from lmdeploy.serve.proxy.core.config import RoutingStrategy
from lmdeploy.serve.proxy.core.replica import ReplicaLoad
from lmdeploy.serve.proxy.registry.pool import ReplicaPool
from lmdeploy.serve.proxy.routing.selector import ReplicaSelector


def test_selector_picks_replica_with_model():
    pool = ReplicaPool(PDConnectionPool())
    pool.add('http://a:1', ReplicaLoad(models=['m1'], speed=10))
    pool.add('http://b:1', ReplicaLoad(models=['m2'], speed=10))
    selector = ReplicaSelector(pool, RoutingStrategy.MIN_EXPECTED_LATENCY)
    assert selector.select('m1', EngineRole.Hybrid) == 'http://a:1'
    assert selector.select('m2', EngineRole.Hybrid) == 'http://b:1'
    assert selector.select('missing', EngineRole.Hybrid) is None

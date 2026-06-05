# Copyright (c) OpenMMLab. All rights reserved.

from unittest.mock import patch

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.disagg.conn.proxy_conn import PDConnectionPool
from lmdeploy.serve.proxy.core.replica import ReplicaLoad
from lmdeploy.serve.proxy.registry.pool import ReplicaPool
from lmdeploy.serve.proxy.routing.selector import ReplicaSelector
from lmdeploy.serve.proxy.utils import RoutingStrategy


def _mock_models(api_url: str, headers=None):
    if 'http://a:1' in api_url:
        return ['m1']
    if 'http://b:1' in api_url:
        return ['m2']
    return None


@patch('lmdeploy.serve.openai.api_client.get_model_list', side_effect=_mock_models)
def test_selector_picks_replica_with_model(_mock_get_model_list):
    pool = ReplicaPool(PDConnectionPool())
    pool.add('http://a:1', ReplicaLoad(models=['m1'], speed=10))
    pool.add('http://b:1', ReplicaLoad(models=['m2'], speed=10))
    selector = ReplicaSelector(pool, RoutingStrategy.MIN_EXPECTED_LATENCY)
    assert selector.select('m1', EngineRole.Hybrid) == 'http://a:1'
    assert selector.select('m2', EngineRole.Hybrid) == 'http://b:1'
    assert selector.select('missing', EngineRole.Hybrid) is None

# Copyright (c) OpenMMLab. All rights reserved.

import pytest

from lmdeploy.pytorch.disagg.conn.proxy_conn import PDConnectionPool
from lmdeploy.serve.proxy.core.replica import ReplicaLoad
from lmdeploy.serve.proxy.registry.pool import ReplicaNotFoundError, ReplicaPool


def test_add_and_snapshot():
    pool = ReplicaPool(PDConnectionPool())
    pool.add('http://replica:8000', ReplicaLoad(models=['model-a']))
    assert pool.snapshot()['http://replica:8000'].models == ['model-a']


def test_add_replaces_existing_entry():
    pool = ReplicaPool(PDConnectionPool())
    pool.add('http://replica:8000', ReplicaLoad(models=['model-a'], speed=10))
    pool.add('http://replica:8000', ReplicaLoad(models=['model-b']))
    assert pool.snapshot()['http://replica:8000'].models == ['model-b']


def test_remove_existing_replica():
    pool = ReplicaPool(PDConnectionPool())
    pool.add('http://replica:8000', ReplicaLoad(models=['model-a']))
    pool.remove('http://replica:8000')
    assert pool.snapshot() == {}


def test_remove_missing_replica_raises():
    pool = ReplicaPool(PDConnectionPool())
    with pytest.raises(ReplicaNotFoundError):
        pool.remove('http://missing:8000')

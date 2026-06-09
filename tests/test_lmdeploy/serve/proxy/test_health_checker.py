# Copyright (c) OpenMMLab. All rights reserved.

from http import HTTPStatus
from unittest.mock import MagicMock, patch

import requests

from lmdeploy.pytorch.disagg.conn.proxy_conn import PDConnectionPool
from lmdeploy.serve.proxy.core.replica import ReplicaLoad
from lmdeploy.serve.proxy.registry.health_checker import (
    HealthChecker,
    HealthProbeResult,
    probe_replica_health,
)
from lmdeploy.serve.proxy.registry.pool import ReplicaPool


def _mock_response(status_code: int, json_data: dict | None = None):
    response = MagicMock()
    response.status_code = status_code
    if json_data is None:
        response.json.side_effect = ValueError('not json')
    else:
        response.json.return_value = json_data
    return response


@patch('lmdeploy.serve.proxy.registry.health_checker.requests.get')
def test_probe_healthy(mock_get):
    mock_get.return_value = _mock_response(HTTPStatus.OK, {'status': 'healthy', 'message': 'ok'})
    result = probe_replica_health('http://replica:1', timeout=1.0)
    assert result.is_serving
    assert result.status == 'healthy'


@patch('lmdeploy.serve.proxy.registry.health_checker.requests.get')
def test_probe_initializing_is_transient(mock_get):
    mock_get.return_value = _mock_response(
        HTTPStatus.SERVICE_UNAVAILABLE,
        {'status': 'initializing', 'message': 'starting'},
    )
    result = probe_replica_health('http://replica:1', timeout=1.0)
    assert result.is_transient
    assert not result.is_serving


@patch('lmdeploy.serve.proxy.registry.health_checker.requests.get')
def test_probe_unreachable(mock_get):
    mock_get.side_effect = requests.ConnectionError('refused')
    result = probe_replica_health('http://replica:1', timeout=1.0)
    assert not result.reachable
    assert not result.is_serving


@patch('lmdeploy.serve.proxy.registry.health_checker.probe_replica_health')
def test_run_once_evicts_after_threshold(mock_probe):
    mock_probe.return_value = HealthProbeResult(
        reachable=True,
        status='unhealthy',
        message='engine dead',
    )
    pool = ReplicaPool(PDConnectionPool())
    pool._replicas['http://dead:1'] = ReplicaLoad(models=['m1'])
    checker = HealthChecker(pool)
    checker.interval_sec = 1
    checker.run_once()
    assert pool.snapshot()
    checker.run_once()
    assert pool.snapshot() == {}


@patch('lmdeploy.serve.proxy.registry.health_checker.probe_replica_health')
def test_run_once_keeps_initializing(mock_probe):
    mock_probe.return_value = HealthProbeResult(
        reachable=True,
        status='initializing',
        message='starting',
    )
    pool = ReplicaPool(PDConnectionPool())
    pool._replicas['http://boot:1'] = ReplicaLoad(models=['m1'])
    checker = HealthChecker(pool)
    for _ in range(3):
        checker.run_once()
    assert 'http://boot:1' in pool.snapshot()


@patch('lmdeploy.serve.proxy.registry.health_checker.probe_replica_health')
def test_run_once_resets_failures_on_recovery(mock_probe):
    pool = ReplicaPool(PDConnectionPool())
    pool._replicas['http://node:1'] = ReplicaLoad(models=['m1'])
    checker = HealthChecker(pool)
    checker.interval_sec = 1
    mock_probe.return_value = HealthProbeResult(reachable=False, message='down')
    checker.run_once()
    mock_probe.return_value = HealthProbeResult(reachable=True, status='healthy', message='ok')
    checker.run_once()
    mock_probe.return_value = HealthProbeResult(reachable=False, message='down')
    checker.run_once()
    assert 'http://node:1' in pool.snapshot()

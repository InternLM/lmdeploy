# Copyright (c) OpenMMLab. All rights reserved.

import os
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus

import requests

from lmdeploy.serve.proxy.registry.pool import ReplicaNotFoundError, ReplicaPool
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

_DEFAULT_INTERVAL_SEC = 30
_DEFAULT_TIMEOUT_SEC = 5.0
_FAILURE_THRESHOLD = 2

_SERVING = frozenset({'healthy', 'sleeping'})
_TRANSIENT = frozenset({'initializing'})


def _interval_sec() -> int:
    for name in ('LMDEPLOY_HEALTH_CHECK_INTERVAL', 'LMDEPLOY_CONTROLLER_HEART_BEAT_EXPIRATION'):
        value = os.getenv(name)
        if value:
            return int(value)
    return _DEFAULT_INTERVAL_SEC


def _timeout_sec() -> float:
    value = os.getenv('LMDEPLOY_HEALTH_CHECK_TIMEOUT')
    return float(value) if value else _DEFAULT_TIMEOUT_SEC


@dataclass(frozen=True)
class HealthProbeResult:
    reachable: bool
    status: str | None = None
    message: str | None = None

    @property
    def is_serving(self) -> bool:
        return self.status in _SERVING

    @property
    def is_transient(self) -> bool:
        return self.status in _TRANSIENT


def probe_replica_health(url: str, timeout: float) -> HealthProbeResult:
    """GET {url}/health and read api_server JSON ``status`` / ``message``."""
    try:
        response = requests.get(
            f'{url}/health',
            headers={'accept': 'application/json'},
            timeout=timeout,
        )
    except requests.exceptions.RequestException as e:
        return HealthProbeResult(reachable=False, message=str(e))

    try:
        data = response.json()
        status = data.get('status')
        message = data.get('message')
    except ValueError:
        if response.status_code == HTTPStatus.OK:
            return HealthProbeResult(reachable=True, status='healthy')
        return HealthProbeResult(reachable=True, status='unhealthy', message='invalid health JSON')

    if isinstance(status, str):
        return HealthProbeResult(reachable=True, status=status, message=message)
    return HealthProbeResult(reachable=True, status='unhealthy', message='missing status field')


class HealthChecker:
    """Background replica health checks aligned with api_server /health."""

    def __init__(self, pool: ReplicaPool) -> None:
        self.pool = pool
        self.interval_sec = _interval_sec()
        self.timeout_sec = _timeout_sec()
        self._failures: dict[str, int] = {}
        self._thread: threading.Thread | None = None

    def start(self) -> threading.Thread:
        if self._thread is not None and self._thread.is_alive():
            return self._thread
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self._thread

    def _loop(self) -> None:
        self.run_once()
        while True:
            time.sleep(self.interval_sec)
            self.run_once()

    def run_once(self) -> None:
        logger.info('Start health check')
        to_delete = [url for url in self.pool.snapshot() if self._should_evict(url)]
        for url in to_delete:
            try:
                self.pool.remove(url)
            except ReplicaNotFoundError:
                continue
            self._failures.pop(url, None)
            logger.info(f'Removed replica {url} after failed health checks')

    def _should_evict(self, url: str) -> bool:
        result = probe_replica_health(url, self.timeout_sec)
        if result.is_serving or result.is_transient:
            self._failures.pop(url, None)
            return False

        count = self._failures.get(url, 0) + 1
        self._failures[url] = count
        detail = result.message or result.status or 'unreachable'
        if count < _FAILURE_THRESHOLD:
            logger.warning(f'Health check failed for {url} ({count}/{_FAILURE_THRESHOLD}): {detail}')
            return False
        logger.warning(f'Evicting {url} after {count} failed health checks: {detail}')
        return True

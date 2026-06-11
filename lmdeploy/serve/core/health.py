# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING

from lmdeploy.utils import get_logger

if TYPE_CHECKING:
    from .async_engine import AsyncEngine

logger = get_logger('lmdeploy')

HEALTH_POLL_INTERVAL = 'LMDEPLOY_HEALTH_POLL_INTERVAL'
HEALTH_PROBE_TIMEOUT = 'LMDEPLOY_HEALTH_PROBE_TIMEOUT'
HEALTH_UNHEALTHY_AFTER = 'LMDEPLOY_HEALTH_UNHEALTHY_AFTER'

DEFAULT_PROBE_TIMEOUT = 10.0
DEFAULT_POLL_INTERVAL = 12.0
DEFAULT_UNHEALTHY_AFTER = 90.0


def _env_override_float(env_var: str, value: float) -> float:
    """Return ``value`` unless ``env_var`` is set, then parse and return it."""
    env_value = os.getenv(env_var)
    if env_value is None:
        return value
    try:
        return float(env_value)
    except ValueError:
        return value


class EngineHealthMonitor:
    """Background engine health monitor."""

    def __init__(self,
                 async_engine: AsyncEngine | None,
                 poll_interval: float = DEFAULT_POLL_INTERVAL,
                 probe_timeout: float = DEFAULT_PROBE_TIMEOUT,
                 unhealthy_after: float = DEFAULT_UNHEALTHY_AFTER):
        """Initialize the background health monitor.

        Args:
            async_engine: Engine instance to probe; ``None`` marks the service
                unhealthy until an engine is attached.
            poll_interval: Seconds between consecutive ``probe_once()`` calls in
                the background loop (default 12.0). Should be greater than
                ``probe_timeout`` to reduce overlapping probes. Overridden by
                ``LMDEPLOY_HEALTH_POLL_INTERVAL`` when that variable is set.
            probe_timeout: Maximum seconds to wait for a single
                ``health_probe()`` (backend ``get_health_status()``) before
                reporting that probe as unhealthy (default 10.0). Overridden by
                ``LMDEPLOY_HEALTH_PROBE_TIMEOUT`` when that variable is set.
            unhealthy_after: Seconds without a successful probe or scheduler
                progress before the service is considered unhealthy (default
                90.0). Overridden by ``LMDEPLOY_HEALTH_UNHEALTHY_AFTER`` when
                that variable is set. Passed to ``health_probe()`` as
                ``scheduler_stall_timeout``, and also used in ``snapshot()``
                to expire stale healthy status or stuck ``initializing`` state.
        """
        self.async_engine = async_engine
        self.poll_interval = _env_override_float(HEALTH_POLL_INTERVAL, poll_interval)
        self.probe_timeout = _env_override_float(HEALTH_PROBE_TIMEOUT, probe_timeout)
        self.unhealthy_after = _env_override_float(HEALTH_UNHEALTHY_AFTER, unhealthy_after)
        if self.poll_interval <= self.probe_timeout:
            logger.warning('Engine health poll_interval (%.1fs) should be greater than probe_timeout (%.1fs) '
                             'to avoid overlapping probes.',
                             self.poll_interval, self.probe_timeout)
        self._task: asyncio.Task | None = None
        self._started_time = time.monotonic()
        self._last_success_time: float | None = None
        self._snapshot = dict(status='initializing',
                              message='Engine health monitor is starting.')

    def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name='EngineHealthMonitor')

    async def stop(self):
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None

    async def _run(self):
        while True:
            await self.probe_once()
            await asyncio.sleep(self.poll_interval)

    async def probe_once(self):
        probe_time = time.monotonic()
        if self.async_engine is None:
            result = dict(status='unhealthy',
                          message='Async engine is not initialized.')
        else:
            try:
                result = await self.async_engine.health_probe(timeout=self.probe_timeout,
                                                              scheduler_stall_timeout=self.unhealthy_after)
            except Exception as e:
                result = dict(status='unhealthy',
                              message=f'Engine health probe failed: {e}')

        status = result['status']
        if status == 'pending':
            logger.info('Engine health probe skipped: previous backend health probe is still pending.')
            return
        if status in ('healthy', 'sleeping'):
            self._last_success_time = probe_time
        self._snapshot = dict(status=status, message=result['message'])

    def snapshot(self) -> dict:
        snapshot = dict(self._snapshot)
        now = time.monotonic()
        if snapshot['status'] == 'healthy' and self._last_success_time is not None:
            if now - self._last_success_time > self.unhealthy_after:
                snapshot['status'] = 'unhealthy'
                snapshot['message'] = f'No successful health probe for {now - self._last_success_time:.1f}s.'
        elif snapshot['status'] == 'initializing' and now - self._started_time > self.unhealthy_after:
            snapshot['status'] = 'unhealthy'
            snapshot['message'] = 'Engine health monitor did not complete an initial probe.'
        return snapshot

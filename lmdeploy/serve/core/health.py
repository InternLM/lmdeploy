# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .async_engine import AsyncEngine


class EngineHealthMonitor:
    """Background engine health monitor."""

    def __init__(self,
                 async_engine: AsyncEngine | None,
                 poll_interval: float = 5.0,
                 probe_timeout: float = 2.0,
                 unhealthy_after: float = 15.0):
        self.async_engine = async_engine
        self.poll_interval = poll_interval
        self.probe_timeout = probe_timeout
        self.unhealthy_after = unhealthy_after
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
        if status in ('healthy', 'sleeping'):
            self._last_success_time = probe_time
        self._snapshot = dict(status=status, message=result['message'])

    def snapshot(self) -> dict:
        snapshot = dict(self._snapshot)
        now = time.monotonic()
        if snapshot['status'] in ('healthy', 'sleeping') and self._last_success_time is not None:
            if now - self._last_success_time > self.unhealthy_after:
                snapshot['status'] = 'unhealthy'
                snapshot['message'] = f'No successful health probe for {now - self._last_success_time:.1f}s.'
        elif snapshot['status'] == 'initializing' and now - self._started_time > self.unhealthy_after:
            snapshot['status'] = 'unhealthy'
            snapshot['message'] = 'Engine health monitor did not complete an initial probe.'
        return snapshot

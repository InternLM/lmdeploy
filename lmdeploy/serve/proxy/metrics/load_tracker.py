# Copyright (c) OpenMMLab. All rights reserved.

from lmdeploy.serve.proxy.registry.pool import ReplicaPool


class InflightTracker:
    """Track inflight requests and record completion latency."""

    def __init__(self, pool: ReplicaPool) -> None:
        self._pool = pool

    def start(self, url: str) -> float | None:
        return self._pool.inflight_start(url)

    def finish(self, url: str, start: float | None) -> None:
        self._pool.inflight_finish(url, start)

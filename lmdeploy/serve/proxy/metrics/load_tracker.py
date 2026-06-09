# Copyright (c) OpenMMLab. All rights reserved.

from lmdeploy.serve.proxy.core.replica import SelectedReplica
from lmdeploy.serve.proxy.registry.pool import ReplicaPool


class InflightTracker:
    """Track inflight requests and record completion latency."""

    def __init__(self, pool: ReplicaPool) -> None:
        self._pool = pool

    def finish(self, selected: SelectedReplica) -> None:
        self._pool.inflight_finish(selected)

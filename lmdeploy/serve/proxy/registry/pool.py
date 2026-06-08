# Copyright (c) OpenMMLab. All rights reserved.

import copy
import threading
import time

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.serve.proxy.core.replica import ReplicaLoad


class ReplicaNotFoundError(Exception):
    """Raised when an operation targets a replica URL that is not
    registered."""

    def __init__(self, url: str) -> None:
        self.url = url
        super().__init__(f'Replica {url} is not in the pool.')


class ReplicaPool:
    """In-memory registry of api_server replicas."""

    def __init__(self, pd_connection_pool) -> None:
        self._replicas: dict[str, ReplicaLoad] = {}
        self._lock = threading.Lock()
        self.pd_connection_pool = pd_connection_pool

    def get_by_role(self, role: EngineRole) -> dict[str, ReplicaLoad]:
        with self._lock:
            items = list(self._replicas.items())
        return {url: load for url, load in items if load.role == role}

    def snapshot(self) -> dict[str, ReplicaLoad]:
        with self._lock:
            return copy.deepcopy(self._replicas)

    @property
    def model_list(self) -> list[str]:
        names: list[str] = []
        with self._lock:
            for load in self._replicas.values():
                names.extend(load.models)
        return list(dict.fromkeys(names))

    def add(self, url: str, load: ReplicaLoad) -> None:
        """Add a replica entry, replacing any existing entry for the same
        URL."""
        self._discard(url)
        with self._lock:
            self._replicas[url] = load.model_copy(deep=True)

    def remove(self, url: str) -> None:
        """Remove a replica entry."""
        with self._lock:
            existed = self._replicas.pop(url, None)
        if existed is None:
            raise ReplicaNotFoundError(url)
        self.pd_connection_pool.dereg_instance(url)

    def _discard(self, url: str) -> None:
        """Remove a replica if present, without raising."""
        with self._lock:
            existed = self._replicas.pop(url, None)
        if existed is not None:
            self.pd_connection_pool.dereg_instance(url)

    def inflight_start(self, url: str) -> float | None:
        with self._lock:
            load = self._replicas.get(url)
            if load is None:
                return None
            load.unfinished += 1
        return time.time()

    def inflight_finish(self, url: str, start: float | None) -> None:
        if start is None:
            return
        with self._lock:
            load = self._replicas.get(url)
            if load is None:
                return
            load.unfinished -= 1
            load.record_latency(time.time() - start)

# Copyright (c) OpenMMLab. All rights reserved.

import random

import numpy as np

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.serve.proxy.core.config import RoutingStrategy
from lmdeploy.serve.proxy.core.replica import ReplicaLoad, SelectedReplica
from lmdeploy.serve.proxy.registry.pool import ReplicaPool


class ReplicaSelector:
    """Select a replica for a model and engine role."""

    def __init__(self, pool: ReplicaPool, routing_strategy: RoutingStrategy) -> None:
        self._pool = pool
        self._routing_strategy = routing_strategy

    def acquire(self, model_name: str, role: EngineRole = EngineRole.Hybrid) -> SelectedReplica | None:
        """Pick a replica and reserve an inflight slot atomically."""
        return self._pool.acquire(role, model_name, self._pick)

    def _pick(self, candidates: dict[str, ReplicaLoad]) -> str | None:
        if self._routing_strategy == RoutingStrategy.RANDOM:
            return self._pick_random(candidates)
        if self._routing_strategy == RoutingStrategy.MIN_EXPECTED_LATENCY:
            return self._pick_expected_latency(candidates)
        if self._routing_strategy == RoutingStrategy.MIN_OBSERVED_LATENCY:
            return self._pick_observed_latency(candidates)
        raise ValueError(f'Invalid strategy: {self._routing_strategy}')

    @staticmethod
    def _mean_replica_speed(candidates: dict[str, ReplicaLoad]) -> float:
        speeds = [load.speed for load in candidates.values() if load.speed and load.speed > 0]
        return sum(speeds) / len(speeds) if speeds else 1.0

    @classmethod
    def _effective_speed(cls, load: ReplicaLoad, candidates: dict[str, ReplicaLoad]) -> float:
        if load.speed and load.speed > 0:
            return float(load.speed)
        return cls._mean_replica_speed(candidates)

    def _pick_random(self, candidates: dict[str, ReplicaLoad]) -> str | None:
        return random.choice(list(candidates.keys()))

    def _pick_expected_latency(self, candidates: dict[str, ReplicaLoad]) -> str | None:
        urls = list(candidates.keys())
        min_latency = float('inf')
        min_index = 0
        indexes = list(range(len(urls)))
        random.shuffle(indexes)
        for index in indexes:
            load = candidates[urls[index]]
            speed = self._effective_speed(load, candidates)
            latency = load.unfinished / max(speed, 1.0)
            if min_latency > latency:
                min_latency = latency
                min_index = index
        return urls[min_index]

    def _pick_observed_latency(self, candidates: dict[str, ReplicaLoad]) -> str | None:
        urls = list(candidates.keys())
        latencies = [candidates[url].mean_latency() for url in urls]
        index = int(np.argmin(np.array(latencies, dtype=float)))
        return urls[index]

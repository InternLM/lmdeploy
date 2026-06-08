# Copyright (c) OpenMMLab. All rights reserved.

import random

import numpy as np

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.serve.proxy.core.config import RoutingStrategy
from lmdeploy.serve.proxy.registry.pool import ReplicaPool


class ReplicaSelector:
    """Select a replica URL for a model and engine role."""

    def __init__(self, pool: ReplicaPool, routing_strategy: RoutingStrategy) -> None:
        self._pool = pool
        self._routing_strategy = routing_strategy

    def select(self, model_name: str, role: EngineRole = EngineRole.Hybrid) -> str | None:
        if self._routing_strategy == RoutingStrategy.RANDOM:
            return self._pick_random(model_name, role)
        if self._routing_strategy == RoutingStrategy.MIN_EXPECTED_LATENCY:
            return self._pick_expected_latency(model_name, role)
        if self._routing_strategy == RoutingStrategy.MIN_OBSERVED_LATENCY:
            return self._pick_observed_latency(model_name, role)
        raise ValueError(f'Invalid strategy: {self._routing_strategy}')

    def _collect_candidates(self, model_name: str, role: EngineRole) -> tuple[list[str], list[float]]:
        candidates: list[str] = []
        speeds: list[float] = []
        replicas = self._pool.get_by_role(role)
        for url, load in replicas.items():
            if model_name not in load.models:
                continue
            candidates.append(url)
            if load.speed is None or load.speed <= 0:
                speeds.append(1.0)
            else:
                speeds.append(float(load.speed))

        if not candidates:
            return candidates, speeds

        observed = [load.speed for load in replicas.values() if load.speed and load.speed > 0]
        if not observed:
            observed = [1.0]
        average_speed = sum(observed) / len(observed)
        return candidates, [
            speed if replicas[url].speed else average_speed
            for url, speed in zip(candidates, speeds)
        ]

    def _pick_random(self, model_name: str, role: EngineRole) -> str | None:
        urls, speeds = self._collect_candidates(model_name, role)
        if not urls:
            return None
        speed_sum = sum(speeds)
        if speed_sum <= 0:
            speed_sum = float(len(speeds))
        weights = [speed / speed_sum for speed in speeds]
        index = random.choices(list(range(len(urls))), weights=weights)[0]
        return urls[index]

    def _pick_expected_latency(self, model_name: str, role: EngineRole) -> str | None:
        urls, speeds = self._collect_candidates(model_name, role)
        if not urls:
            return None
        replicas = self._pool.get_by_role(role)
        min_latency = float('inf')
        min_index = 0
        indexes = list(range(len(speeds)))
        random.shuffle(indexes)
        for index in indexes:
            latency = replicas[urls[index]].unfinished / max(speeds[index], 1.0)
            if min_latency > latency:
                min_latency = latency
                min_index = index
        return urls[min_index]

    def _pick_observed_latency(self, model_name: str, role: EngineRole) -> str | None:
        urls: list[str] = []
        latencies: list[float] = []
        for url, load in self._pool.get_by_role(role).items():
            if model_name not in load.models:
                continue
            urls.append(url)
            latencies.append(load.mean_latency())
        if not urls:
            return None
        index = int(np.argmin(np.array(latencies, dtype=float)))
        return urls[index]

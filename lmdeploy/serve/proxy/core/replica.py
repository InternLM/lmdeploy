# Copyright (c) OpenMMLab. All rights reserved.

from pydantic import BaseModel, Field

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.serve.proxy.utils import LATENCY_DEQUE_LEN


class ReplicaLoad(BaseModel):
    """Per-replica load snapshot and capability metadata."""

    role: EngineRole = EngineRole.Hybrid
    models: list[str] = Field(default_factory=list)
    unfinished: int = 0
    latency: list[float] = Field(default_factory=list)
    speed: int | None = None

    def record_latency(self, duration: float) -> None:
        self.latency.append(duration)
        if len(self.latency) > LATENCY_DEQUE_LEN:
            self.latency = self.latency[-LATENCY_DEQUE_LEN:]

    def mean_latency(self) -> float:
        if not self.latency:
            return float('inf')
        return sum(self.latency) / len(self.latency)


class ReplicaRegistration(BaseModel):
    """Registration payload for POST /nodes/add."""

    url: str
    status: ReplicaLoad | None = None

# Copyright (c) OpenMMLab. All rights reserved.

from dataclasses import dataclass

from pydantic import BaseModel, Field

from lmdeploy.pytorch.disagg.config import EngineRole


class ReplicaLoad(BaseModel):
    """Per-replica load snapshot and capability metadata."""

    _LATENCY_DEQUE_LEN = 15

    role: EngineRole = EngineRole.Hybrid
    models: list[str] = Field(default_factory=list)
    unfinished: int = 0
    latency: list[float] = Field(default_factory=list)
    speed: int | None = None

    def record_latency(self, duration: float) -> None:
        self.latency.append(duration)
        if len(self.latency) > self._LATENCY_DEQUE_LEN:
            self.latency = self.latency[-self._LATENCY_DEQUE_LEN:]

    def mean_latency(self) -> float:
        if not self.latency:
            return float('inf')
        return sum(self.latency) / len(self.latency)


@dataclass(frozen=True)
class SelectedReplica:
    """A replica chosen for one request, with inflight slot already
    reserved."""

    url: str
    start_time: float


class ReplicaRegistration(BaseModel):
    """Request body for replica admin endpoints (``/nodes/add``,
    ``/nodes/remove``, ``/nodes/terminate``).

    Attributes:
        url: Base URL of the api_server replica (e.g. ``http://127.0.0.1:23333``).
        status: Optional metadata supplied by the caller. On add, ``role`` and
            ``speed`` are kept; ``models`` may be declared for validation against
            ``GET {url}/v1/models`` but are always replaced with the discovered
            model list. Omitted fields use :class:`ReplicaLoad` defaults.
    """

    url: str
    status: ReplicaLoad | None = None

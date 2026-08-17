# Copyright (c) OpenMMLab. All rights reserved.
"""Base interfaces for external KV-cache connectors.

A connector has a scheduler-side instance and one worker-side instance per model worker. The scheduler instance
discovers external cache hits and builds serializable metadata for a model step. Worker instances consume that metadata
to load or save the GPU KV cache.

The interface intentionally contains only the lifecycle needed by lmdeploy's PyTorch engine.
"""

import enum
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from lmdeploy.pytorch.messages import SchedulerSequence
    from lmdeploy.pytorch.paging.scheduler import SchedulerOutput

RequestId = int
KVCacheValue = torch.Tensor | Sequence[torch.Tensor]


class KVConnectorRole(enum.Enum):
    """Process role of a KV connector instance."""

    SCHEDULER = enum.auto()
    WORKER = enum.auto()


class KVConnectorMetadata(ABC):
    """Scheduler-to-worker metadata for one engine step.

    Implementations must remain serializable because distributed executors may send an instance to model workers through
    multiprocessing RPC.
    """


@dataclass
class KVConnectorOutput:
    """Terminal worker-side KV-transfer operations.

    Operation IDs, rather than user request IDs, are used because one request
    can own more than one asynchronous transfer generation.  Load failures are
    terminal load completions as well: the scheduler can release the transfer
    pin and fall back to normal model computation once every TP rank has
    reached a terminal state.
    """

    completed_save_ids: set[RequestId] = field(default_factory=set)
    completed_load_ids: set[RequestId] = field(default_factory=set)
    failed_load_ids: set[RequestId] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Own mutable, integer-normalized ID sets after deserialization."""
        self.completed_save_ids = {int(op_id) for op_id in self.completed_save_ids}
        self.completed_load_ids = {int(op_id) for op_id in self.completed_load_ids}
        self.failed_load_ids = {int(op_id) for op_id in self.failed_load_ids}
        self.completed_load_ids.update(self.failed_load_ids)

    def __bool__(self) -> bool:
        """Return whether this output contains a terminal operation."""
        return bool(self.completed_save_ids or self.completed_load_ids or self.failed_load_ids)

    @property
    def finished_sending(self) -> set[RequestId]:
        """Compatibility alias for connectors ported from vLLM."""
        return self.completed_save_ids

    @property
    def finished_recving(self) -> set[RequestId]:
        """Compatibility alias retaining vLLM's historical spelling."""
        return self.completed_load_ids


class KVConnectorBase(ABC):
    """Common scheduler- and worker-side contract for KV connectors."""

    def __init__(self, role: KVConnectorRole) -> None:
        if not isinstance(role, KVConnectorRole):
            raise TypeError(f'role must be a KVConnectorRole, got {type(role).__name__}')
        self._role = role
        self._connector_metadata: KVConnectorMetadata | None = None

    @property
    def role(self) -> KVConnectorRole:
        """Return the role of this connector instance."""
        return self._role

    # Worker-side metadata lifecycle.

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """Bind scheduler metadata before executing one model step."""
        self._connector_metadata = connector_metadata

    def clear_connector_metadata(self) -> None:
        """Clear scheduler metadata after executing one model step."""
        self._connector_metadata = None

    def _get_connector_metadata(self) -> KVConnectorMetadata:
        """Return currently bound metadata for use inside a connector."""
        assert self._connector_metadata is not None, 'connector metadata is not bound'
        return self._connector_metadata

    def has_connector_metadata(self) -> bool:
        """Return whether scheduler metadata is currently bound."""
        return self._connector_metadata is not None

    # Worker-side methods.

    def register_kv_caches(self, kv_caches: Mapping[str, KVCacheValue]) -> None:
        """Register GPU KV-cache tensors with the external store.

        This is a no-op for connectors that do not require memory registration. Implementations must not retain
        temporary tensor views in a way that changes ownership of the underlying cache allocation.
        """
        return None

    def handle_preemptions(self, connector_metadata: KVConnectorMetadata) -> None:
        """Handle preempted requests before their GPU blocks are
        overwritten."""
        return None

    def has_pending_step_transfers(self) -> bool:
        """Return whether the bound metadata contains work to submit.

        Connectors should override this when an empty metadata object is common so callers can avoid creating an
        unnecessary device-readiness fence.
        """
        return self.has_connector_metadata()

    def has_pending_step_loads(self) -> bool:
        """Return whether bound metadata contains loads to submit."""
        return False

    def has_pending_step_saves(self) -> bool:
        """Return whether bound metadata contains saves to submit.

        The default delegates to the original combined hook so connectors implemented against the save-only interface
        remain compatible.
        """
        return self.has_pending_step_transfers()

    def submit_loads(self) -> None:
        """Submit bound load work before the model forward is queued."""
        return None

    def submit_saves(
        self,
        *,
        save_ready_event: Any | None = None,
    ) -> None:
        """Submit bound save work after the model forward is queued."""
        self.submit_transfers(save_ready_event=save_ready_event)

    def submit_transfers(
        self,
        *,
        save_ready_event: Any | None = None,
    ) -> None:
        """Submit transfers described by the currently bound step metadata.

        ``save_ready_event`` is a worker-local CUDA event recorded after the
        model has queued the writes that produce this step's KV cache. It must
        never be embedded in scheduler metadata because that metadata is
        serialized for distributed workers.
        """
        return None

    def get_finished(
        self,
        finished_req_ids: set[RequestId],
        *,
        ready_event: Any | None = None,
    ) -> KVConnectorOutput:
        """Compatibility wrapper combining submission and completion polling.

        ``finished_req_ids`` is retained for connectors ported from vLLM. New
        LMDeploy integrations should submit via :meth:`submit_loads` and
        :meth:`submit_saves`, then collect sticky completions via
        :meth:`poll_finished`. A connector may use operation identifiers
        rather than user request identifiers when one request has multiple
        concurrent transfer waves.
        """
        del finished_req_ids
        self.submit_loads()
        self.submit_saves(save_ready_event=ready_event)
        return self.poll_finished()

    def poll_finished(
        self,
        acknowledged_sending: set[RequestId] | None = None,
        acknowledged_recving: set[RequestId] | None = None,
    ) -> KVConnectorOutput:
        """Poll sticky transfer completions without executing a model step.

        Distributed executors use this connector-only hook after the final
        forward as well as between normal forwards.  Implementations should
        retain local completions until they appear in
        ``acknowledged_sending`` or ``acknowledged_recving`` so completions
        from different tensor-parallel ranks cannot be lost across polling
        rounds.
        """
        return KVConnectorOutput()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return GPU block IDs whose external-cache load failed."""
        return set()

    def shutdown(self) -> None:
        """Drain asynchronous work and release connector resources."""
        return None

    # Scheduler-side methods.

    @abstractmethod
    def get_num_new_matched_tokens(
        self,
        request: 'SchedulerSequence',
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Return the external prefix hit beyond local computed tokens.

        The first value is ``None`` while an asynchronous lookup is pending.
        The second value indicates that loading will continue asynchronously
        across scheduler steps; it must be false when the hit length is zero.
        This method may be called repeatedly and must be side-effect free until
        a lookup result is available.
        """
        raise NotImplementedError

    @abstractmethod
    def update_state_after_alloc(
        self,
        request: 'SchedulerSequence',
        block_ids: Sequence[int],
        num_external_tokens: int,
        generation: int = 0,
    ) -> Any | None:
        """Update connector state after GPU blocks are allocated for a request.

        ``block_ids`` are physical scheduler block IDs. A connector is
        responsible for translating them to the cache engine's kernel-page
        layout when the two block sizes differ. ``generation`` distinguishes a
        reused request ID after cancellation or recompute preemption. For an
        asynchronous load, the scheduler may call this hook once for the
        externally matched range and again after the transfer when it
        allocates additional compute blocks. The returned connector-specific
        plan, when present, is kept sticky by the scheduler until it reaches a
        terminal worker completion.
        """
        raise NotImplementedError

    @abstractmethod
    def build_connector_meta(self, scheduler_output: 'SchedulerOutput') -> KVConnectorMetadata:
        """Build serializable worker metadata for the current scheduler step.

        Implementations must not mutate ``scheduler_output``. They may consume
        and reset connector-owned per-step bookkeeping while building the
        returned metadata.
        """
        raise NotImplementedError

    def on_new_request(self, request: 'SchedulerSequence') -> None:
        """Record a newly admitted request when connector bookkeeping needs
        it."""
        return None

    def update_connector_output(self, connector_output: Any) -> None:
        """Consume an executor-aggregated output from worker connectors."""
        return None

    def request_finished(
        self,
        request: 'SchedulerSequence',
        block_ids: Sequence[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Notify the connector before a finished request's blocks are freed.

        The boolean is true when the connector temporarily takes ownership of
        the blocks for an asynchronous save. In that case the scheduler must
        defer freeing them until the request is reported by ``get_finished``.
        The optional dictionary is reserved for connector-specific response
        metadata.
        """
        return False, None

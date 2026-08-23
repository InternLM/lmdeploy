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

    def get_finished(
        self,
        finished_req_ids: set[RequestId],
    ) -> tuple[set[RequestId] | None, set[RequestId] | None]:
        """Return requests whose asynchronous save/load has completed.

        The tuple order is ``(finished_sending, finished_receiving)``. A
        request returned in ``finished_sending`` must have appeared in
        ``finished_req_ids`` in this call or an earlier call. The receiving set
        reports requests whose asynchronous external-cache load is complete.
        """
        return None, None

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
    ) -> None:
        """Update connector state after GPU blocks are allocated for a request.

        ``block_ids`` are physical scheduler block IDs. A connector is
        responsible for translating them to the cache engine's kernel-page
        layout when the two block sizes differ. For an asynchronous load, the
        scheduler may call this hook once for the externally matched range and
        again after the transfer when it allocates additional compute blocks.
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

    def is_lookup_pending(self, request_id: RequestId) -> bool:
        """Return whether an asynchronous prefix lookup is still running."""
        return False

    def cancel_lookup(self, request_id: RequestId) -> None:
        """Discard scheduler-side lookup state for an aborted request."""
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

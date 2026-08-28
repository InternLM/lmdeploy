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
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from lmdeploy.pytorch.messages import SchedulerSequence
    from lmdeploy.pytorch.paging.scheduler import SchedulerOutput

RequestId = int
KVOperationId = int
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

    def get_save_block_leases(self) -> tuple['KVSaveBlockLease', ...]:
        """Return scheduler-owned block leases required by this step."""
        return ()


@dataclass(frozen=True)
class KVSaveBlockLease:
    """Logical blocks kept alive until one save operation completes."""

    operation_id: KVOperationId
    logical_block_ids: tuple[int, ...]


@dataclass
class KVConnectorOutput:
    """Worker connector progress returned by one executor step.

    Completion sets are rank-local until the executor aggregates every TP
    worker. Load completions use request IDs because a request is paused while
    its single active load completes. Save completions need operation IDs
    because chunked prefill can enqueue several concurrent saves for the same
    request. Save completion is terminal whether the Store write succeeded or
    failed, allowing the scheduler to release its block lease in either case.
    ``invalid_block_ids`` may arrive before the request-level receive
    completion and is therefore consumed by the scheduler-side connector.
    """

    completed_save_ids: set[KVOperationId] | None = None
    finished_receiving: set[RequestId] | None = None
    invalid_block_ids: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class KVLoadResult:
    """Backend-neutral terminal result for one asynchronous KV load.

    Attributes:
        request_id: Key-like identity of the load. It is the owning
            ``SchedulerSequence.seq_id``; a request has at most one active
            external load, so no separate load-operation ID is needed.
        success: Value-like terminal status. ``True`` lets paging publish the
            loaded prefix, while ``False`` makes paging discard it and fall
            back to local computation.
    """

    request_id: RequestId
    success: bool


@dataclass(frozen=True)
class KVConnectorResult:
    """Backend-neutral updates passed from a connector to paging.

    Attributes:
        load_results: Terminal load records. In each record,
            ``KVLoadResult.request_id`` is the key-like request identity and
            ``KVLoadResult.success`` is its value-like outcome.
        completed_save_ids: Terminal save operation IDs. Each element matches a ``KVSaveBlockLease``
            ``operation_id``; membership alone tells paging to release that
            operation's pinned logical blocks. There is no value because save
            completion is terminal whether the Store write succeeded or
            failed. One request may own several IDs during chunked prefill.
    """

    load_results: tuple[KVLoadResult, ...] = ()
    completed_save_ids: frozenset[KVOperationId] = frozenset()


class KVConnectorOutputAggregator:
    """Accumulate rank-local completions until every worker reports them.

    Worker connectors return only completions observed since their previous
    poll. TP workers may finish the same operation on different engine steps,
    so this executor-owned object remembers which worker slots have reported
    each ID. An ID is emitted once all ``world_size`` slots have reported it,
    then its accumulated state is removed.

    Save and load completions use different identities: saves use operation
    IDs because one request may have several concurrent save waves, while loads
    use request IDs because one request has at most one active external load.
    """

    def __init__(self, world_size: int) -> None:
        # Number of rank-local outputs required to publish one completion.
        self.world_size = world_size
        # Save operation ID -> executor worker slots that reported it terminal.
        # The slots are positions in ``outputs``, not distributed global ranks.
        self._saving_ranks: dict[KVOperationId, set[int]] = {}
        # Load request ID -> executor worker slots that finished receiving it.
        # Load success is carried separately by ``invalid_block_ids``.
        self._receiving_ranks: dict[RequestId, set[int]] = {}

    def _aggregate_completions(
        self,
        rank_completions: Sequence[set[RequestId] | None],
        rank_state: dict[RequestId, set[int]],
    ) -> set[RequestId] | None:
        """Return IDs newly reported by every worker slot.

        ``rank_completions[rank]`` contains the IDs newly reported by that
        worker slot. ``rank_state`` maps each ID to all slots accumulated over
        previous calls. Completed IDs are removed from ``rank_state`` after
        being returned. Although the annotation uses ``RequestId``, the same
        integer-ID logic is also used for save operation IDs.
        """
        completed = set()
        for rank, request_ids in enumerate(rank_completions):
            for request_id in request_ids or ():
                ranks = rank_state.setdefault(request_id, set())
                ranks.add(rank)
                if len(ranks) == self.world_size:
                    completed.add(request_id)
        for request_id in completed:
            rank_state.pop(request_id, None)
        return completed or None

    def aggregate(
        self,
        outputs: Sequence[KVConnectorOutput | None],
    ) -> KVConnectorOutput:
        """Merge one rank-local output per worker into all-rank progress.

        Save/load completions wait for every worker. Invalid destination block IDs are unioned and published immediately
        because one rank failure is sufficient to make a load unusable; the scheduler-side connector keeps them until
        the corresponding request-level completion arrives.
        """
        if len(outputs) != self.world_size:
            raise ValueError(
                f'expected {self.world_size} TP connector outputs, got {len(outputs)}')
        saving_by_rank: list[set[KVOperationId] | None] = []
        receiving_by_rank: list[set[RequestId] | None] = []
        invalid_block_ids = set()
        for output in outputs:
            if output is None:
                saving_by_rank.append(None)
                receiving_by_rank.append(None)
                continue
            saving_by_rank.append(output.completed_save_ids)
            receiving_by_rank.append(output.finished_receiving)
            invalid_block_ids.update(output.invalid_block_ids)
        return KVConnectorOutput(
            completed_save_ids=self._aggregate_completions(
                saving_by_rank,
                self._saving_ranks,
            ),
            finished_receiving=self._aggregate_completions(
                receiving_by_rank,
                self._receiving_ranks,
            ),
            invalid_block_ids=invalid_block_ids,
        )

    def clear(self) -> None:
        """Discard partial completions when pending executor work is reset."""
        self._saving_ranks.clear()
        self._receiving_ranks.clear()


class KVConnectorBase(ABC):
    """Common scheduler- and worker-side contract for KV connectors."""

    def __init__(self, role: KVConnectorRole) -> None:
        if not isinstance(role, KVConnectorRole):
            raise TypeError(f'role must be a KVConnectorRole, got {type(role).__name__}')
        self._connector_metadata: KVConnectorMetadata | None = None

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

    # Worker-side methods.

    def register_kv_caches(self, kv_caches: Mapping[str, KVCacheValue]) -> None:
        """Register GPU KV-cache tensors with the external store.

        This is a no-op for connectors that do not require memory registration. Implementations must not retain
        temporary tensor views in a way that changes ownership of the underlying cache allocation.
        """
        return None

    def start_load_kv(self) -> None:
        """Submit loads described by the currently bound metadata."""
        return None

    def start_save_kv(self) -> None:
        """Submit saves after the current model forward has been queued."""
        return None

    def get_finished(self) -> KVConnectorOutput:
        """Return rank-local terminal transfer progress since the last poll."""
        return KVConnectorOutput()

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
        This method may be called repeatedly. An implementation may submit one
        request-local lookup, but repeated polls must not duplicate that work
        or mutate paging state before a result is available.
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
        layout when the two block sizes differ. For an asynchronous load they
        cover exactly the externally matched destination range.
        """
        raise NotImplementedError

    @abstractmethod
    def build_connector_meta(self, scheduler_output: 'SchedulerOutput') -> KVConnectorMetadata | None:
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

    def update_connector_output(
        self,
        connector_output: KVConnectorOutput,
    ) -> KVConnectorResult:
        """Consume all-TP worker progress and return backend-neutral
        updates."""
        return KVConnectorResult()

    def finish_transfers_after_worker_drain(self) -> None:
        """Drop scheduler-side transfer state after workers have drained."""
        return None

    def request_finished(
        self,
        request: 'SchedulerSequence',
    ) -> None:
        """Discard connector-owned state when a request is removed."""
        return None

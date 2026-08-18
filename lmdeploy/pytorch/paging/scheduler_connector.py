# Copyright (c) OpenMMLab. All rights reserved.
"""Scheduler-side KV connector lifecycle and paging ownership."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from lmdeploy.pytorch.messages import MessageStatus, SchedulerSequence
from lmdeploy.utils import get_logger

if TYPE_CHECKING:
    from lmdeploy.pytorch.kv_connector.base import KVConnectorBase

logger = get_logger('lmdeploy')

MapType = dict[int, int]
SeqList = list[SchedulerSequence]

@dataclass
class SchedulerOutput:
    """Output of schedule."""

    running: SeqList
    swap_in_map: MapType
    swap_out_map: MapType
    copy_map: MapType
    connector_token_lens: tuple[int, ...] = ()
    connector_block_ids: tuple[tuple[int, ...], ...] = ()
    connector_logical_block_ids: tuple[tuple[int, ...], ...] = ()
    connector_generations: tuple[int, ...] = ()
    preempted_save_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class _PendingKVSave:
    """Scheduler-owned logical-block pins for one async save wave."""

    req_id: int
    generation: int
    logical_block_ids: np.ndarray


@dataclass
class _PendingKVLoad:
    """Paging ownership retained until one async load reaches all-TP
    terminal."""

    seq: SchedulerSequence
    req_id: int
    load_id: int
    generation: int
    local_token_len: int
    remote_token_len: int
    logical_block_ids: np.ndarray
    prefill_target_blocks: int
    fallback_step: int
    fallback_logical_block_ids: np.ndarray
    fallback_restored: bool = False
    dispatching: bool = False
    dispatched: bool = False
    cancelled: bool = False
    needs_fence: bool = False

    @property
    def prefill_reservation_blocks(self) -> int:
        """Unallocated rows still owed to this request's complete prefill."""
        return max(0, self.prefill_target_blocks - int(self.seq.num_blocks))


@dataclass(frozen=True)
class _RemoteKVLoadPlan:
    """Validated token and block bounds for one remote-load allocation."""

    local_token_len: int
    remote_token_len: int
    fallback_step: int
    prealloc_size: int
    local_blocks: int
    remote_blocks: int


class KVConnectorSchedulerMixin:
    """Own remote KV transfer state without bloating the core scheduler."""

    def _init_kv_connector_scheduler(
        self,
        kv_connector: 'KVConnectorBase | None',
    ) -> None:
        self.kv_connector = kv_connector
        # Async connector saves own one reference to every logical block they
        # read. Request eviction can release its reference immediately without
        # allowing physical rows to be reused before all TP workers finish.
        self._kv_seq_generations: dict[int, int] = {}
        # More than one eviction layer can report the same failed admission in
        # one scheduler tick. Track the tick to advance generation only once.
        self._kv_preemption_ticks: dict[int, int] = {}
        self._pending_kv_saves: dict[int, _PendingKVSave] = {}
        self._pending_kv_loads: dict[int, _PendingKVLoad] = {}
        self._active_kv_load_by_req: dict[int, int] = {}
        # A successful load retains one-shot admission protection plus enough
        # headroom for the full prefill, including long-context chunks.
        self._remote_prefill_reservations: dict[int, int] = {}
        self._prefill_reservation_targets: dict[
            int, tuple[SchedulerSequence, int]
        ] = {}

    def _has_remote_prefill_reservation(self, seq: SchedulerSequence) -> bool:
        """Whether a completed remote load still awaits its first forward."""
        return int(seq.seq_id) in self._remote_prefill_reservations

    def _consume_remote_prefill_reservation(self, seq: SchedulerSequence) -> None:
        """Consume only the completed-load first-admission marker."""
        self._remote_prefill_reservations.pop(int(seq.seq_id), None)

    def _release_prefill_reservation(self, seq: SchedulerSequence) -> None:
        """Release both remote-ready protection and full-prefill headroom."""
        req_id = int(seq.seq_id)
        self._remote_prefill_reservations.pop(req_id, None)
        self._prefill_reservation_targets.pop(req_id, None)

    def _prefill_target_blocks(
        self,
        seq: SchedulerSequence,
        prealloc_size: int,
    ) -> int:
        """Return the absolute rows required by the complete prompt."""
        block_size = self.cache_config.block_size
        target_tokens = int(seq.num_all_ids) + max(0, int(prealloc_size))
        return (target_tokens + block_size - 1) // block_size

    def _register_prefill_reservation(
        self,
        seq: SchedulerSequence,
        *,
        prealloc_size: int = 0,
        target_blocks: int | None = None,
    ) -> None:
        """Retain complete-ISL capacity until the final prefill completes."""
        if self.kv_connector is None:
            return
        req_id = int(seq.seq_id)
        if target_blocks is None:
            target_blocks = self._prefill_target_blocks(seq, prealloc_size)
        previous = self._prefill_reservation_targets.get(req_id)
        if previous is not None:
            target_blocks = max(int(target_blocks), int(previous[1]))
        self._prefill_reservation_targets[req_id] = (seq, int(target_blocks))

    def _release_completed_prefill_reservations(
        self,
        seqs: SeqList,
    ) -> None:
        """Release capacity only after scheduler-visible final-prefill output.

        Long-context intermediate chunks advance ``num_history_ids`` before
        their forward finishes.  The last chunk does not: its model output
        advances the sequence to ``input_end_pos``.  This boundary therefore
        distinguishes an allocated final chunk from a completed one.
        """
        for seq in seqs:
            req_id = int(seq.seq_id)
            if req_id not in self._prefill_reservation_targets:
                continue
            if self._has_remote_prefill_reservation(seq):
                continue
            if int(seq.num_history_ids) >= int(seq.input_end_pos):
                self._prefill_reservation_targets.pop(req_id, None)

    def _exclude_remote_prefill_victims(self, seqs: SeqList) -> SeqList:
        """Keep successfully loaded waiters out of every eviction path."""
        return [
            seq for seq in seqs
            if not self._has_remote_prefill_reservation(seq)
        ]

    def _remote_prefill_reserved_blocks(
        self,
        *,
        exclude_req_id: int | None = None,
    ) -> int:
        """Return headroom owed to other in-flight prefills.

        Targets cover the entire remaining input, not merely the next chunk. They shrink dynamically as blocks are
        allocated. Pending remote loads are not registered until completion and are counted separately.
        """
        total = sum(
            max(0, int(target_blocks) - int(seq.num_blocks))
            for req_id, (seq, target_blocks)
            in self._prefill_reservation_targets.items()
            if req_id != exclude_req_id
        )
        total += sum(
            int(pending.prefill_reservation_blocks)
            for pending in self._pending_kv_loads.values()
            if not pending.cancelled and pending.req_id != exclude_req_id
        )
        return total

    def _estimate_remote_prefill_reservation(
        self,
        seq: SchedulerSequence,
        *,
        remote_token_len: int,
        prealloc_size: int,
    ) -> int:
        """Return rows needed to reach the complete-prefill allocation target.

        ``remote_token_len`` is validated by the caller and retained in the
        signature to make the admission contract explicit.  The reservation
        intentionally covers every remaining ISL chunk (vLLM #44560).
        """
        if remote_token_len > int(seq.num_all_ids):
            raise ValueError('remote prefix exceeds request token length')
        target_blocks = self._prefill_target_blocks(seq, prealloc_size)
        return max(0, target_blocks - int(seq.num_blocks))

    def shutdown(self) -> None:
        """Release scheduler-side connector resources exactly once."""
        connector = getattr(self, 'kv_connector', None)
        self.kv_connector = None
        try:
            if connector is not None:
                connector.shutdown()
        finally:
            self._release_all_kv_save_pins()
            self._release_all_kv_load_pins()

    def _plan_remote_kv_load(
        self,
        seq: SchedulerSequence,
        *,
        local_token_len: int,
        remote_token_len: int,
        fallback_step: int,
        prealloc_size: int,
    ) -> _RemoteKVLoadPlan:
        """Validate immutable remote-load bounds before mutating paging."""
        block_size = self.cache_config.block_size
        if (local_token_len % block_size != 0
                or remote_token_len % block_size != 0
                or remote_token_len <= local_token_len):
            raise ValueError('remote KV token bounds must be increasing and block aligned')
        if remote_token_len > int(seq.num_all_ids):
            raise ValueError('remote KV prefix exceeds request token length')
        if not local_token_len <= fallback_step < local_token_len + block_size:
            raise ValueError('fallback_step must be within the local boundary block')
        return _RemoteKVLoadPlan(
            local_token_len=local_token_len,
            remote_token_len=remote_token_len,
            fallback_step=fallback_step,
            prealloc_size=prealloc_size,
            local_blocks=local_token_len // block_size,
            remote_blocks=remote_token_len // block_size,
        )

    def _start_remote_kv_load(
        self,
        seq: SchedulerSequence,
        *,
        local_token_len: int,
        remote_token_len: int,
        fallback_step: int,
        prealloc_size: int,
    ) -> Any:
        """Plan and allocate private destinations for one remote load."""
        plan = self._plan_remote_kv_load(
            seq,
            local_token_len=local_token_len,
            remote_token_len=remote_token_len,
            fallback_step=fallback_step,
            prealloc_size=prealloc_size,
        )
        return self._apply_remote_kv_load_plan(seq, plan)

    def _apply_remote_kv_load_plan(
        self,
        seq: SchedulerSequence,
        plan: _RemoteKVLoadPlan,
    ) -> Any:
        """Apply a validated plan while preserving the partial-tail
        fallback."""
        connector = self.kv_connector
        if connector is None:
            raise RuntimeError('remote KV allocation requires a connector')
        local_token_len = plan.local_token_len
        remote_token_len = plan.remote_token_len
        fallback_step = plan.fallback_step
        prealloc_size = plan.prealloc_size
        local_blocks = plan.local_blocks
        remote_blocks = plan.remote_blocks
        fallback_ids = np.empty((0, ), dtype=np.int64)
        fallback_detached = False
        logical_ids = np.empty((0, ), dtype=np.int64)
        load_suffix_pinned = False
        generation = self._kv_seq_generations.setdefault(int(seq.seq_id), 0)
        load_request = None
        try:
            if fallback_step > local_token_len:
                all_logical_ids = seq.logical_blocks.get_real_blocks()
                if len(all_logical_ids) <= local_blocks:
                    raise RuntimeError(
                        f'request {seq.seq_id} has no partial fallback block '
                        f'at index {local_blocks}')
                fallback_ids = np.asarray(
                    all_logical_ids[local_blocks:local_blocks + 1],
                    dtype=np.int64,
                ).copy()
                self.block_manager.pin_logical_blocks(fallback_ids)
                self.block_manager.truncate(seq, local_blocks)
                fallback_detached = True

            self.block_manager.allocate(seq, prealloc_size)
            all_logical_ids = seq.logical_blocks.get_real_blocks()
            if len(all_logical_ids) < remote_blocks:
                raise RuntimeError(
                    f'request {seq.seq_id} owns {len(all_logical_ids)} blocks, '
                    f'but remote load needs {remote_blocks}')
            logical_ids = np.asarray(
                all_logical_ids[local_blocks:remote_blocks],
                dtype=np.int64,
            ).copy()
            physical_ids = self.block_manager.pin_logical_blocks(logical_ids)
            load_suffix_pinned = True
            load_request = connector.update_state_after_alloc(
                seq,
                tuple(int(block_id) for block_id in physical_ids),
                remote_token_len - local_token_len,
                generation=generation,
            )
            if load_request is None:
                raise RuntimeError(
                    f'connector did not create a load plan for request {seq.seq_id}')
            expected = (
                int(seq.seq_id),
                generation,
                local_token_len,
                remote_token_len,
            )
            actual = (
                int(load_request.req_id),
                int(load_request.generation),
                int(load_request.local_token_len),
                int(load_request.remote_token_len),
            )
            if actual != expected:
                raise RuntimeError(
                    f'connector load plan mismatch for request {seq.seq_id}: '
                    f'{actual} != {expected}')
            load_id = int(load_request.load_id)
            if load_id in self._pending_kv_loads:
                raise RuntimeError(f'duplicate connector load_id {load_id}')
            req_id = int(seq.seq_id)
            if req_id in self._active_kv_load_by_req:
                raise RuntimeError(f'request {req_id} already has an active remote load')
            if tuple(int(block_id) for block_id in load_request.block_ids) != tuple(
                    int(block_id) for block_id in physical_ids):
                raise RuntimeError(
                    f'connector physical load suffix changed for request {seq.seq_id}')
            prefill_target_blocks = self._prefill_target_blocks(
                seq, prealloc_size)
            self._pending_kv_loads[load_id] = _PendingKVLoad(
                seq=seq,
                req_id=int(seq.seq_id),
                load_id=load_id,
                generation=generation,
                local_token_len=local_token_len,
                remote_token_len=remote_token_len,
                logical_block_ids=logical_ids,
                prefill_target_blocks=prefill_target_blocks,
                fallback_step=fallback_step,
                fallback_logical_block_ids=fallback_ids,
            )
            self._active_kv_load_by_req[req_id] = load_id
            seq.state.begin_remote_load()
            return load_request
        except BaseException:
            if load_request is not None:
                load_id = int(load_request.load_id)
                self._pending_kv_loads.pop(load_id, None)
                if self._active_kv_load_by_req.get(int(seq.seq_id)) == load_id:
                    self._active_kv_load_by_req.pop(int(seq.seq_id), None)
                connector.update_connector_output({
                    'cancelled_load_ids': {load_id},
                })
            if load_suffix_pinned:
                self.block_manager.release_pinned_logical_blocks(logical_ids)
            self.block_manager.truncate(seq, local_blocks)
            if fallback_detached:
                # Adopt the fallback pin as the restored sequence reference.
                seq.logical_blocks.append(fallback_ids)
            elif len(fallback_ids) > 0:
                self.block_manager.release_pinned_logical_blocks(fallback_ids)
            seq.set_step(min(fallback_step, seq.num_all_ids))
            seq.kv_token_limit = None
            raise

    def mark_kv_connector_preempted(self, seq: SchedulerSequence) -> None:
        """Start a new connector generation before a sequence recomputes.

        A request ID survives recompute preemption in LMDeploy.  A generation therefore distinguishes work submitted
        against the old block table from work submitted after the request is allocated again.  Existing jobs remain
        pinned until workers acknowledge their save IDs.
        """
        self._release_prefill_reservation(seq)
        if self.kv_connector is None:
            return

        req_id = int(seq.seq_id)
        self._cancel_remote_kv_load(seq, make_waiting=True)
        if self._kv_preemption_ticks.get(req_id) == self.scheduler_tick:
            return
        self._kv_preemption_ticks[req_id] = self.scheduler_tick
        old_generation = self._kv_seq_generations.get(req_id, 0)
        self._kv_seq_generations[req_id] = old_generation + 1

    def build_kv_connector_metadata(
        self,
        running: SeqList,
        token_lens: tuple[int, ...] | None = None,
    ) -> Any | None:
        """Build and pin metadata for one executor dispatch.

        ``token_lens`` is supplied only for prefill-like forwards.  Decode
        still emits empty metadata so preemption notifications reach workers,
        but it never creates save work: its token IDs can be one step stale due
        to engine prefetching.
        """
        connector = self.kv_connector
        if connector is None:
            return None

        if token_lens is None:
            token_lens = ()
        if len(token_lens) not in (0, len(running)):
            raise ValueError('token_lens must be empty or contain one value per running sequence')

        aligned_token_lens: list[int] = []
        logical_block_ids: list[tuple[int, ...]] = []
        physical_block_ids: list[tuple[int, ...]] = []
        generations: list[int] = []
        snapshots: dict[int, tuple[np.ndarray, tuple[int, ...], int]] = {}
        block_size = self.cache_config.block_size

        if token_lens:
            for seq, token_len_value in zip(running, token_lens):
                token_len = int(token_len_value)
                if token_len < 0:
                    raise ValueError('connector token length must be non-negative')
                token_len = token_len // block_size * block_size
                num_blocks = token_len // block_size

                all_logical_ids = seq.logical_blocks.get_real_blocks()
                if num_blocks > len(all_logical_ids):
                    raise RuntimeError(
                        f'request {seq.seq_id} has {len(all_logical_ids)} allocated blocks '
                        f'but connector save needs {num_blocks}')
                logical_ids = np.asarray(all_logical_ids[:num_blocks], dtype=np.int64).copy()
                physical_ids = self.block_manager.resolve_gpu_block_offsets(logical_ids)
                physical_tuple = tuple(int(block_id) for block_id in physical_ids)
                logical_tuple = tuple(int(block_id) for block_id in logical_ids)
                generation = self._kv_seq_generations.setdefault(int(seq.seq_id), 0)

                aligned_token_lens.append(token_len)
                logical_block_ids.append(logical_tuple)
                physical_block_ids.append(physical_tuple)
                generations.append(generation)
                snapshots[int(seq.seq_id)] = (logical_ids, physical_tuple, generation)

        scheduler_output = SchedulerOutput(
            running=running,
            swap_in_map={},
            swap_out_map={},
            copy_map={},
            connector_token_lens=tuple(aligned_token_lens),
            connector_block_ids=tuple(physical_block_ids),
            connector_logical_block_ids=tuple(logical_block_ids),
            connector_generations=tuple(generations),
            # Pinned old-generation jobs always drain to completion.  Do not
            # let a worker that has not enqueued the job yet interpret a
            # preemption notice as permission to acknowledge it early.
            preempted_save_ids=(),
        )
        metadata = connector.build_connector_meta(scheduler_output)
        try:
            self._lease_kv_load_metadata(metadata)
            self._pin_kv_connector_metadata(metadata, snapshots)
        except BaseException:
            self.rollback_kv_connector_metadata(metadata)
            raise
        return metadata

    def build_kv_connector_progress_metadata(self) -> Any | None:
        """Build connector-only work when there is no model forward.

        Only ready loads are emitted. Building acquires a temporary dispatch
        lease; the engine commits successful delivery with
        :meth:`mark_kv_connector_metadata_dispatched`.  A failed RPC therefore
        rolls the lease back and retries the same load ID without serializing
        in-flight work every poll.
        """
        connector = self.kv_connector
        if connector is None or not self._has_ready_kv_loads():
            return None
        scheduler_output = SchedulerOutput(
            running=[],
            swap_in_map={},
            swap_out_map={},
            copy_map={},
        )
        metadata = connector.build_connector_meta(scheduler_output)
        if not getattr(metadata, 'load_requests', ()):
            return None
        try:
            self._lease_kv_load_metadata(metadata)
        except BaseException:
            self.rollback_kv_connector_metadata(metadata)
            raise
        return metadata

    def _lease_kv_load_metadata(self, metadata: Any) -> None:
        """Protect the build-to-RPC await window from cancellation."""
        leased_ids = []
        try:
            for load_request in getattr(metadata, 'load_requests', ()):
                load_id = int(load_request.load_id)
                pending = self._pending_kv_loads.get(load_id)
                if pending is None:
                    raise RuntimeError(
                        f'connector metadata refers to unknown load_id {load_id}')
                if ((pending.cancelled and not pending.needs_fence)
                        or pending.dispatching or pending.dispatched):
                    raise RuntimeError(
                        f'connector load_id {load_id} is not ready for dispatch')
                pending.dispatching = True
                leased_ids.append(load_id)
        except BaseException:
            for load_id in leased_ids:
                pending = self._pending_kv_loads.get(load_id)
                if pending is not None:
                    pending.dispatching = False
            raise

    def mark_kv_connector_metadata_dispatched(self, metadata: Any | None) -> None:
        """Commit metadata only after its executor RPC was accepted."""
        if metadata is None:
            return
        load_requests = tuple(getattr(metadata, 'load_requests', ()) or ())
        if not load_requests:
            return
        connector = self.kv_connector
        if connector is not None:
            connector.mark_connector_meta_dispatched(metadata)
        for load_request in load_requests:
            pending = self._pending_kv_loads.get(int(load_request.load_id))
            if pending is None:
                continue
            if (pending.req_id != int(load_request.req_id)
                    or pending.generation != int(load_request.generation)):
                raise RuntimeError(
                    f'stale connector load dispatch for load_id {load_request.load_id}')
            pending.dispatching = False
            pending.dispatched = True
            pending.needs_fence = False

    def _has_ready_kv_loads(self) -> bool:
        return any(
            not pending.dispatching and not pending.dispatched
            and (not pending.cancelled or pending.needs_fence)
            for pending in self._pending_kv_loads.values()
        )

    def _pin_kv_connector_metadata(
        self,
        metadata: Any,
        snapshots: dict[int, tuple[np.ndarray, tuple[int, ...], int]],
    ) -> None:
        """Acquire one allocator reference for every metadata save wave."""
        save_requests = getattr(metadata, 'save_requests', ())
        pinned_save_ids: list[int] = []
        try:
            for save_request in save_requests:
                save_id = int(save_request.save_id)
                req_id = int(save_request.req_id)
                generation = int(save_request.generation)
                if save_id in self._pending_kv_saves:
                    raise RuntimeError(f'duplicate connector save_id {save_id}')
                if req_id not in snapshots:
                    raise RuntimeError(f'connector metadata refers to unscheduled request {req_id}')

                logical_ids, physical_ids, expected_generation = snapshots[req_id]
                num_blocks = int(save_request.token_len) // self.cache_config.block_size
                if int(save_request.token_len) % self.cache_config.block_size != 0:
                    raise ValueError('connector save token_len must be block aligned')
                if generation != expected_generation:
                    raise RuntimeError(
                        f'connector generation mismatch for request {req_id}: '
                        f'{generation} != {expected_generation}')
                if len(save_request.block_ids) != num_blocks or len(save_request.block_hashes) != num_blocks:
                    raise ValueError('connector save block IDs and hashes must match token_len')

                wave_logical_ids = logical_ids[:num_blocks].copy()
                if tuple(int(block_id) for block_id in save_request.block_ids) != physical_ids[:num_blocks]:
                    raise RuntimeError(f'connector physical block snapshot changed for request {req_id}')

                self.block_manager.pin_logical_blocks(wave_logical_ids)
                self._pending_kv_saves[save_id] = _PendingKVSave(
                    req_id=req_id,
                    generation=generation,
                    logical_block_ids=wave_logical_ids,
                )
                pinned_save_ids.append(save_id)
        except BaseException:
            self._release_kv_save_pins(pinned_save_ids)
            raise

    @staticmethod
    def _completed_kv_save_ids(connector_output: Any) -> set[int]:
        """Normalize executor connector output to completed save IDs."""
        if connector_output is None:
            return set()
        output = connector_output
        if isinstance(output, dict):
            output = output.get('completed_save_ids', output.get('finished_sending'))
        elif hasattr(output, 'completed_save_ids'):
            output = output.completed_save_ids
        elif hasattr(output, 'finished_sending'):
            output = output.finished_sending
        elif isinstance(output, tuple) and len(output) == 2:
            output = output[0]
        if output is None:
            return set()
        if isinstance(output, (set, frozenset, list, tuple)):
            return {int(save_id) for save_id in output}
        return set()

    def update_connector_output(self, connector_output: Any) -> None:
        """Consume all-TP completions and publish or roll back loaded KV."""
        completed_save_ids = self._completed_kv_save_ids(connector_output)
        completed_load_ids = self._completed_kv_load_ids(connector_output)
        failed_load_ids = self._failed_kv_load_ids(connector_output)
        connector = self.kv_connector
        try:
            if connector is not None:
                connector.update_connector_output(connector_output)
        finally:
            self._release_kv_save_pins(completed_save_ids)
            for load_id in completed_load_ids:
                self._finish_kv_load(
                    load_id,
                    failed=load_id in failed_load_ids,
                )

    @staticmethod
    def _output_ids(connector_output: Any, name: str) -> set[int]:
        if connector_output is None:
            return set()
        if isinstance(connector_output, dict):
            values = connector_output.get(name)
        else:
            values = getattr(connector_output, name, None)
        if values is None:
            return set()
        return {int(value) for value in values}

    @classmethod
    def _completed_kv_load_ids(cls, connector_output: Any) -> set[int]:
        completed = cls._output_ids(connector_output, 'completed_load_ids')
        if completed:
            return completed
        completed = cls._output_ids(connector_output, 'finished_recving')
        if completed:
            return completed
        if isinstance(connector_output, tuple) and len(connector_output) == 2:
            values = connector_output[1]
            if values is not None:
                return {int(value) for value in values}
        return set()

    @classmethod
    def _failed_kv_load_ids(cls, connector_output: Any) -> set[int]:
        return cls._output_ids(connector_output, 'failed_load_ids')

    def _finish_kv_load(self, load_id: int, *, failed: bool) -> None:
        pending = self._pending_kv_loads.pop(int(load_id), None)
        if pending is None:
            return
        seq = pending.seq
        current_generation = self._kv_seq_generations.get(pending.req_id)
        active = (
            not pending.cancelled
            and current_generation == pending.generation
            and self._active_kv_load_by_req.get(pending.req_id) == pending.load_id
            and seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS
        )
        if self._active_kv_load_by_req.get(pending.req_id) == pending.load_id:
            self._active_kv_load_by_req.pop(pending.req_id, None)
        try:
            if not active:
                if (not pending.fallback_restored
                        and seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS):
                    self._detach_remote_kv_suffix(pending)
                return
            if failed:
                self._detach_remote_kv_suffix(pending)
            else:
                try:
                    seq.kv_token_limit = pending.remote_token_len
                    if self.block_trie.enabled:
                        # Publish only after every TP worker reports that the
                        # destination suffix is completely populated.
                        self.block_trie.allocate(seq)
                    seq.set_step(pending.remote_token_len)
                    seq.kv_token_limit = None
                    if self.block_trie.enabled:
                        self._finish_prefix_cache_schedule(seq)
                    self._register_prefill_reservation(
                        seq,
                        target_blocks=pending.prefill_target_blocks,
                    )
                    self._remote_prefill_reservations[pending.req_id] = (
                        pending.prefill_reservation_blocks)
                except BaseException:
                    logger.exception(
                        'Failed to publish completed remote KV; falling back '
                        'to local prefix: req_id=%s load_id=%s',
                        pending.req_id,
                        pending.load_id,
                    )
                    self._detach_remote_kv_suffix(pending)
            seq.state.finish_remote_load()
            logger.info(
                'Mooncake load applied: req_id=%s load_id=%s generation=%s '
                'failed=%s step=%s',
                pending.req_id,
                pending.load_id,
                pending.generation,
                failed,
                seq.num_history_ids,
            )
        finally:
            self.block_manager.release_pinned_logical_blocks(
                pending.logical_block_ids)
            if (len(pending.fallback_logical_block_ids) > 0
                    and not pending.fallback_restored):
                self.block_manager.release_pinned_logical_blocks(
                    pending.fallback_logical_block_ids)

    def _detach_remote_kv_suffix(self, pending: _PendingKVLoad) -> None:
        """Drop the request reference while preserving any transfer pin."""
        seq = pending.seq
        self._release_prefill_reservation(seq)
        local_blocks = pending.local_token_len // self.cache_config.block_size
        if len(seq.logical_blocks) >= local_blocks:
            self.block_manager.truncate(seq, local_blocks)
        if (len(pending.fallback_logical_block_ids) > 0
                and not pending.fallback_restored):
            # The fallback pin becomes the sequence's restored ownership.
            seq.logical_blocks.append(pending.fallback_logical_block_ids)
            pending.fallback_restored = True
        seq.set_step(min(pending.fallback_step, seq.num_all_ids))
        seq.kv_token_limit = None

    def cancel_remote_kv_load(
        self,
        seq: SchedulerSequence,
        *,
        make_waiting: bool = True,
    ) -> None:
        """Cancel scheduling ownership without racing an in-flight device
        write."""
        self._cancel_remote_kv_load(seq, make_waiting=make_waiting)

    def _cancel_remote_kv_load(
        self,
        seq: SchedulerSequence,
        *,
        make_waiting: bool,
    ) -> None:
        self._release_prefill_reservation(seq)
        pending_loads = [
            pending
            for pending in self._pending_kv_loads.values()
            if pending.req_id == int(seq.seq_id) and not pending.cancelled
        ]
        if not pending_loads:
            return
        cancelled_ids = set()
        for pending in pending_loads:
            pending.cancelled = True
            cancelled_ids.add(pending.load_id)
            if self._active_kv_load_by_req.get(pending.req_id) == pending.load_id:
                self._active_kv_load_by_req.pop(pending.req_id, None)
            if seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS:
                self._detach_remote_kv_suffix(pending)
                if make_waiting:
                    seq.state.finish_remote_load()
            if (not pending.dispatching and not pending.dispatched
                    and not pending.needs_fence):
                self._pending_kv_loads.pop(pending.load_id, None)
                self.block_manager.release_pinned_logical_blocks(
                    pending.logical_block_ids)
        connector = self.kv_connector
        if connector is not None and cancelled_ids:
            connector.update_connector_output({
                'cancelled_load_ids': cancelled_ids,
            })

    def rollback_kv_connector_metadata(self, metadata: Any | None) -> None:
        """Undo pins and save bookkeeping after executor dispatch fails."""
        if metadata is None:
            return
        save_ids = {
            int(save_request.save_id)
            for save_request in getattr(metadata, 'save_requests', ())
        }
        self._release_kv_save_pins(save_ids)
        load_ids = {
            int(load_request.load_id)
            for load_request in getattr(metadata, 'load_requests', ())
        }
        self._rollback_kv_load_dispatches(load_ids)
        connector = self.kv_connector
        if connector is not None and (save_ids or load_ids):
            rolled_back = {}
            if save_ids:
                rolled_back['rolled_back_save_ids'] = save_ids
            if load_ids:
                rolled_back['rolled_back_load_ids'] = load_ids
            connector.update_connector_output(rolled_back)

    def _rollback_kv_load_dispatches(self, load_ids) -> None:
        """Release a failed RPC lease, retaining the same ready load ID."""
        for load_id in load_ids:
            pending = self._pending_kv_loads.get(int(load_id))
            if pending is None:
                continue
            pending.dispatching = False
            if pending.cancelled and not pending.dispatched:
                # The failed collective may have submitted on only a subset
                # of TP ranks. Keep the tombstone pin and re-dispatch this
                # idempotent load ID until one all-rank RPC succeeds.
                pending.needs_fence = True

    def _release_kv_save_pins(self, save_ids) -> None:
        pending_saves = getattr(self, '_pending_kv_saves', {})
        for save_id in save_ids:
            save_id = int(save_id)
            pending = pending_saves.pop(save_id, None)
            if pending is not None:
                self.block_manager.release_pinned_logical_blocks(pending.logical_block_ids)

    def _release_all_kv_save_pins(self) -> None:
        pending_saves = getattr(self, '_pending_kv_saves', {})
        self._release_kv_save_pins(tuple(pending_saves))

    def _release_all_kv_load_pins(self) -> None:
        pending_loads = getattr(self, '_pending_kv_loads', {})
        for load_id, pending in tuple(pending_loads.items()):
            pending_loads.pop(load_id, None)
            self.block_manager.release_pinned_logical_blocks(
                pending.logical_block_ids)
            if (len(pending.fallback_logical_block_ids) > 0
                    and not pending.fallback_restored):
                self.block_manager.release_pinned_logical_blocks(
                    pending.fallback_logical_block_ids)
        getattr(self, '_active_kv_load_by_req', {}).clear()
        getattr(self, '_remote_prefill_reservations', {}).clear()
        getattr(self, '_prefill_reservation_targets', {}).clear()

    def has_pending_kv_transfer_work(self) -> bool:
        """Return whether scheduler-owned save/load block pins remain."""
        return bool(
            getattr(self, '_pending_kv_saves', {})
            or getattr(self, '_pending_kv_loads', {})
        )

    def has_pending_kv_lookup_work(self) -> bool:
        """Return whether a scheduler-side non-blocking lookup is running."""
        connector = self.kv_connector
        return bool(
            connector is not None
            and connector.has_pending_kv_lookup_work()
        )

    def has_pending_kv_connector_work(self) -> bool:
        """Compatibility union used by existing engine-loop wakeup paths."""
        return (
            self.has_pending_kv_transfer_work()
            or self.has_pending_kv_lookup_work()
        )

    def request_kv_connector_finished(self, seq: SchedulerSequence) -> None:
        """Notify the connector before a sequence's request ref is freed."""
        connector = self.kv_connector
        try:
            if connector is not None:
                block_ids = tuple(
                    int(block_id)
                    for block_id in self.block_manager.get_block_table(seq)
                )
                connector.request_finished(seq, block_ids)
        finally:
            self._release_prefill_reservation(seq)
            self._cancel_remote_kv_load(seq, make_waiting=False)
            req_id = int(seq.seq_id)
            self._kv_seq_generations.pop(req_id, None)
            self._kv_preemption_ticks.pop(req_id, None)

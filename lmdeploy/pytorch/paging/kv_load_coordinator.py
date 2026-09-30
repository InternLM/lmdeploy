# Copyright (c) OpenMMLab. All rights reserved.
"""Paging ownership for asynchronous external KV-cache loads.

The connector owns lookup keys, remote I/O, and worker progress. This
coordinator owns destination admission and bridges connector progress to
sequence and GPU-block lifetimes without performing worker I/O itself:

1. :meth:`try_load` polls lookup, admits the complete prefill, allocates exact
   destinations, binds connector metadata, and calls :meth:`start_load`.
2. While workers may write those blocks, the sequence stays in
   ``WAITING_FOR_REMOTE_KVS`` and cannot be evicted or removed.
3. :meth:`apply_load_results` publishes a successful load. Failed/cancelled
   FA loads retain their local full blocks; hybrid loads release the request
   to rematch because the runtime state may have been overwritten.
4. The completed request is admitted for its remaining prefill, after which
   its load record and soft reservation can be released.

The coordinator also keeps a *soft* reservation for incomplete prefills.  It
does not allocate or pin the missing blocks; it only prevents another external
load from being admitted when their combined prefill tails cannot fit.  Normal
paging work may still borrow those blocks so the reservation does not reduce
GPU utilization unnecessarily.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lmdeploy.pytorch.kv_connector import KVLoadResult
from lmdeploy.pytorch.messages import SchedulerSequence, SchedulerSession

from .block_trie import PrefixCacheStats

if TYPE_CHECKING:
    from lmdeploy.pytorch.kv_connector import KVConnectorBase

    from .block_manager.base_block_manager import BaseBlockManager
    from .block_trie import BlockTrie
    from .eviction_helper.recompute_eviction_helper import RecomputeEvictionHelper
    from .state_manager import StateManager


class KVLoadAdmission(enum.Enum):
    """Narrow external-load result interpreted by prefill queue policy."""

    NO_LOAD = enum.auto()
    PENDING = enum.auto()
    FULL_PREFILL_UNAVAILABLE = enum.auto()
    SOFT_BUDGET_UNAVAILABLE = enum.auto()
    STARTED = enum.auto()


class _LoadPhase(enum.Enum):
    """Scheduler-side phase, distinct from ``MessageStatus``.

    ``READY`` means remote KV is valid but the remaining prefill has not yet
    passed scheduler admission.  The sequence itself is therefore back in
    ``MessageStatus.WAITING`` at that point.
    """

    # Workers may still write the allocated destination blocks. Paging must
    # retain the sequence and its load record in this phase.
    LOADING = enum.auto()
    # The load was published and the request is waiting to be scheduled.
    READY = enum.auto()
    # The remote-ready request was admitted for its remaining local prefill.
    PREFILLING = enum.auto()


class _DeferredLoadCleanup(enum.Enum):
    """Cleanup to apply after an active device write becomes safe."""

    NONE = enum.auto()
    STOP = enum.auto()
    END = enum.auto()


@dataclass(frozen=True, slots=True)
class _LoadPlan:
    """Block-aligned remote interval and full-prefill capacity target."""

    load_start_step: int
    remote_step: int
    target_blocks: int
    original_kv_token_limit: int | None


@dataclass
class _LoadRecord:
    """Paging state retained for one asynchronous load.

    ``load_start_step`` is the first FA block written by the load. Pure FA can
    roll back there; hybrid needs a new local match after releasing its state.
    ``remote_step`` is published only after every TP rank reports success.
    ``deferred_cleanup`` records the strongest user-requested action until
    device writes are safe; ending a request takes precedence over stopping it.
    """

    seq: SchedulerSequence
    # Exact query origin before local matching, not the rounded FA load start.
    match_start_step: int
    load_start_step: int
    remote_step: int
    phase: _LoadPhase = _LoadPhase.LOADING
    deferred_cleanup: _DeferredLoadCleanup = _DeferredLoadCleanup.NONE


class KVLoadCoordinator:
    """Keep transfer lifecycle and load admission out of connectors.

    Connectors own keys, remote I/O and worker progress. This coordinator owns
    only paging resources: allocated destinations, sequence transitions, and a
    soft budget that prevents over-admitting external loads. Normal paging may
    borrow that budget.

    Invariants:

    * A ``LOADING`` record is retained as long as a worker may write its blocks.
    * ``_prefill_targets`` is admission accounting, not physical allocation.
    * A load is removed only after terminal worker progress, worker drain, or
      completion/preemption of the remaining prefill.
    """

    def __init__(
        self,
        *,
        lookup_enabled: bool,
        connector: KVConnectorBase | None,
        block_manager: BaseBlockManager,
        block_trie: BlockTrie,
        sessions: dict[int, SchedulerSession],
        state_manager: StateManager | None = None,
    ) -> None:
        self.lookup_enabled = lookup_enabled
        self.connector = connector
        self.block_manager = block_manager
        self.block_trie = block_trie
        self.sessions = sessions
        self.state_manager = state_manager
        self.prefix_cache_stats = PrefixCacheStats()
        # Active load lifecycle. Records survive the LOADING -> READY ->
        # PREFILLING transitions so stop/end and preemption can find the owner.
        self._loads: dict[int, _LoadRecord] = {}
        # Full-prefill target for both remote loads and already-admitted local
        # prefills. The difference from seq.num_blocks is only softly reserved.
        self._prefill_targets: dict[int, tuple[SchedulerSequence, int]] = {}

    def prefill_target_blocks(
        self,
        seq: SchedulerSequence,
        prealloc_size: int = 0,
    ) -> int:
        """Return blocks needed for the complete request plus preallocation.

        This deliberately ignores a temporary ``kv_token_limit``: chunk or
        remote-hit allocation may stop earlier, but load admission must know
        whether the eventual full prefill has a path to completion.
        """
        block_size = seq.block_size
        target_tokens = int(seq.num_all_ids) + max(0, int(prealloc_size))
        return (target_tokens + block_size - 1) // block_size

    def track_prefill(
        self,
        seq: SchedulerSequence,
        *,
        prealloc_size: int = 0,
        target_blocks: int | None = None,
    ) -> None:
        """Track an incomplete prefill for later external-load admission.

        Local prefill work is tracked too: pipelined scheduling can consider a
        new load before an admitted prefill has produced all of its KV. Keeping
        both paths in one table prevents the new load from consuming capacity
        required by work that the scheduler has already accepted.
        """
        if not self.lookup_enabled:
            return
        if target_blocks is None:
            target_blocks = self.prefill_target_blocks(seq, prealloc_size)
        self._prefill_targets[int(seq.seq_id)] = (seq, int(target_blocks))

    def soft_reserved_blocks(self, exclude_seq: SchedulerSequence | None = None) -> int:
        """Return unallocated prefill blocks visible only to load admission.

        ``exclude_seq`` avoids counting the candidate twice: its missing blocks
        are added explicitly by :meth:`can_admit_load`.
        """
        exclude_id = None if exclude_seq is None else int(exclude_seq.seq_id)
        return sum(
            max(0, target_blocks - int(seq.num_blocks))
            for request_id, (seq, target_blocks) in self._prefill_targets.items()
            if request_id != exclude_id
        )

    def can_admit_load(
        self,
        seq: SchedulerSequence,
        target_blocks: int,
    ) -> bool:
        """Whether this load fits alongside tracked incomplete prefills.

        The candidate is admitted only when free blocks can cover both its full tail and every other soft reservation.
        This prevents several loads from each pinning a remote prefix and then deadlocking on their local tails.
        """
        missing_blocks = max(0, int(target_blocks) - int(seq.num_blocks))
        soft_reserved = self.soft_reserved_blocks(exclude_seq=seq)
        free_blocks = self.block_manager.get_num_free_gpu_blocks()
        return missing_blocks + soft_reserved <= free_blocks

    def is_lookup_pending(self, seq: SchedulerSequence) -> bool:
        """Whether this request already has an asynchronous lookup in
        flight."""
        connector = self.connector
        if not self.lookup_enabled or connector is None:
            return False
        return connector.is_lookup_pending(seq.seq_id)

    def record_cache_query(self, seq: SchedulerSequence, local_step: int, external_tokens: int = 0) -> None:
        """Record an admitted external query outside recomputation."""
        if self.connector is None or seq.prefix_cache.suppress_match_stats:
            return
        self.prefix_cache_stats.num_query_tokens += seq.num_all_ids - local_step
        self.prefix_cache_stats.num_hit_tokens += external_tokens

    def try_load(
        self,
        seq: SchedulerSequence,
        *,
        prealloc_size: int,
        evictable_seqs: Iterable[SchedulerSequence],
        eviction_helper: RecomputeEvictionHelper,
    ) -> KVLoadAdmission:
        """Poll and admit one external prefix without choosing queue policy.

        The return value describes only the connector/paging result. Prefill admission remains responsible for mapping
        it to skip, stop, continue, or load-started and for committing or rolling back its local match.
        """
        connector = self.connector
        if not self.lookup_enabled or connector is None:
            return KVLoadAdmission.NO_LOAD

        num_external_tokens, _ = connector.get_num_new_matched_tokens(
            seq,
            seq.num_history_ids,
        )
        if num_external_tokens is None:
            return KVLoadAdmission.PENDING
        if num_external_tokens <= 0:
            return KVLoadAdmission.NO_LOAD
        return self._admit_load(
            seq,
            num_external_tokens=int(num_external_tokens),
            prealloc_size=prealloc_size,
            evictable_seqs=evictable_seqs,
            eviction_helper=eviction_helper,
        )

    def _admit_load(
        self,
        seq: SchedulerSequence,
        *,
        num_external_tokens: int,
        prealloc_size: int,
        evictable_seqs: Iterable[SchedulerSequence],
        eviction_helper: RecomputeEvictionHelper,
    ) -> KVLoadAdmission:
        """Admit the complete prefill, then allocate the remote interval."""
        plan = self._plan_load(seq, num_external_tokens, prealloc_size)
        if plan is None:
            return KVLoadAdmission.NO_LOAD

        failure = self._admit_load_capacity(
            seq,
            plan,
            prealloc_size,
            evictable_seqs,
            eviction_helper,
        )
        if failure is not None:
            return failure

        self._allocate_and_start_load(seq, plan)
        return KVLoadAdmission.STARTED

    def _plan_load(
        self,
        seq: SchedulerSequence,
        num_external_tokens: int,
        prealloc_size: int,
    ) -> _LoadPlan | None:
        """Plan the safe block-aligned interval before mutating ownership."""
        block_size = seq.block_size
        local_step = int(seq.num_history_ids)
        # Transfers are block-granular. Reuse a private partial boundary block,
        # publish only full loaded blocks, and leave the final token to compute.
        load_start_step = local_step // block_size * block_size
        remote_step = local_step + num_external_tokens
        max_step = int(seq.get_prefix_cache_max_match_step())
        if self.state_manager is not None:
            # GDN can only resume from the exact checkpoint accepted by lookup.
            # Clipping it would pair FA with a state from a different boundary.
            if remote_step > max_step or seq.clamp_prefix_cache_match_step(remote_step) != remote_step:
                return None
        else:
            remote_step = min(remote_step, max_step)
            remote_step = remote_step // block_size * block_size
        if remote_step <= load_start_step:
            return None

        return _LoadPlan(
            load_start_step=load_start_step,
            remote_step=remote_step,
            target_blocks=self.prefill_target_blocks(seq, prealloc_size),
            original_kv_token_limit=seq.kv_token_limit,
        )

    def _admit_load_capacity(
        self,
        seq: SchedulerSequence,
        plan: _LoadPlan,
        prealloc_size: int,
        evictable_seqs: Iterable[SchedulerSequence],
        eviction_helper: RecomputeEvictionHelper,
    ) -> KVLoadAdmission | None:
        """Admit the full prefill against physical and soft capacity."""
        # Only the remote hit is allocated now, but admission guarantees the
        # complete prefill can finish beside every existing soft reservation.
        seq.kv_token_limit = None
        full_prefill_fits = eviction_helper.try_make_capacity_for(
            seq,
            list(evictable_seqs),
            prealloc_size,
        )
        if not full_prefill_fits:
            seq.kv_token_limit = plan.original_kv_token_limit
            return KVLoadAdmission.FULL_PREFILL_UNAVAILABLE
        if not self.can_admit_load(seq, plan.target_blocks):
            seq.kv_token_limit = plan.original_kv_token_limit
            return KVLoadAdmission.SOFT_BUDGET_UNAVAILABLE
        return None

    def _allocate_and_start_load(
        self,
        seq: SchedulerSequence,
        plan: _LoadPlan,
    ) -> None:
        """Allocate destinations, bind them, then transfer paging ownership."""
        connector = self.connector
        assert connector is not None
        original_num_blocks = seq.num_blocks
        original_state_slot = seq.logical_state
        try:
            # Allocate only the checked remote interval. The unallocated local
            # tail remains represented by the plan's soft target.
            seq.kv_token_limit = plan.remote_step
            self.block_manager.allocate(seq)
            if self.state_manager is not None:
                self.state_manager.allocate(seq)
            block_table = self.block_manager.get_block_table(seq)
            load_start_block = plan.load_start_step // seq.block_size
            remote_block = plan.remote_step // seq.block_size
            load_block_ids = tuple(
                int(block_id)
                for block_id in block_table[load_start_block:remote_block]
            )
            connector.update_state_after_alloc(
                seq,
                load_block_ids,
                plan.remote_step - plan.load_start_step,
            )
            # From start_load onward, cleanup must retain destinations until
            # workers report terminal progress or their queues are drained.
            self.start_load(
                seq,
                load_start_step=plan.load_start_step,
                remote_step=plan.remote_step,
                target_blocks=plan.target_blocks,
            )
        except Exception:
            if seq.num_blocks > original_num_blocks:
                self.block_manager.truncate(seq, original_num_blocks)
            if self.state_manager is not None and original_state_slot < 0:
                self.state_manager.free(seq)
            seq.kv_token_limit = plan.original_kv_token_limit
            raise
        seq.kv_token_limit = None

    def start_load(
        self,
        seq: SchedulerSequence,
        *,
        load_start_step: int,
        remote_step: int,
        target_blocks: int,
    ) -> None:
        """Take paging ownership after destinations have been allocated.

        :meth:`try_load` first allocates the exact destination range and binds
        connector metadata. Only then does this method move the sequence out of
        normal waiting and expose the asynchronous write to cleanup paths.
        """
        request_id = int(seq.seq_id)
        if request_id in self._loads:
            raise RuntimeError(f'request {request_id} already has an external KV load')
        if self.state_manager is not None:
            # A tentative local match may have selected an older checkpoint.
            # Store will fill the runtime slot directly, so no restore copy
            # may later overwrite it. We do not retain a rollback snapshot.
            self.block_trie.state_checkpoints.unpin_restore(seq)
            seq.prefix_cache.restore.clear()
        seq.state.begin_remote_load()
        self.track_prefill(seq, target_blocks=target_blocks)
        match_start_step = seq.prefix_cache.match_start_step
        if match_start_step < 0:
            match_start_step = int(seq.num_history_ids)
        self._loads[request_id] = _LoadRecord(
            seq=seq,
            match_start_step=match_start_step,
            load_start_step=load_start_step,
            remote_step=remote_step,
        )
        # Match vLLM's admission-time counters: the reloaded partial FA block
        # belongs to the external prefix, and transfer outcomes do not undo a hit.
        self.record_cache_query(seq, load_start_step, remote_step - load_start_step)

    def is_remote_ready(self, seq: SchedulerSequence) -> bool:
        """Return whether loaded KV is published but prefill is not
        admitted."""
        record = self._loads.get(int(seq.seq_id))
        return record is not None and record.phase is _LoadPhase.READY

    def mark_prefill_scheduled(self, seq: SchedulerSequence) -> None:
        """Record that the remote-ready request entered remaining prefill."""
        record = self._loads.get(int(seq.seq_id))
        if record is not None and record.phase is _LoadPhase.READY:
            record.phase = _LoadPhase.PREFILLING

    def apply_load_results(self, results: tuple[KVLoadResult, ...]) -> None:
        """Apply terminal load results aggregated across all TP ranks.

        Missing or non-``LOADING`` records are stale/duplicate progress and are
        ignored. A stop/end request wins over a successful worker result because
        the user no longer wants the loaded prefix to become schedulable.
        """
        for result in results:
            record = self._loads.get(int(result.request_id))
            if record is None or record.phase is not _LoadPhase.LOADING:
                continue
            if (record.deferred_cleanup is not _DeferredLoadCleanup.NONE
                    or not result.success):
                self._rollback(record)
                self._finish_cancelled_or_failed(record)
            else:
                self._publish(record)

    def _publish(self, record: _LoadRecord) -> None:
        """Publish fully written blocks and return the sequence to waiting.

        The blocks were private destinations while loading. Publishing inserts their full prefix into the local trie,
        advances sequence history, and exposes cached-token metrics only after all ranks have valid contents.
        """
        seq = record.seq
        if self.state_manager is not None:
            # The old partial local checkpoint no longer needs private blocks:
            # this load has filled the complete prefix through remote_step.
            seq.prefix_cache.recompute_overlap.clear_tracking()
        # Limit trie publication to the successfully loaded prefix. The request
        # may already own preallocated blocks after remote_step.
        seq.kv_token_limit = record.remote_step
        if self.block_trie.enabled:
            self.block_trie.allocate(seq)
            # The connector excluded the overlap from its reusable boundary.
            # Keep that suffix private when allocating the remaining prefill:
            # another request may have published matching target blocks while
            # this load was pending, with different MTP boundary contents.
            overlap = seq.prefix_cache.recompute_overlap
            start_block = record.remote_step // seq.block_size
            overlap.set_fresh_block_range(start_block, start_block + overlap.recompute_blocks)
        seq.set_step(record.remote_step)
        seq.kv_token_limit = None
        if seq.prefix_cache.match_start_step < 0:
            seq.prefix_cache.match_start_step = record.match_start_step
        self.block_trie.finalize_match(seq)
        seq.state.finish_remote_load()
        record.phase = _LoadPhase.READY

    def _rollback(self, record: _LoadRecord) -> None:
        """Discard destinations that a failed/cancelled load may have touched.

        Hybrid state cannot be truncated to a token boundary. Release it and
        let normal admission rematch a complete local checkpoint or recompute.
        Pure FA retains the untouched full blocks before ``load_start_step``.
        """
        seq = record.seq
        if self.state_manager is not None:
            seq.state.release_paging_resources()
            return
        load_start_blocks = record.load_start_step // seq.block_size
        if seq.num_blocks > load_start_blocks:
            self.block_manager.truncate(seq, load_start_blocks)
        seq.set_step(record.load_start_step)
        seq.kv_token_limit = None

        cursor = seq.prefix_cache.trie_cursor
        while cursor is not None and cursor.prefix_len > record.load_start_step:
            cursor = cursor.parent
        seq.prefix_cache.trie_cursor = cursor
        if seq.prefix_cache.match_start_step > record.load_start_step:
            seq.prefix_cache.match_start_step = -1
        if not seq.prefix_cache.suppress_match_stats:
            self.block_trie.finalize_match(seq)

    def _finish_cancelled_or_failed(self, record: _LoadRecord) -> None:
        """Release accounting and honor cleanup deferred during ``LOADING``."""
        seq = record.seq
        request_id = int(seq.seq_id)
        self._loads.pop(request_id, None)
        self._prefill_targets.pop(request_id, None)
        seq.state.finish_remote_load()
        if record.deferred_cleanup is _DeferredLoadCleanup.END:
            self._finish_deferred_end(seq)
        elif record.deferred_cleanup is _DeferredLoadCleanup.STOP:
            seq.state.stop()

    def defer_stop_if_loading(self, seq: SchedulerSequence) -> bool:
        """Return True when stop must wait for an active device write.

        Dropping the record or freeing sequence blocks now could let paging
        reuse memory that workers still address. Mark the intent and let
        :meth:`update` stop the sequence after terminal progress.
        """
        record = self._loads.get(int(seq.seq_id))
        if record is None or record.phase is not _LoadPhase.LOADING:
            self.release_tracking(seq)
            return False
        if record.deferred_cleanup is _DeferredLoadCleanup.NONE:
            record.deferred_cleanup = _DeferredLoadCleanup.STOP
        return True

    def defer_end_if_loading(self, seq: SchedulerSequence) -> bool:
        """Return True when removal must wait for an active device write.

        End differs from stop only in final cleanup: after the write terminates,
        the sequence is removed from its session and connector bookkeeping.
        """
        record = self._loads.get(int(seq.seq_id))
        if record is None or record.phase is not _LoadPhase.LOADING:
            self.release_tracking(seq)
            return False
        record.deferred_cleanup = _DeferredLoadCleanup.END
        return True

    def release_tracking(self, seq: SchedulerSequence) -> None:
        """Drop tracking after prefill completion, preemption, or removal.

        ``LOADING`` is intentionally a no-op because only terminal worker
        progress or worker drain can prove that destination writes have ended.
        """
        request_id = int(seq.seq_id)
        record = self._loads.get(request_id)
        if record is not None and record.phase is _LoadPhase.LOADING:
            return
        self._prefill_targets.pop(request_id, None)
        if record is not None:
            self._loads.pop(request_id, None)

    def release_completed_prefill_reservations(self, seqs: list[SchedulerSequence]) -> None:
        """Release reservations after model output advances sequence history.

        Dispatch alone is insufficient proof: only after EngineLoop applies the
        forward output can ``num_history_ids`` show that the accepted prefill
        reached ``input_end_pos``.
        """
        for seq in seqs:
            request_id = int(seq.seq_id)
            if request_id not in self._prefill_targets:
                continue
            if self.is_remote_ready(seq):
                continue
            if int(seq.num_history_ids) >= int(seq.input_end_pos):
                self.release_tracking(seq)

    def finish_deferred_loads_after_worker_drain(self) -> None:
        """Remove ended requests after workers can no longer write KV blocks.

        Engine sleep intentionally discards prefetched connector outputs. The worker-side shutdown drains accepted loads
        before releasing the cache, so deferred session removal becomes safe once that shutdown returns.
        """
        records = [
            record
            for record in self._loads.values()
            if (record.phase is _LoadPhase.LOADING
                and record.deferred_cleanup is _DeferredLoadCleanup.END)
        ]
        for record in records:
            self._rollback(record)
            self._finish_cancelled_or_failed(record)

    def _finish_deferred_end(self, seq: SchedulerSequence) -> None:
        session = seq.session
        session.end_sequence(seq)
        if not session.sequences:
            self.sessions.pop(session.session_id, None)

    def shutdown(self) -> None:
        """Stop new lookup admission and discard scheduler-side ownership."""
        self.lookup_enabled = False
        self.clear()
        self.connector = None

    def clear(self) -> None:
        self._loads.clear()
        self._prefill_targets.clear()

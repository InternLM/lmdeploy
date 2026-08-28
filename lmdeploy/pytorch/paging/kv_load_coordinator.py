# Copyright (c) OpenMMLab. All rights reserved.
"""Paging ownership for asynchronous external KV-cache loads.

The connector owns lookup keys, remote I/O, and worker progress, while the
scheduler owns sequences and GPU blocks.  This coordinator bridges those two
lifetimes without performing I/O itself:

1. The scheduler allocates destination blocks and calls :meth:`start_load`.
2. While workers may write those blocks, the sequence stays in
   ``WAITING_FOR_REMOTE_KVS`` and cannot be evicted or removed.
3. :meth:`update` publishes a successful load or rolls a failed/cancelled load
   back to the last block-aligned safe step.
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
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lmdeploy.pytorch.kv_connector import KVLoadResult
from lmdeploy.pytorch.messages import SchedulerSequence

if TYPE_CHECKING:
    from .scheduler import Scheduler


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


@dataclass
class _LoadRecord:
    """Paging state retained for one asynchronous load.

    ``fallback_step`` is the block-aligned prefix that remains trustworthy if
    a worker fails or is cancelled after partially writing a destination.
    ``remote_step`` is published only after every TP rank reports success.
    Stop/end flags defer user-requested cleanup until device writes are safe.
    """

    seq: SchedulerSequence
    fallback_step: int
    remote_step: int
    phase: _LoadPhase = _LoadPhase.LOADING
    stop_requested: bool = False
    end_requested: bool = False


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

    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler
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
        block_size = self.scheduler.cache_config.block_size
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
        if not self.scheduler._external_lookup_enabled:
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
        free_blocks = self.scheduler.block_manager.get_num_free_gpu_blocks()
        return missing_blocks + soft_reserved <= free_blocks

    def start_load(
        self,
        seq: SchedulerSequence,
        *,
        fallback_step: int,
        remote_step: int,
        target_blocks: int,
    ) -> None:
        """Take paging ownership after destinations have been allocated.

        ``Scheduler._start_external_load`` must first allocate the exact
        destination range and bind it to connector metadata. Only then does
        this method move the sequence out of the normal waiting queue and make
        the asynchronous write visible to paging cleanup paths.
        """
        request_id = int(seq.seq_id)
        if request_id in self._loads:
            raise RuntimeError(f'request {request_id} already has an external KV load')
        seq.state.begin_remote_load()
        self.track_prefill(seq, target_blocks=target_blocks)
        self._loads[request_id] = _LoadRecord(
            seq=seq,
            fallback_step=fallback_step,
            remote_step=remote_step,
        )

    def is_remote_ready(self, seq: SchedulerSequence) -> bool:
        """Return whether loaded KV is published but prefill is not
        admitted."""
        record = self._loads.get(int(seq.seq_id))
        return record is not None and record.phase is _LoadPhase.READY

    def mark_scheduled(self, seq: SchedulerSequence) -> None:
        """Record that the remote-ready request entered remaining prefill."""
        record = self._loads.get(int(seq.seq_id))
        if record is not None and record.phase is _LoadPhase.READY:
            record.phase = _LoadPhase.PREFILLING

    def update(self, results: tuple[KVLoadResult, ...]) -> None:
        """Apply terminal load results aggregated across all TP ranks.

        Missing or non-``LOADING`` records are stale/duplicate progress and are
        ignored. A stop/end request wins over a successful worker result because
        the user no longer wants the loaded prefix to become schedulable.
        """
        for result in results:
            record = self._loads.get(int(result.request_id))
            if record is None or record.phase is not _LoadPhase.LOADING:
                continue
            if record.stop_requested or record.end_requested or not result.success:
                self._rollback(record)
                self._finish_cancelled_or_failed(record)
            else:
                self._publish(record)

    def _publish(self, record: _LoadRecord) -> None:
        """Publish fully written blocks and return the sequence to waiting.

        The blocks were private destinations while loading. Publishing inserts their full prefix into the local trie,
        advances sequence history, and exposes cached-token metrics only after all ranks have valid contents.
        """
        scheduler = self.scheduler
        seq = record.seq
        # Limit trie publication to the successfully loaded prefix. The request
        # may already own preallocated blocks after remote_step.
        seq.kv_token_limit = record.remote_step
        if scheduler.block_trie.enabled:
            scheduler.block_trie.allocate(seq)
        seq.set_step(record.remote_step)
        seq.kv_token_limit = None
        if seq.prefix_cache.match_start_step < 0:
            # With no preceding local trie hit, the block-aligned load start is
            # the beginning of this request's externally cached interval.
            seq.prefix_cache.match_start_step = record.fallback_step
        scheduler._finish_prefix_cache_schedule(seq)
        seq.state.finish_remote_load()
        record.phase = _LoadPhase.READY

    def _rollback(self, record: _LoadRecord) -> None:
        """Discard destinations that a failed/cancelled load may have touched.

        A rank may fail after another rank has already written some blocks, so
        the original partial boundary block cannot be trusted. Roll back to the
        block-aligned ``fallback_step`` and move the trie cursor to an ancestor
        that refers only to retained blocks.
        """
        scheduler = self.scheduler
        seq = record.seq
        fallback_blocks = record.fallback_step // seq.block_size
        if seq.num_blocks > fallback_blocks:
            scheduler.block_manager.truncate(seq, fallback_blocks)
        seq.set_step(record.fallback_step)
        seq.kv_token_limit = None

        cursor = seq.prefix_cache.trie_cursor
        while cursor is not None and cursor.prefix_len > record.fallback_step:
            cursor = cursor.parent
        seq.prefix_cache.trie_cursor = cursor
        if seq.prefix_cache.match_start_step > record.fallback_step:
            seq.prefix_cache.match_start_step = -1
        scheduler._finish_prefix_cache_schedule(seq)

    def _finish_cancelled_or_failed(self, record: _LoadRecord) -> None:
        """Release accounting and honor cleanup deferred during ``LOADING``."""
        seq = record.seq
        request_id = int(seq.seq_id)
        self._loads.pop(request_id, None)
        self._prefill_targets.pop(request_id, None)
        seq.state.finish_remote_load()
        if record.end_requested:
            self._remove_sequence(seq)
        elif record.stop_requested:
            seq.state.stop()

    def request_stop(self, seq: SchedulerSequence) -> bool:
        """Return True when stop must wait for an active device write.

        Dropping the record or freeing sequence blocks now could let paging
        reuse memory that workers still address. Mark the intent and let
        :meth:`update` stop the sequence after terminal progress.
        """
        record = self._loads.get(int(seq.seq_id))
        if record is None or record.phase is not _LoadPhase.LOADING:
            self.release(seq)
            return False
        record.stop_requested = True
        return True

    def request_end(self, seq: SchedulerSequence) -> bool:
        """Return True when removal must wait for an active device write.

        End differs from stop only in final cleanup: after the write terminates,
        the sequence is removed from its session and connector bookkeeping.
        """
        record = self._loads.get(int(seq.seq_id))
        if record is None or record.phase is not _LoadPhase.LOADING:
            self.release(seq)
            return False
        record.end_requested = True
        return True

    def release(self, seq: SchedulerSequence) -> None:
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

    def release_completed_prefills(self, seqs: list[SchedulerSequence]) -> None:
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
                self.release(seq)

    def finish_deferred_loads_after_worker_drain(self) -> None:
        """Remove ended requests after workers can no longer write KV blocks.

        Engine sleep intentionally discards prefetched connector outputs. The worker-side shutdown drains accepted loads
        before releasing the cache, so deferred session removal becomes safe once that shutdown returns.
        """
        records = [
            record
            for record in self._loads.values()
            if record.phase is _LoadPhase.LOADING and record.end_requested
        ]
        for record in records:
            self._rollback(record)
            self._finish_cancelled_or_failed(record)

    def _remove_sequence(self, seq: SchedulerSequence) -> None:
        scheduler = self.scheduler
        connector = scheduler.kv_connector
        if connector is not None:
            connector.request_finished(seq)

        session = seq.session
        session.remove_sequence(seq)
        if not session.sequences:
            scheduler.sessions.pop(session.session_id, None)

    def clear(self) -> None:
        self._loads.clear()
        self._prefill_targets.clear()

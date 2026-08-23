# Copyright (c) OpenMMLab. All rights reserved.
"""Paging ownership for asynchronous external KV-cache loads."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lmdeploy.pytorch.kv_connector import KVLoadResult
from lmdeploy.pytorch.messages import SchedulerSequence

if TYPE_CHECKING:
    from .scheduler import Scheduler


class _LoadPhase(enum.Enum):
    LOADING = enum.auto()
    READY = enum.auto()
    PREFILLING = enum.auto()


@dataclass
class _LoadRecord:
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
    """

    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler
        self._loads: dict[int, _LoadRecord] = {}
        self._prefill_targets: dict[int, tuple[SchedulerSequence, int]] = {}

    def prefill_target_blocks(
        self,
        seq: SchedulerSequence,
        prealloc_size: int = 0,
    ) -> int:
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
        """Track the complete prefill target for external-load admission."""
        if not self.scheduler._external_lookup_enabled:
            return
        if target_blocks is None:
            target_blocks = self.prefill_target_blocks(seq, prealloc_size)
        self._prefill_targets[int(seq.seq_id)] = (seq, int(target_blocks))

    def soft_reserved_blocks(self, exclude_seq: SchedulerSequence | None = None) -> int:
        """Return unallocated prefill blocks visible only to load admission."""
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
        """Whether this load fits alongside tracked incomplete prefills."""
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
        record = self._loads.get(int(seq.seq_id))
        return record is not None and record.phase is _LoadPhase.READY

    def mark_scheduled(self, seq: SchedulerSequence) -> None:
        record = self._loads.get(int(seq.seq_id))
        if record is not None and record.phase is _LoadPhase.READY:
            record.phase = _LoadPhase.PREFILLING

    def update(self, results: tuple[KVLoadResult, ...]) -> None:
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
        scheduler = self.scheduler
        seq = record.seq
        seq.kv_token_limit = record.remote_step
        if scheduler.block_trie.enabled:
            scheduler.block_trie.allocate(seq)
        seq.set_step(record.remote_step)
        seq.kv_token_limit = None
        if seq.prefix_cache.match_start_step < 0:
            seq.prefix_cache.match_start_step = record.fallback_step
        scheduler._finish_prefix_cache_schedule(seq)
        seq.state.finish_remote_load()
        record.phase = _LoadPhase.READY

    def _rollback(self, record: _LoadRecord) -> None:
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
        """Return True when stop must wait for an active device write."""
        record = self._loads.get(int(seq.seq_id))
        if record is None or record.phase is not _LoadPhase.LOADING:
            self.release(seq)
            return False
        record.stop_requested = True
        return True

    def request_end(self, seq: SchedulerSequence) -> bool:
        """Return True when sequence removal must wait for load completion."""
        record = self._loads.get(int(seq.seq_id))
        if record is None or record.phase is not _LoadPhase.LOADING:
            self.release(seq)
            return False
        record.end_requested = True
        return True

    def release(self, seq: SchedulerSequence) -> None:
        """Drop prefill tracking after completion, preemption or removal."""
        request_id = int(seq.seq_id)
        record = self._loads.get(request_id)
        if record is not None and record.phase is _LoadPhase.LOADING:
            return
        self._prefill_targets.pop(request_id, None)
        if record is not None:
            self._loads.pop(request_id, None)

    def release_completed_prefills(self, seqs: list[SchedulerSequence]) -> None:
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
            block_ids = tuple(
                int(block_id)
                for block_id in scheduler.block_manager.get_block_table(seq)
            )
            connector.request_finished(seq, block_ids)

        session = seq.session
        session.remove_sequence(seq)
        if not session.sequences:
            scheduler.sessions.pop(session.session_id, None)

    def clear(self) -> None:
        self._loads.clear()
        self._prefill_targets.clear()

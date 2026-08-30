# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
"""Public paging scheduler and sequence-lifecycle facade."""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING

from torch.profiler import record_function

from lmdeploy.messages import ScheduleMetrics

from ..config import CacheConfig, SchedulerConfig
from ..messages import MessageStatus, SchedulerSequence, SchedulerSession, SequenceManager, SequenceMeta
from .block_manager import build_block_manager
from .block_trie import BlockTrie
from .eviction_helper import build_eviction_helper
from .kv_load_coordinator import KVLoadCoordinator
from .kv_save_coordinator import KVSaveCoordinator
from .prefill_scheduler import _PrefillScheduler, _PrefillTurnPolicy
from .state_manager import build_state_manager

if TYPE_CHECKING:
    from lmdeploy.pytorch.kv_connector.base import KVConnectorBase

MapType = dict[int, int]
SeqList = list[SchedulerSequence]


@dataclass
class SchedulerOutput:
    """Paging selection and connector snapshots for one model step."""

    running: SeqList
    swap_in_map: MapType
    swap_out_map: MapType
    copy_map: MapType
    # Absolute post-forward token boundary for each request. The connector
    # rounds it down to full blocks and saves only the suffix not saved before.
    connector_token_lens: tuple[int, ...] = ()
    # Current physical GPU block table for each request. Workers use these IDs
    # to locate KV tensors, but paging may reuse them after ownership is lost.
    connector_block_ids: tuple[tuple[int, ...], ...] = ()
    # Logical paging IDs corresponding to connector_block_ids. Save leases pin
    # these stable ownership handles until all TP ranks finish asynchronous I/O,
    # preventing their physical blocks from being reassigned too early.
    connector_logical_block_ids: tuple[tuple[int, ...], ...] = ()


class Scheduler:
    """Coordinate sequence lifecycle and paging-resource admission.

    Args:
        scheduler_config: Batch and eviction policy.
        cache_config: KV and state-cache configuration.
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        seq_meta: SequenceMeta | None = None,
        kv_connector: 'KVConnectorBase | None' = None,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.sessions: dict[int, SchedulerSession] = OrderedDict()
        self.kv_connector = kv_connector

        self.state_manager = build_state_manager(self.cache_config)
        self.block_manager = build_block_manager(cache_config)
        self.is_ssm = len(self.cache_config.states_shapes) > 0
        transfer_config = cache_config.kv_transfer_config
        # A producer-only connector still needs the save path below, but must
        # not issue lookups. SSM restore owns a different state-cache protocol
        # and is deliberately excluded from external KV load admission.
        external_lookup_enabled = (
            kv_connector is not None
            and transfer_config is not None
            and transfer_config.is_kv_consumer
            and not self.is_ssm
        )
        checkpoint_state_manager = self.state_manager if self.is_ssm else None
        self.block_trie = BlockTrie(allocator=self.block_manager.allocator,
                                   block_size=self.cache_config.block_size,
                                   enabled=self.cache_config.enable_prefix_caching,
                                   checkpoint_state_manager=checkpoint_state_manager)

        self.eviction_helper = build_eviction_helper(self, self.scheduler_config.eviction_type)
        # Load admission receives only paging owners plus request-local queue
        # candidates from its caller; it does not reach back through Scheduler.
        self.kv_load_coordinator = KVLoadCoordinator(
            lookup_enabled=external_lookup_enabled,
            connector=kv_connector,
            block_manager=self.block_manager,
            block_trie=self.block_trie,
            eviction_helper=self.eviction_helper,
            sessions=self.sessions,
        )
        self._prefill_scheduler = _PrefillScheduler(
            scheduler_config=self.scheduler_config,
            cache_config=self.cache_config,
            is_ssm=self.is_ssm,
            block_manager=self.block_manager,
            block_trie=self.block_trie,
            state_manager=self.state_manager,
            eviction_helper=self.eviction_helper,
            load_coordinator=self.kv_load_coordinator,
        )
        # Keep save call sites uniform even when the producer role is disabled.
        self.kv_save_coordinator = KVSaveCoordinator(self)

        seq_meta = seq_meta or SequenceMeta(self.cache_config.block_size)
        self.seq_meta = seq_meta
        self.seq_manager = SequenceManager(seq_meta)
        self.scheduler_tick = 0

    def tick(self):
        """Mark one scheduler progress step (once per forward dispatch)."""
        self.scheduler_tick += 1

    def shutdown(self) -> None:
        """Release scheduler-side connector resources exactly once.

        Engine shutdown normally drains worker queues first. Clearing scheduler ownership here makes repeated shutdown
        harmless and prevents any later scheduling path from starting new external work.
        """
        connector = self.kv_connector
        self.kv_connector = None
        self.kv_load_coordinator.disable()
        self.kv_save_coordinator.clear()
        if connector is not None:
            connector.shutdown()

    @property
    def _external_lookup_enabled(self) -> bool:
        """Whether external KV lookup admission is currently enabled."""
        return self.kv_load_coordinator.lookup_enabled

    @property
    def last_schedule_had_pending_lookup(self) -> bool:
        """Whether the latest prefill turn encountered a pending lookup."""
        return self._prefill_scheduler.last_schedule_had_pending_lookup

    @last_schedule_had_pending_lookup.setter
    def last_schedule_had_pending_lookup(self, value: bool) -> None:
        self._prefill_scheduler.last_schedule_had_pending_lookup = value

    def has_waiting_long_prefill(self):
        """Whether a waiting request would need a non-final prefill chunk."""
        return self._prefill_scheduler.has_waiting_long_prefill(self.waiting)

    def reserve_long_context_chunk(self,
                                   seq: SchedulerSequence,
                                   chunk_size: int,
                                   prealloc_size: int = 0,
                                   is_last_chunk: bool = False):
        """Reserve KV blocks for the next chunk of a running long prefill."""
        return self._prefill_scheduler.reserve_long_context_chunk(
            seq,
            stopped=self.hanging,
            waiting=self.waiting,
            chunk_size=chunk_size,
            prealloc_size=prealloc_size,
            is_last_chunk=is_last_chunk,
        )

    # Remote-loading sequences are intentionally separate from WAITING: workers
    # may address their destination blocks, so ordinary scheduling/eviction
    # must not treat them as candidates until the coordinator publishes them.

    # Sequence views.
    @property
    def waiting(self) -> SeqList:
        return list(self.seq_manager.get_sequences(MessageStatus.WAITING).values())

    @property
    def remote_loading(self) -> SeqList:
        return list(self.seq_manager.get_sequences(MessageStatus.WAITING_FOR_REMOTE_KVS).values())

    @property
    def ready(self) -> SeqList:
        return list(self.seq_manager.get_sequences(MessageStatus.READY).values())

    @property
    def hanging(self) -> SeqList:
        return list(self.seq_manager.get_sequences(MessageStatus.STOPPED).values())

    @property
    def running(self) -> SeqList:
        return list(self.seq_manager.get_sequences(MessageStatus.RUNNING).values())

    @property
    def migration_waiting(self) -> SeqList:
        return list(self.seq_manager.get_sequences(MessageStatus.MIGRATION_WAITING).values())

    @property
    def migration_done(self) -> SeqList:
        return list(self.seq_manager.get_sequences(MessageStatus.MIGRATION_DONE).values())

    # Sequence counts.
    def num_waiting(self) -> int:
        return self.seq_manager.num_sequences(MessageStatus.WAITING)

    def num_remote_loading(self) -> int:
        return self.seq_manager.num_sequences(MessageStatus.WAITING_FOR_REMOTE_KVS)

    def num_ready(self) -> int:
        return self.seq_manager.num_sequences(MessageStatus.READY)

    def num_running(self) -> int:
        return self.seq_manager.num_sequences(MessageStatus.RUNNING)

    def num_migration_waiting(self) -> int:
        return self.seq_manager.num_sequences(MessageStatus.MIGRATION_WAITING)

    def num_migration_done(self) -> int:
        return self.seq_manager.num_sequences(MessageStatus.MIGRATION_DONE)

    # Non-empty status checks used by engine control flow.
    def has_waiting(self) -> bool:
        return self.seq_manager.num_sequences(MessageStatus.WAITING) > 0

    def has_remote_loading(self) -> bool:
        return self.seq_manager.num_sequences(MessageStatus.WAITING_FOR_REMOTE_KVS) > 0

    def has_ready(self) -> bool:
        return self.seq_manager.num_sequences(MessageStatus.READY) > 0

    def has_migration_waiting(self) -> bool:
        return self.seq_manager.num_sequences(MessageStatus.MIGRATION_WAITING) > 0

    def has_migration_done(self) -> bool:
        return self.seq_manager.num_sequences(MessageStatus.MIGRATION_DONE) > 0

    def add_session(self, session_id: int):
        """Add new session.

        Args:
            session_id (int): New session id.
        """
        assert session_id not in self.sessions
        session = SchedulerSession(session_id, seq_manager=self.seq_manager, scheduler=self)
        self.sessions[session_id] = session
        return session

    def _schedule_migration(self):
        migration_ready: SeqList = []
        migration_waiting = sorted(
            self.migration_waiting,
            key=lambda seq: seq.arrive_time,
        )

        max_batches = self.scheduler_config.max_batches - self.num_ready() - self.num_running()
        while migration_waiting and len(migration_ready) < max_batches:
            seq = migration_waiting.pop(0)
            self.block_trie.match(seq)
            evictable = list(
                chain(
                    reversed(self.hanging),
                    reversed(migration_waiting),
                ))
            if not self.eviction_helper.evict_for_seq(seq, evictable, 0):
                break

            # allocate session memory
            self.block_manager.allocate(seq)
            self.block_trie.finalize_match(seq)
            seq.state.activate()
            migration_ready.append(seq)

        return migration_ready

    def schedule(self,
                 is_prefill: bool,
                 prealloc_size: int = 0,
                 allow_long_prefill: bool = True,
                 prefer_long_prefill: bool = False):
        """Select the next prefill batch.

        Decode capacity is admitted by :meth:`schedule_running`.
        """
        if not is_prefill:
            raise ValueError(
                'schedule only selects prefill work; use schedule_running '
                'for decode capacity admission')

        turn_policy = _PrefillTurnPolicy.from_flags(
            allow_long_prefill,
            prefer_long_prefill,
        )
        running = self._prefill_scheduler.schedule(
            waiting=self.waiting,
            stopped=self.hanging,
            num_ready=self.num_ready(),
            num_running=self.num_running(),
            turn_policy=turn_policy,
            prealloc_size=prealloc_size,
        )
        return SchedulerOutput(
            running=running,
            swap_in_map={},
            swap_out_map={},
            copy_map={},
        )

    @record_function('schedule_running')
    def schedule_running(self, running: SeqList, num_required_tokens: int = 1, prealloc_size: int = 1):
        """Admit KV growth for running sequences and return their validity."""
        assert len(running) > 0
        eviction_helper = self.eviction_helper

        valid_mask = [True] * len(running)
        for idx in reversed(range(len(running))):
            seq = running[idx]
            if seq.status != MessageStatus.RUNNING:
                valid_mask[idx] = False
                continue
            num_required_blocks = self.block_manager.num_required_blocks(seq, num_required_tokens)
            if num_required_blocks == 0:
                continue

            if eviction_helper.evict_for_seq(seq, self.hanging + self.waiting, prealloc_size):
                self.block_manager.allocate(seq, prealloc_size)
                self.block_trie.allocate(seq)
                continue

            seq.state.deactivate()
            self.kv_load_coordinator.release(seq)
            seq.state.evict()
            valid_mask[idx] = False
        return valid_mask

    def stop_session(self, session_id: int):
        """Stop session.

        Args:
            session_id (int): The session id.
        """
        assert session_id in self.sessions
        session = self.sessions[session_id]
        connector = self.kv_connector
        for seq in session.sequences.values():
            # A lookup owns no GPU destinations and can be cancelled directly.
            if connector is not None:
                connector.cancel_lookup(seq.seq_id)
            # An active load may still write GPU memory. Defer the state change
            # until all ranks terminate instead of making its blocks evictable.
            if self.kv_load_coordinator.request_stop(seq):
                continue
            seq.state.stop()

    def end_session(self, session_id: int):
        """End session.

        Args:
            session_id (int): The session id.
        """
        if self.seq_meta.sampling_strategy is not None:
            self.seq_meta.sampling_strategy.on_session_end(session_id)
        session = self.sessions[session_id]
        seqs = list(session.sequences.values())
        connector = self.kv_connector
        for seq in seqs:
            if connector is not None:
                connector.cancel_lookup(seq.seq_id)
            # Session removal also frees sequence blocks, so it must be deferred
            # while a worker may still address an in-flight load destination.
            if self.kv_load_coordinator.request_end(seq):
                continue
            # stop session so it won't get scheduled again
            seq.state.stop()
            if connector is not None:
                connector.request_finished(seq)
            session.remove_sequence(seq)
        if not session.sequences:
            self.sessions.pop(session_id)

    def has_unfinished(self):
        """Whether model, migration, load, or save ownership is outstanding.

        Remote-loading requests are outside the normal waiting queue, and save leases may outlive their request. Both
        must keep the engine alive until workers stop accessing paging-owned memory.
        """
        return (
            self.has_ready()
            or self.has_waiting()
            or self.has_remote_loading()
            or self.has_migration_done()
            or self.kv_save_coordinator.has_pending()
        )

    def build_connector_meta(
        self,
        running: SeqList,
        swap_in_map: MapType | None = None,
        swap_out_map: MapType | None = None,
        connector_token_lens: tuple[int, ...] = (),
    ):
        """Build and lease one connector payload after work selection.

        This is called even when ``running`` is empty because connector-only
        executor steps submit pending transfers and poll completions. For a
        prefill save, block tables are snapshotted only after the model batch is
        fixed, then logical leases are acquired before metadata reaches workers.
        """
        connector = self.kv_connector
        if connector is None:
            return None
        if connector_token_lens:
            # Workers address the current physical cache slots, while save
            # leases pin logical block ownership across asynchronous I/O.
            logical_block_ids = tuple(
                tuple(int(block_id) for block_id in seq.logical_blocks.get_real_blocks())
                for seq in running
            )
            block_ids = tuple(
                tuple(int(block_id) for block_id in self.block_manager.get_block_table(seq))
                for seq in running
            )
        else:
            logical_block_ids = ()
            block_ids = ()
        scheduler_output = SchedulerOutput(
            running=running,
            swap_in_map=swap_in_map or {},
            swap_out_map=swap_out_map or {},
            copy_map={},
            connector_token_lens=connector_token_lens,
            connector_block_ids=block_ids,
            connector_logical_block_ids=logical_block_ids,
        )
        metadata = connector.build_connector_meta(scheduler_output)
        if metadata is not None:
            # Acquire before the caller queues metadata. Sequence cleanup may
            # otherwise release the last block reference before save starts.
            self.kv_save_coordinator.acquire(metadata)
        return metadata

    def update_connector_output(self, connector_output) -> None:
        """Convert all-TP worker progress into paging state transitions.

        The connector filters/aggregates rank-local output first. Paging then publishes or rolls back loads and releases
        only save operations known to be terminal across all ranks.
        """
        if connector_output is None or self.kv_connector is None:
            return
        result = self.kv_connector.update_connector_output(connector_output)
        self.kv_load_coordinator.update(result.load_results)
        self.kv_save_coordinator.update(result.completed_save_ids)

    def release_completed_prefill_reservations(self, seqs: SeqList) -> None:
        """Release soft targets only after forward output advanced history."""
        self.kv_load_coordinator.release_completed_prefills(seqs)

    def finish_deferred_kv_transfers_after_worker_drain(self) -> None:
        """Release paging ownership after worker transfer queues have drained.

        Engine sleep may discard prefetched completion outputs. Worker drain is
        the alternate terminal proof: deferred ended loads can be removed and
        save leases can be released even without their normal output events.
        """
        self.kv_load_coordinator.finish_deferred_loads_after_worker_drain()
        if self.kv_connector is not None:
            self.kv_connector.finish_transfers_after_worker_drain()
        self.kv_save_coordinator.clear()

    def get_block_tables(self, seqs: SeqList):
        """Get block tables for the sequences."""
        return [self.block_manager.get_block_table(seq) for seq in seqs]

    def resolve_gpu_block_offsets(self, logical_block_ids):
        """Resolve paging-owned logical ids for a forward cache-copy plan."""
        return self.block_manager.resolve_gpu_block_offsets(logical_block_ids)

    def evict_seqs(self, running: SeqList):
        """Evict running sequences."""
        for seq in running:
            self.kv_load_coordinator.release(seq)
            seq.state.evict()

    def activate_seqs(self, running: SeqList, filter_status: MessageStatus = MessageStatus.READY):
        """Lock running sequence."""
        for seq in running:
            if seq.status == filter_status:
                seq.state.activate()

    def deactivate_seqs(self, running: SeqList, filter_status: MessageStatus = MessageStatus.RUNNING):
        for seq in running:
            if seq.status == filter_status:
                seq.state.deactivate()

    @contextmanager
    def seqs_activation(self, running: SeqList):
        """Context manager to activate and deactivate sequences."""
        self.activate_seqs(running, MessageStatus.READY)
        try:
            yield running
        finally:
            self.deactivate_seqs(running, MessageStatus.RUNNING)

    def activate_migration_seqs(self, running: SeqList):
        """Lock running sequence."""
        return self.activate_seqs(running, filter_status=MessageStatus.MIGRATION_READY)

    def deactivate_migration_seqs(self, running: SeqList):
        """Unlock running migration."""
        return self.deactivate_seqs(running, filter_status=MessageStatus.MIGRATION_RUNNING)

    @contextmanager
    def seqs_migration_activation(self, running: SeqList):
        """Context manager to activate and deactivate sequences."""
        self.activate_migration_seqs(running)
        try:
            yield running
        finally:
            self.deactivate_migration_seqs(running)

    def collect_migration_done(self):
        for seq in self.migration_done:
            seq.state.activate()

    @property
    def schedule_metrics(self):
        total_blocks = self.block_manager.num_gpu_blocks
        free_blocks = self.block_manager.get_num_free_gpu_blocks()
        cache_usage = 1.0 - free_blocks / total_blocks if total_blocks else 0.0
        return ScheduleMetrics(
            active_seqs=self.num_running(),
            waiting_seqs=self.num_waiting() + self.num_ready() + self.num_remote_loading(),
            cache_usage=cache_usage,
            prefix_cache_hit_rate=self.block_trie.stats.hit_rate(),
            scheduler_tick=self.scheduler_tick,
        )

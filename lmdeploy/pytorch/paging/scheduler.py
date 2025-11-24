# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List

from lmdeploy.messages import EventType, ScheduleMetrics
from lmdeploy.utils import get_logger, logging_timer

from ..config import CacheConfig, SchedulerConfig
from ..messages import MessageStatus, SchedulerSequence, SchedulerSession, SequenceManager, SequenceMeta
from .block_manager import build_block_manager
from .block_trie import BlockTrie
from .state_manager import StateManager

logger = get_logger('lmdeploy')

SeqList = List[SchedulerSequence]


@dataclass
class SchedulerOutput:
    """Output of schedule."""

    running: SeqList
    swap_in_map: Dict[int, int]
    swap_out_map: Dict[int, int]
    copy_map: Dict[int, int]


class Scheduler:
    """Tools to schedule next step.

    Args:
        scheduler_config (SchedulerConfig): The config of scheduler.
        cache_config (CacheConfig): The config of cache info.
    """

    def __init__(self,
                 scheduler_config: SchedulerConfig,
                 cache_config: CacheConfig,
                 seq_meta: SequenceMeta = None) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config

        self.sessions: Dict[int, SchedulerSession] = OrderedDict()

        # For Disaggregation
        self.locked_sessions: Dict[int, SchedulerSession] = OrderedDict()

        self.block_manager = build_block_manager(cache_config)
        self.block_trie = BlockTrie(self.cache_config, self.block_manager)
        self.state_manager = StateManager(self.cache_config.num_state_caches)
        self.is_ssm = len(self.cache_config.states_shapes) > 0

        self.eviction_helper = self.build_eviction_helper(self.scheduler_config.eviction_type)

        seq_meta = seq_meta or SequenceMeta(self.cache_config.block_size)
        self.seq_manager = SequenceManager(seq_meta)

    @property
    def waiting(self):
        """Get waiting sequence."""
        seq_map = self.seq_manager.get_sequences(MessageStatus.WAITING)
        return list(seq_map.values())

    @property
    def running(self):
        """Get waiting sequence."""
        seq_map = self.seq_manager.get_sequences(MessageStatus.RUNNING)
        return list(seq_map.values())

    @property
    def hanging(self):
        """Get waiting sequence."""
        seq_map = self.seq_manager.get_sequences(MessageStatus.STOPPED)
        return list(seq_map.values())

    @property
    def locked(self):
        """Get waiting sequence."""
        seq_map = self.seq_manager.get_sequences(MessageStatus.LOCKED)
        return list(seq_map.values())

    @property
    def waiting_migration(self):
        """Get migration sequence."""
        seq_map = self.seq_manager.get_sequences(MessageStatus.WAITING_MIGRATION)
        return list(seq_map.values())

    @property
    def running_migration(self):
        """Get migration sequence."""
        seq_map = self.seq_manager.get_sequences(MessageStatus.RUNNING_MIGRATION)
        return list(seq_map.values())

    @property
    def migration_done(self):
        """Get waiting sequence."""
        seq_map = self.seq_manager.get_sequences(MessageStatus.MIGRATION_DONE)
        return list(seq_map.values())

    def build_eviction_helper(self, eviction_type: str):
        if eviction_type == 'copy':
            logger.warning('`copy` eviction has been deprecated, '
                           'use `recompute` instead.')
            eviction_type = 'recompute'
        if eviction_type == 'recompute':
            from .eviction_helper import RecomputeEvictionHelper
            return RecomputeEvictionHelper(self)
        else:
            raise TypeError(f'Unknown eviction type: {eviction_type}')

    def _set_message_status(self, message: SchedulerSequence, status: MessageStatus):
        """Set status of message.

        Args:
            message (SchedulerSequence): message to setup status.
            status (MessageStatus): New message status.
        """
        message.status = status

    def add_session(self, session_id: int):
        """Add new session.

        Args:
            session_id (int): New session id.
        """
        assert session_id not in self.sessions
        session = SchedulerSession(session_id, seq_manager=self.seq_manager)
        self.sessions[session_id] = session
        return session

    def add_sequence(self, seq: SchedulerSequence):
        """Add sequence.

        Args:
            seq (SchedulerSequence): New sequence.
        """
        assert (seq.session_id in self.sessions), f'Unknown session id {seq.session_id}'

        # push message to waiting queue
        self._set_message_status(seq, MessageStatus.WAITING)

        seq.record_event(EventType.QUEUED)

    @logging_timer('ScheduleMigration', logger)
    def _schedule_migration(self):
        running_migration: SeqList = []
        migrating_token_count = 0

        def _to_running(seq: SchedulerSequence):
            """To running."""
            seq.status = MessageStatus.RUNNING_MIGRATION
            running_migration.append(seq)
            nonlocal migrating_token_count
            migrating_token_count += seq.num_token_ids

        def __evict_for_seq(seq: SchedulerSequence, waiting):
            """Evict until can append."""
            from itertools import chain

            hanging = reversed(self.hanging)
            waiting = reversed(waiting)
            evictable = list(chain(hanging, waiting))
            return self.eviction_helper.evict_for_seq(seq, evictable, 0)

        def _reorder_migrating():
            """Reorder waiting."""
            return sorted(self.waiting_migration, key=lambda seq: seq.arrive_time)

        waiting_migration = _reorder_migrating()

        max_batches = self.scheduler_config.max_batches - self.num_running() - self.num_locked()
        while len(waiting_migration) > 0 and len(running_migration) < max_batches:
            seq = waiting_migration.pop(0)
            self.block_trie.match(waiting_migration)
            if not __evict_for_seq(seq, waiting_migration):
                break

            # allocate session memory
            self.block_manager.allocate(seq)
            _to_running(seq)

        return running_migration

    @logging_timer('SchedulePrefilling', logger)
    def _schedule_prefill(self, prealloc_size: int = 0):
        """Schedule for prefilling."""

        max_batches = self.scheduler_config.max_batches - self.num_running() - self.num_locked()
        eviction_helper = self.eviction_helper
        swap_out_map: Dict[int, int] = dict()
        swap_in_map: Dict[int, int] = dict()
        copy_map: Dict[int, int] = dict()
        running: SeqList = []
        token_count = 0

        def _to_running(seq: SchedulerSequence):
            """To running."""
            seq.status = MessageStatus.RUNNING
            running.append(seq)
            nonlocal token_count
            token_count += seq.num_token_ids

        def __evict_for_seq(seq: SchedulerSequence, waiting):
            """Evict until can append."""
            from itertools import chain
            hanging = reversed(self.hanging)
            waiting = reversed(waiting)
            evictable = list(chain(hanging, waiting))
            return eviction_helper.evict_for_seq(seq, evictable, prealloc_size)

        def _reorder_waiting():
            """Reorder waiting."""
            return sorted(self.waiting, key=lambda seq: seq.arrive_time)

        num_waiting = self.seq_manager.num_sequences(MessageStatus.WAITING)
        if (len(running) >= max_batches or num_waiting == 0):
            return running, swap_in_map, swap_out_map, copy_map

        waiting = _reorder_waiting()
        while len(waiting) > 0 and len(running) < max_batches:
            seq = waiting.pop(0)

            if (len(running) > 0 and token_count + seq.num_token_ids > self.cache_config.max_prefill_token_num):
                break

            self.block_trie.match(seq)

            if not __evict_for_seq(seq, waiting):
                break

            # allocate session memory
            self.block_manager.allocate(seq, prealloc_size)
            if self.is_ssm:
                self.state_manager.allocate(seq)
            _to_running(seq)

            seq.record_event(EventType.SCHEDULED)

        return running, swap_in_map, swap_out_map, copy_map

    @logging_timer('ScheduleDecoding', logger)
    def _schedule_decoding(self, prealloc_size: int = 0):
        """Schedule decoding."""

        running = self.running
        assert len(running) != 0

        eviction_helper = self.eviction_helper
        swap_out_map: Dict[int, int] = dict()
        swap_in_map: Dict[int, int] = dict()
        copy_map: Dict[int, int] = dict()

        def __evict_for_seq(seq: SchedulerSequence, num_required_blocks: int):
            """Evict until can append."""
            if num_required_blocks == 0:
                # No need to evict, just return True.
                return True
            elif num_required_blocks < self.block_manager.get_num_free_gpu_blocks():
                # Enough free blocks, just return True.
                return True

            from itertools import chain
            hanging = reversed(self.hanging)
            waiting = reversed(self.waiting)
            evictable = list(chain(hanging, waiting))
            return eviction_helper.evict_for_seq(seq, evictable, prealloc_size)

        # 1. running
        for seq in running:
            # token + n

            num_required_blocks = self.block_manager.num_required_blocks(seq, prealloc_size)
            if len(seq.logical_blocks) + num_required_blocks > self.block_manager.num_gpu_blocks:
                # Reach max gpu cache size.
                logger.warning(f'session[{seq.session_id}] '
                               f'sequence[{seq.seq_id}] '
                               'reach max gpu size.')
                self._set_message_status(seq, MessageStatus.ABORTED)
                self.block_manager.free(seq)
                seq.set_step(0)
                continue

            if not __evict_for_seq(seq, num_required_blocks):
                self._set_message_status(seq, MessageStatus.WAITING)
                continue

            self.block_manager.allocate(seq, prealloc_size)
            self.block_trie.allocate(seq)

        return self.running, swap_in_map, swap_out_map, copy_map

    def schedule(self, is_prefill: bool, prealloc_size: int = 0):
        """Schedule inputs for next steps."""
        if is_prefill:
            output = self._schedule_prefill(0)
        else:
            output = self._schedule_decoding(prealloc_size)
        running, swap_in_map, swap_out_map, copy_map = output

        return SchedulerOutput(running=running, swap_in_map=swap_in_map, swap_out_map=swap_out_map, copy_map=copy_map)

    def _set_session_status(self, session_id: int, status: MessageStatus):
        """Setup the status of session.

        Args:
            session_id (int): The session id.
            status (MessageStatus): New status.
        """
        assert session_id in self.sessions
        session = self.sessions[session_id]
        session.status = status
        for seq in session.sequences.values():
            seq.status = status

    def stop_session(self, session_id: int):
        """Stop session.

        Args:
            session_id (int): The session id.
        """
        self._set_session_status(session_id, MessageStatus.STOPPED)

    def _remove_sequence(self, seq: SchedulerSequence):
        """Remove sequence(unsafe)

        Args:
            seq (SchedulerSequence): sequence to remove
        """
        self.block_manager.free(seq)
        self.state_manager.free(seq)
        seq.set_step(0)
        seq.session.remove_sequence(seq)

    def end_session(self, session_id: int):
        """End session.

        Args:
            session_id (int): The session id.
        """
        session = self.sessions[session_id]
        seqs = list(session.sequences.values())
        for seq in seqs:
            self._remove_sequence(seq)
        self.sessions.pop(session_id)

    def has_unfinished(self):
        """Check if there are any unfinished message."""
        return self.has_running() or self.has_waiting() or self.has_migration_done()

    def has_running(self):
        return self.num_running() > 0

    def has_waiting(self):
        return self.num_waiting() > 0

    def has_to_be_migrated(self):
        return self.num_to_be_migrated() > 0

    def has_migration_running(self):
        return self.num_running() > 0

    def has_migration_waiting(self):
        return self.num_migration_waiting() > 0

    def has_migration_done(self):
        return self.num_migration_done() > 0

    def get_block_tables(self, seqs: SeqList):
        """Get block table of the sequences."""
        return [self.block_manager.get_block_table(seq) for seq in seqs]

    def num_running(self):
        """Num running."""
        return self.seq_manager.num_sequences(MessageStatus.RUNNING)

    def num_waiting(self):
        """Num waiting."""
        return self.seq_manager.num_sequences(MessageStatus.WAITING)

    def num_to_be_migrated(self):
        """Num waiting."""
        return self.seq_manager.num_sequences(MessageStatus.TO_BE_MIGRATED)

    def num_migration_locked(self):
        """Num waiting."""
        return self.seq_manager.num_sequences(MessageStatus.MIGRATION_LOCKED)

    def num_migration_running(self):
        """Num migration running."""
        return self.seq_manager.num_sequences(MessageStatus.RUNNING_MIGRATION)

    def num_migration_done(self):
        """Num migration done."""
        return self.seq_manager.num_sequences(MessageStatus.MIGRATION_DONE)

    def num_migration_waiting(self):
        """Num waiting."""
        return self.seq_manager.num_sequences(MessageStatus.WAITING_MIGRATION)

    def num_locked(self):
        """Num locked."""
        return self.seq_manager.num_sequences(MessageStatus.LOCKED)

    def lock_running(self, running: SeqList):
        """Lock running sequence."""
        for seq in running:
            if seq.status == MessageStatus.RUNNING:
                self._set_message_status(seq, MessageStatus.LOCKED)

    def unlock_running(self, locked: SeqList):
        for seq in locked:
            if seq.status == MessageStatus.LOCKED:
                self._set_message_status(seq, MessageStatus.RUNNING)

    def lock_running_migration(self, running: SeqList):
        """Lock running sequence."""
        for seq in running:
            if seq.status == MessageStatus.RUNNING_MIGRATION:
                self._set_message_status(seq, MessageStatus.MIGRATION_LOCKED)

    def unlock_running_migration(self, locked: SeqList):
        """Unlock running migration."""
        for seq in locked:
            if seq.status == MessageStatus.MIGRATION_LOCKED:
                self._set_message_status(seq, MessageStatus.MIGRATION_DONE)

    def collect_migration_done(self):
        migration_done = self.migration_done
        for seq in migration_done:
            self._set_message_status(seq, MessageStatus.RUNNING)

    @property
    def schedule_metrics(self):
        return ScheduleMetrics(
            active_seqs=self.num_locked(),
            waiting_seqs=self.num_waiting() + self.num_running(),
            total_blocks=self.block_manager.num_gpu_blocks,
            free_blocks=self.block_manager.get_num_free_gpu_blocks(),
        )

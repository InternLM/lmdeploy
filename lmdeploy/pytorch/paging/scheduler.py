# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List

from lmdeploy.utils import get_logger

from ..block import PhysicalTokenBlock
from ..config import CacheConfig, SchedulerConfig
from ..messages import MessageStatus, SchedulerSequence, SchedulerSession
from .block_manager import BlockManager

logger = get_logger('lmdeploy')

SeqList = List[SchedulerSequence]
BlockTable = List[PhysicalTokenBlock]


def _find_seq_with_session_id(group: SeqList, session_id: int):
    return [seq for seq in group if seq.session_id == session_id]


@dataclass
class SchedulerOutput:
    """Output of schedule."""

    running: SeqList
    swap_in_map: Dict[int, int]
    swap_out_map: Dict[int, int]
    copy_map: Dict[int, int]
    block_tables: List[BlockTable]


class Scheduler:
    """Tools to schedule next step.

    Args:
        scheduler_config (SchedulerConfig): The config of scheduler.
        cache_config (CacheConfig): The config of cache info.
    """

    def __init__(self, scheduler_config: SchedulerConfig,
                 cache_config: CacheConfig) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config

        self.waiting: SeqList = []
        self.running: SeqList = []
        self.hanging: SeqList = []
        self.aborted: SeqList = []
        self.sessions: Dict[int, SchedulerSession] = OrderedDict()

        self.block_manager = BlockManager(
            cache_config.block_size,
            cache_config.num_gpu_blocks,
            cache_config.num_cpu_blocks,
        )

        self.eviction_helper = self.build_eviction_helper(
            self.scheduler_config.eviction_type, self.block_manager)

    def build_eviction_helper(ctx, eviction_type: str,
                              block_manager: BlockManager):
        if eviction_type == 'copy':
            from .eviction_helper import CopyEvictionHelper
            return CopyEvictionHelper(block_manager)
        elif eviction_type == 'recompute':
            from .eviction_helper import RecomputeEvictionHelper
            return RecomputeEvictionHelper(block_manager)
        else:
            raise TypeError(f'Unknown eviction type: {eviction_type}')

    def _set_message_status(self, message: SchedulerSequence,
                            status: MessageStatus):
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
        session = SchedulerSession(session_id)
        self.sessions[session_id] = session
        return session

    def add_sequence(self, seq: SchedulerSequence):
        """Add sequence.

        Args:
            seq (SchedulerSequence): New sequence.
        """
        assert (seq.session_id
                in self.sessions), f'Unknown session id {seq.session_id}'

        # push message to waiting queue
        self._set_message_status(seq, MessageStatus.WAITING)
        if seq.remain_output_len <= 0:
            seq.remain_output_len = \
                self.scheduler_config.max_request_output_len
        self.waiting.append(seq)

    def _schedule(self):
        """Schedule next step.

        Running is the messages to perform inference. Swap in/swap out is the
        table used to perform memory paging (between host and device)

        The schedule follow steps:
        1. Try allocate resources for all running sequence. If there are no
            enough resources, try `swap out` caches of hanging and waiting
            sequence. If there are still no enough resources, move the sequence
            to waiting.
        2. Check if sequence in the waiting list can be moved to running
        """
        running: SeqList = []
        swap_out_map: Dict[int, int] = {}
        swap_in_map: Dict[int, int] = {}
        copy_map: Dict[int, int] = {}
        block_manager = self.block_manager
        eviction_helper = self.eviction_helper

        def _to_running(seq: SchedulerSequence):
            self._set_message_status(seq, MessageStatus.RUNNING)
            running.append(seq)

        def _try_append_slot(seq):
            """try append slot."""
            if self.block_manager.can_append_slot(seq):
                self.block_manager.append_slot(seq)
                _to_running(seq)
                return True
            else:
                return False

        block_size = self.cache_config.block_size

        # 1. running
        for seq in self.running:
            # token + 1
            num_required_tokens = seq.num_required_tokens()
            seq.append_tokens(num_required_tokens, block_size)

            if len(seq.logical_blocks) > self.block_manager.num_gpu_blocks:
                # Reach max gpu cache size.
                logger.warning(f'session[{seq.session_id}] '
                               f'sequence[{seq.seq_id}] '
                               'reach max gpu size.')
                self._set_message_status(seq, MessageStatus.ABORTED)
                self.block_manager.free(seq)
                self.aborted.append(seq)

            if not _try_append_slot(seq):
                # try free unused cache from hanging and waiting
                do_running = False
                while eviction_helper.try_swap_out(self.hanging, self.waiting,
                                                   swap_out_map):
                    if _try_append_slot(seq):
                        do_running = True
                        break
                if not do_running:
                    # move to waiting
                    self._set_message_status(seq, MessageStatus.WAITING)
                    self.waiting.append(seq)

        max_batches = self.scheduler_config.max_batches

        # 2. waiting
        self.waiting = sorted(self.waiting, key=lambda seq: seq.arrive_time)
        while len(self.waiting) > 0 and len(running) < max_batches:
            seq = self.waiting[0]
            num_required_tokens = seq.num_required_tokens()
            seq.append_tokens(num_required_tokens, block_size)

            block_table = block_manager.get_block_table(seq)
            if block_table is not None:
                if not block_manager.can_append_slot(seq):
                    can_append = False
                    while eviction_helper.try_swap_out_seqs(
                            self.hanging, swap_out_map):
                        if block_manager.can_append_slot(seq):
                            can_append = True
                            break
                    if not can_append:
                        break
                if eviction_helper.need_swap_in(seq):
                    if eviction_helper.can_swap_in(seq):
                        eviction_helper.swap_in(seq, swap_in_map)
                    else:
                        break
                block_manager.append_slot(seq)
                self.waiting.pop(0)
                _to_running(seq)
            else:
                if not block_manager.can_allocate(seq):
                    can_alloc = False
                    while eviction_helper.try_swap_out_seqs(
                            self.hanging, swap_out_map):
                        if block_manager.can_allocate(seq):
                            can_alloc = True
                            break
                    if not can_alloc:
                        break
                # allocate session memory
                block_manager.allocate(seq)
                self.waiting.pop(0)
                _to_running(seq)

        self.running = running

        running = [
            msg for msg in self.running if msg.status == MessageStatus.RUNNING
        ]
        if len(running) == 0:
            logger.warning('No enough resources. Free gpu blocks: '
                           f'{self.block_manager.get_num_free_gpu_blocks()}, '
                           'Please end sessions.')
        return running, swap_in_map, swap_out_map, copy_map

    def schedule(self):
        """Schedule inputs for next steps."""
        running, swap_in_map, swap_out_map, copy_map = self._schedule()

        block_tables = [
            self.block_manager.get_block_table(seq) for seq in running
        ]

        return SchedulerOutput(
            running=running,
            swap_in_map=swap_in_map,
            swap_out_map=swap_out_map,
            copy_map=copy_map,
            block_tables=block_tables,
        )

    def _set_session_status(self, session_id: int, status: MessageStatus):
        """Setup the status of session.

        Args:
            session_id (int): The session id.
            status (MessageStatus): New status.
        """
        assert session_id in self.sessions
        session = self.sessions[session_id]
        session.status = status
        running_seq = _find_seq_with_session_id(self.running, session_id)
        waiting_seq = _find_seq_with_session_id(self.waiting, session_id)
        hanging_seq = _find_seq_with_session_id(self.hanging, session_id)

        for seq in running_seq + waiting_seq + hanging_seq:
            seq.status = status

    def stop_session(self, session_id: int):
        """Stop session.

        Args:
            session_id (int): The session id.
        """
        self._set_session_status(session_id, MessageStatus.STOPPED)

    def end_session(self, session_id: int):
        """End session.

        Args:
            session_id (int): The session id.
        """
        self._set_session_status(session_id, MessageStatus.ENDED)

    def has_unfinished(self):
        """Check if there are any unfinished message."""
        return self.waiting or self.running

    def _remove_sequence(self, seq: SchedulerSequence):
        """Remove sequence(unsafe)

        Args:
            seq (SchedulerSequence): sequence to remove
        """
        self.block_manager.free(seq)
        seq.session.sequences.pop(seq.seq_id)

    def update(self):
        """Update scheduler status after one step.

        A full step inference should include:
        0. end unused sequence
        1. schedule the running sequence
        2. forward with the running sequence
        3. update scheduler status
        """
        seq_to_remove = []
        session_id_to_remove = set()

        def _update_queue(group: SeqList, expect_status: MessageStatus):
            for seq in group:
                if seq.status == expect_status:
                    continue

                if seq.status == MessageStatus.WAITING:
                    self.waiting.append(seq)

                if seq.status == MessageStatus.STOPPED:
                    self.hanging.append(seq)

                # remove stopped session
                if seq.status == MessageStatus.ENDED:
                    seq_to_remove.append(seq)

            return [seq for seq in group if seq.status == expect_status]

        self.running = _update_queue(self.running, MessageStatus.RUNNING)
        self.waiting = _update_queue(self.waiting, MessageStatus.WAITING)
        self.hanging = _update_queue(self.hanging, MessageStatus.STOPPED)

        for session_id, session in self.sessions.items():
            if session.status == MessageStatus.ENDED:
                session_id_to_remove.add(session_id)

        # remove seqs
        for seq in seq_to_remove:
            self._remove_sequence(seq)

        # remove sessions
        for session_id in session_id_to_remove:
            self.sessions.pop(session_id)

    def get_block_tables(self, seqs: SeqList):
        """get block table of the sequences."""
        return [self.block_manager.get_block_table(seq) for seq in seqs]

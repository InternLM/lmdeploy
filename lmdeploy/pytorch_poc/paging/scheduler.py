# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List

from lmdeploy.pytorch_poc.block import PhysicalTokenBlock
from lmdeploy.pytorch_poc.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch_poc.messages import (MessageStatus, SchedulerSequence,
                                           SchedulerSession)
from lmdeploy.pytorch_poc.paging.block_manager import BlockManager
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def _find_message_with_session_id(message_list: List[SchedulerSequence],
                                  session_id: int):
    return [
        message for message in message_list if message.session_id == session_id
    ]


BlockTable = List[PhysicalTokenBlock]


@dataclass
class SchedulerOutput:
    """Output of schedule."""

    running: List[SchedulerSequence]
    swap_in_map: Dict[int, int]
    swap_out_map: Dict[int, int]
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

        self.waiting: List[SchedulerSequence] = []
        self.running: List[SchedulerSequence] = []
        self.hanging: List[SchedulerSequence] = []
        self.aborted: List[SchedulerSequence] = []
        self.sessions: Dict[int, SchedulerSession] = OrderedDict()

        self.block_manager = BlockManager(
            cache_config.block_size,
            cache_config.num_gpu_blocks,
            cache_config.num_cpu_blocks,
        )

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
        """Add message.

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
        running: List[SchedulerSequence] = []
        swap_out_map: Dict[int, int] = {}
        swap_in_map: Dict[int, int] = {}
        copy_map: Dict[int, int] = {}
        block_manager = self.block_manager

        def _to_running(seq: SchedulerSequence):
            self._set_message_status(seq, MessageStatus.RUNNING)
            running.append(seq)

        def _can_swap_out(seq: SchedulerSequence):
            block_table = block_manager.get_block_table(seq)
            if block_table is None or len(block_table) == 0:
                return False
            first_block = block_table[0]
            device = first_block.device
            return device == 'gpu'

        def _need_swap_in(seq: SchedulerSequence):
            block_table = block_manager.get_block_table(seq)
            if block_table is None or len(block_table) == 0:
                return False
            first_block = block_table[0]
            device = first_block.device
            return device == 'cpu'

        def _try_swap_out_seqs(seqs: List[SchedulerSequence]):
            for seq in seqs:
                if not _can_swap_out(seq):
                    continue
                if not block_manager.can_swap_out(seq):
                    continue
                swap_out_map.update(block_manager.swap_out(seq))
                return True

            return False

        def _try_swap_out():
            if _try_swap_out_seqs(self.hanging):
                return True
            else:
                return _try_swap_out_seqs(self.waiting)

        # 1. running
        for seq in self.running:
            # token + 1
            block_size = self.cache_config.block_size
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

            if self.block_manager.can_append_slot(seq):
                # append slot
                self.block_manager.append_slot(seq)
                _to_running(seq)
            else:
                # try free unused cache from hanging and waiting
                do_running = False
                while _try_swap_out():
                    if self.block_manager.can_append_slot():
                        # append slot
                        self.block_manager.append_slot(seq)
                        _to_running(seq)
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

            block_table = block_manager.get_block_table(seq)
            if block_table is not None:
                num_required_tokens = seq.num_required_tokens()
                seq.append_tokens(num_required_tokens, block_size)
                if not block_manager.can_append_slot(seq):
                    break
                if _need_swap_in(seq):
                    if block_manager.can_swap_in(seq):
                        swap_in_map.update(block_manager.swap_in(seq))
                    else:
                        break
                block_manager.append_slot(seq)
                self.waiting.pop(seq)
                _to_running(seq)
            else:
                if block_manager.can_allocate(seq):
                    # update logical blocks of seq
                    block_size = self.cache_config.block_size
                    # allocate session memory
                    block_manager.allocate(seq)
                    self.waiting.pop(seq)
                    _to_running(seq)
                else:
                    break

        self.running = running

        running = [
            msg for msg in self.running if msg.status == MessageStatus.RUNNING
        ]
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
        session = self.sessions[session_id]
        session.status = status
        running_msg = _find_message_with_session_id(self.running, session_id)
        waiting_msg = _find_message_with_session_id(self.waiting, session_id)

        for msg in running_msg + waiting_msg:
            msg.status = status

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

    def _remove_session(self, session_id: int):
        """Remove session.

        Args:
            session_id (int): The session id.
        """
        assert session_id in self.sessions
        session = self.sessions.pop(session_id)
        self.block_manager.free(session)

    def update(self):
        """Update scheduler status after one step.

        A full step inference should include:
        1. schedule
        2. forward
        3. update
        """
        session_id_to_remove = set()

        for msg in self.running:
            session = self.sessions[msg.session_id]
            session.history_length = sum(block.num_tokens
                                         for block in session.logical_blocks)

        def _update_queue(que: List[SchedulerSequence],
                          expect_status: MessageStatus):
            for msg in que:
                if msg.status == expect_status:
                    continue

                if msg.status == MessageStatus.ENDED:
                    session_id_to_remove.add(msg.session_id)

            return [msg for msg in que if msg.status == expect_status]

        self.waiting = _update_queue(self.waiting, MessageStatus.WAITING)
        self.running = _update_queue(self.running, MessageStatus.RUNNING)

        for session_id, session in self.sessions.items():
            if session.status == MessageStatus.ENDED:
                session_id_to_remove.add(session_id)

        # remove session
        for session_id in session_id_to_remove:
            self._remove_session(session_id)

    def get_block_tables(self, seqs: List[SchedulerSequence]):
        """get block table of the sequences."""
        return [self.block_manager.get_block_table(seq) for seq in seqs]

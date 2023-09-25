# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List

from lmdeploy.pytorch_poc.block import PhysicalTokenBlock
from lmdeploy.pytorch_poc.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch_poc.messages import (MessageStatus, SchedulerMessage,
                                           SchedulerSession)
from lmdeploy.pytorch_poc.paging.block_manager import BlockManager
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def _find_message_with_session_id(message_list: List[SchedulerMessage],
                                  session_id: int):
    return [
        message for message in message_list if message.session_id == session_id
    ]


BlockTable = List[PhysicalTokenBlock]


@dataclass
class SchedulerOutput:
    """Output of schedule."""

    running: List[SchedulerMessage]
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

        self.waiting: List[SchedulerMessage] = []
        self.running: List[SchedulerMessage] = []
        self.swapped: List[SchedulerMessage] = []
        self.aborted: List[SchedulerMessage] = []
        self.sessions: Dict[int, SchedulerSession] = OrderedDict()

        self.block_manager = BlockManager(
            cache_config.block_size,
            cache_config.num_gpu_blocks,
            cache_config.num_cpu_blocks,
        )

    def _set_message_status(self, message: SchedulerMessage,
                            status: MessageStatus):
        """Set status of message.

        Args:
            message (SchedulerMessage): message to setup status.
            status (MessageStatus): New message status.
        """
        session_id = message.session_id
        session = self.sessions[session_id]

        message.status = status
        session.status = status

    def add_session(self, session: SchedulerSession):
        """Add new session.

        Args:
            session (SchedulerSession): New session.
        """
        assert session.session_id not in self.sessions
        self.sessions[session.session_id] = session

    def get_sessions(self, messages: List[SchedulerMessage]):
        """Get sessions by message.

        Args:
            messages (List[SchedulerMessage]): Messages

        Returns:
            List[SchedulerSession]: Sessions of input messages.
        """
        return [self.sessions[msg.session_id] for msg in messages]

    def add_message(self, message: SchedulerMessage):
        """Add message.

        Args:
            message (SchedulerMessage): New message.
        """
        assert (message.session_id
                in self.sessions), f'Unknown session id {message.session_id}'

        # push message to waiting queue
        self._set_message_status(message, MessageStatus.WAITING)
        if message.max_request_output_len == 0:
            message.max_request_output_len = (
                self.scheduler_config.max_request_output_len)
        self.waiting.append(message)

    def _schedule(self):
        """Schedule next step.

        Running is the messages to perform inference. Swap in/swap out is the
        table used to perform memory paging (between host and device)
        """
        running: List[SchedulerMessage] = []
        swap_out_map: Dict[int, int] = {}
        swap_in_map: Dict[int, int] = {}

        def _get_session(msg: SchedulerMessage):
            session_id = msg.session_id
            assert session_id in self.sessions
            return self.sessions[session_id]

        def _to_running(msg: SchedulerMessage):
            self._set_message_status(msg, MessageStatus.RUNNING)
            running.append(msg)

        # check if running can be appended
        for msg in self.running:
            session = _get_session(msg)
            # token + 1
            block_size = self.cache_config.block_size
            session.append_tokens(1, block_size)

            if len(session.logical_blocks) > self.block_manager.num_gpu_blocks:
                logger.warning(f'session {session.session_id} '
                               'reach max gpu size.')
                self._set_message_status(msg, MessageStatus.ABORTED)
                self.aborted.append(msg)
            if self.block_manager.can_append_slot(session):
                # append slot
                self.block_manager.append_slot(session)
                _to_running(msg)
            else:
                # swap out
                assert self.block_manager.can_swap_out(
                    session), 'Can not swap out'
                tmp_map = self.block_manager.swap_out(session)
                swap_out_map.update(tmp_map)
                self._set_message_status(msg, MessageStatus.SWAP_OUT)
                self.swapped.append(msg)

        max_batches = self.scheduler_config.max_batches
        # swap in
        while len(self.swapped) > 0 and len(running) < max_batches:
            msg = self.swapped[0]
            session = _get_session(msg)

            if self.block_manager.can_swap_in(session):
                self.swapped.pop(0)
                tmp_map = self.block_manager.swap_in(session)
                swap_in_map.update(tmp_map)
                _to_running(msg)
            else:
                break

        # check waiting list
        while len(self.waiting) > 0 and len(running) < max_batches:
            msg = self.waiting[0]
            session = _get_session(msg)

            enable_running = False
            if self.block_manager.get_block_table(session):
                if self.block_manager.can_append_slot(session):
                    msg_length = len(msg.token_ids)
                    block_size = self.cache_config.block_size
                    session.append_tokens(msg_length, block_size)
                    self.block_manager.append_slot(session)
                    enable_running = True
            else:
                if self.block_manager.can_allocate(session):
                    # update logical blocks of session
                    msg_length = len(msg.token_ids)
                    block_size = self.cache_config.block_size
                    session.append_tokens(msg_length, block_size)

                    # allocate session memory
                    self.block_manager.allocate(session)
                    enable_running = True

            if enable_running:
                self.waiting.pop(0)
                _to_running(msg)

        self.running = running

        running = [
            msg for msg in self.running if msg.status == MessageStatus.RUNNING
        ]
        return running, swap_in_map, swap_out_map

    def schedule(self):
        """Schedule inputs for next steps."""
        running, swap_in_map, swap_out_map = self._schedule()

        sessions = [self.sessions[msg.session_id] for msg in running]

        block_tables = [
            self.block_manager.get_block_table(session) for session in sessions
        ]

        return SchedulerOutput(
            running=running,
            swap_in_map=swap_in_map,
            swap_out_map=swap_out_map,
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
        swaping_msg = _find_message_with_session_id(self.swapped, session_id)

        for msg in running_msg + waiting_msg + swaping_msg:
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
        return self.waiting or self.running or self.swapped

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

        def _update_queue(que: List[SchedulerMessage],
                          expect_status: MessageStatus):
            for msg in que:
                if msg.status == expect_status:
                    continue

                if msg.status == MessageStatus.ENDED:
                    session_id_to_remove.add(msg.session_id)

            return [msg for msg in que if msg.status == expect_status]

        self.waiting = _update_queue(self.waiting, MessageStatus.WAITING)
        self.running = _update_queue(self.running, MessageStatus.RUNNING)
        self.swapped = _update_queue(self.swapped, MessageStatus.SWAP_OUT)

        for session_id, session in self.sessions.items():
            if session.status == MessageStatus.ENDED:
                session_id_to_remove.add(session_id)

        # remove session
        for session_id in session_id_to_remove:
            self._remove_session(session_id)

    def get_block_tables(self, messages: List[SchedulerMessage]):
        """get block table of the messages."""
        sessions = [self.sessions[msg.session_id] for msg in messages]
        return [self.block_manager.get_block_table(sess) for sess in sessions]

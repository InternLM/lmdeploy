# modify from: https://github.com/vllm-project/vllm
from typing import List, Dict
from collections import OrderedDict
from dataclasses import dataclass
from lmdeploy.pytorch_poc.block import LogicalTokenBlock, PhysicalTokenBlock
from lmdeploy.pytorch_poc.config import SchedulerConfig, CacheConfig
from lmdeploy.pytorch_poc.paging.block_manager import BlockManager
from lmdeploy.pytorch_poc.messages import (SchedulerSession, SchedulerMessage,
                                           MessageStatus)


def _find_message_with_session_id(message_list: List[SchedulerMessage],
                                  session_id: int):
    return [
        message for message in message_list
        if message.session.session_id == session_id
    ]


BlockTable = List[PhysicalTokenBlock]


@dataclass
class SchedulerOutput:
    running: List[SchedulerMessage]
    swap_in_map: Dict[int, int]
    swap_out_map: Dict[int, int]
    block_tables: List[BlockTable]


class Scheduler:

    def __init__(self, scheduler_config: SchedulerConfig,
                 cache_config: CacheConfig) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config

        self.waiting: List[SchedulerMessage] = []
        self.running: List[SchedulerMessage] = []
        self.swapped: List[SchedulerMessage] = []
        self.sessions: Dict[int, SchedulerSession] = OrderedDict()

        self.block_manager = BlockManager(cache_config.block_size,
                                          cache_config.num_gpu_blocks,
                                          cache_config.num_cpu_blocks)

    def add_session(self, session: SchedulerSession):
        assert session.session_id not in self.sessions
        self.sessions[session.session_id] = self.sessions

    def add_message(self, message: SchedulerMessage):
        assert message.session_id in self.sessions, (
            f'Unknown session id {message.session_id}')

        # push message to waiting queue
        message.status = MessageStatus.WAITING
        self.waiting.append(message)

    def _schedule(self):

        running: List[SchedulerMessage] = []
        swap_out_map: Dict[int, int] = {}
        swap_in_map: Dict[int, int] = {}

        def _get_session(msg: SchedulerMessage):
            session_id = msg.session_id
            assert session_id in self.sessions
            return self.sessions[session_id]

        def _add_session_logical_block(session: SchedulerSession):
            logical_blocks = session.logical_blocks
            if len(logical_blocks) == 0:
                block = LogicalTokenBlock(
                    0, block_size=self.cache_config.block_size)
                block.append_tokens(1)
            else:
                block = logical_blocks[-1]
                if block.is_full():
                    block = LogicalTokenBlock(
                        len(logical_blocks),
                        block_size=self.cache_config.block_size)
                    logical_blocks.append(block)
                block.append_tokens(1)

        def _to_running(msg: SchedulerMessage):
            msg.status = MessageStatus.RUNNING
            running.append(msg)

        # check if running can be appended
        for msg in self.running:
            session = _get_session(msg)
            # token + 1
            _add_session_logical_block(session)

            if self.block_manager.can_append_slot(session):
                # append slot
                self.block_manager.append_slot(session)
                _to_running(msg)
            else:
                # swap out
                assert self.block_manager.can_swap_out(session), (
                    'Can not swap out')
                tmp_map = self.block_manager.swap_out(session)
                swap_out_map.update(tmp_map)
                msg.status = MessageStatus.SWAP_OUT
                self.swapped.append(msg)

        # swap in
        while len(self.swapped) > 0:
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
        while len(self.waiting) > 0:
            msg = self.waiting[0]
            session = _get_session(msg)

            enable_running = False
            if self.block_manager.get_block_table(session):
                if self.block_manager.can_append_slot(session):
                    self.block_manager.append_slot(session)
                    enable_running = True
            else:
                if self.block_manager.can_allocate(session):
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
        running, swap_in_map, swap_out_map = self._schedule()

        sessions = [self.sessions[msg.session_id] for msg in running]

        block_tables = [
            self.block_manager.get_block_table(session) for session in sessions
        ]

        return SchedulerOutput(
            running=running,
            swap_in_map=swap_in_map,
            swap_out_map=swap_out_map,
            block_tables=block_tables)

    def _set_session_status(self, session_id, status):
        session = self.sessions[session_id]
        session.status = status
        running_msg = _find_message_with_session_id(self.running, session_id)
        waiting_msg = _find_message_with_session_id(self.waiting, session_id)

        for msg in running_msg:
            msg.status = status
        for msg in waiting_msg:
            msg.status = status

    def stop_session(self, session_id):
        self._set_session_status(session_id, MessageStatus.STOPPED)

    def end_session(self, session_id):
        self._set_session_status(session_id, MessageStatus.ENDED)

    def has_unfinished(self):
        return self.waiting or self.running or self.swapped

    def _remove_session(self, session_id: int):
        assert session_id in self.sessions
        session = self.sessions.pop(session_id)
        self.block_manager.free(session)

    def update(self):

        session_id_to_remove = set()

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
        self.swapped = _update_queue(self.running, MessageStatus.RUNNING)

        # remove session
        for session_id in session_id_to_remove:
            self._remove_session(session_id)

# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Set, Union

from lmdeploy.utils import get_logger, logging_timer

from ..adapter.adapter import ADAPTER_MANAGER, SchedulerAdapter
from ..config import CacheConfig, SchedulerConfig
from ..messages import MessageStatus, SchedulerSequence, SchedulerSession
from .block_manager import DefaultBlockManager as BlockManager
from .block_manager import build_block_manager

logger = get_logger('lmdeploy')

SeqList = List[SchedulerSequence]
AdapterList = List[SchedulerAdapter]


def _find_seq_with_session_id(group: SeqList, session_id: int):
    return [seq for seq in group if seq.session_id == session_id]


@dataclass
class SchedulerOutput:
    """Output of schedule."""

    running: SeqList
    swap_in_map: Dict[int, int]
    swap_out_map: Dict[int, int]
    copy_map: Dict[int, int]
    adapters: AdapterList


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
        self.sessions: Dict[int, SchedulerSession] = OrderedDict()
        self.actived_adapters: Set[str] = set()

        self.block_manager = build_block_manager(cache_config)

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
        session = SchedulerSession(session_id, self.cache_config.block_size)
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
        self.waiting.append(seq)

    def add_adapter(self, adapter_path: str, adapter_name: str):
        """Add adapter.

        Args:
            adapter_path (str): The path of adapter.
            adapter_name (str): The name of the adapter.
        """
        adapter = ADAPTER_MANAGER.add_adapter_from_pretrained(
            adapter_path, adapter_name=adapter_name)
        self.block_manager.allocate_adapter(adapter)
        block_table = self.block_manager.get_block_table(
            adapter) - self.block_manager.num_gpu_blocks
        return adapter.build_weight_map(block_table)

    @logging_timer('SchedulePrefilling', logger)
    def _schedule_prefill(self):
        """Schedule for prefilling."""

        max_batches = self.scheduler_config.max_batches - len(self.running)
        block_manager = self.block_manager
        eviction_helper = self.eviction_helper
        swap_out_map: Dict[int, int] = dict()
        swap_in_map: Dict[int, int] = dict()
        copy_map: Dict[int, int] = dict()
        running: SeqList = []
        required_adapters = set(seq.adapter_name for seq in self.running)
        max_adapters = self.scheduler_config.max_active_adapters - len(
            required_adapters)
        token_count = 0

        def _to_running(seq: SchedulerSequence):
            """to running."""
            self._set_message_status(seq, MessageStatus.RUNNING)
            running.append(seq)
            nonlocal token_count
            token_count += seq.num_token_ids

        def _evict_until_can_append(seq: SchedulerSequence):
            """evict until can append."""
            while eviction_helper.try_swap_out_unused(self.hanging,
                                                      self.waiting[1:],
                                                      swap_out_map):
                if block_manager.can_append_slot(seq):
                    return True
            return False

        def _reorder_waiting():
            """reorder waiting."""
            self.waiting = sorted(self.waiting,
                                  key=lambda seq: seq.arrive_time)

        def _active_adapter(adapter_name):
            """active adapter of a seq."""
            if adapter_name is None:
                required_adapters.add(adapter_name)
                return
            if adapter_name not in required_adapters:
                adapter = ADAPTER_MANAGER.get_adapter(adapter_name)
                if not adapter.is_actived():
                    success, tmp_map = self.block_manager.try_swap_in(adapter)
                    assert success
                    swap_in_map.update(tmp_map)
            required_adapters.add(adapter_name)

        def _deactive_adapter(adapter_name):
            """deactive_adapter."""
            if adapter_name is None:
                return
            adapter = ADAPTER_MANAGER.get_adapter(adapter_name)
            if adapter.is_actived():
                success, tmp_map = self.block_manager.try_swap_out(adapter)
                assert success
                swap_out_map.update(tmp_map)

        if len(running) >= max_batches or len(self.waiting) == 0:
            return running, swap_in_map, swap_out_map, copy_map

        _reorder_waiting()
        while len(self.waiting) > 0 and len(running) < max_batches:
            seq = self.waiting[0]

            if (len(running) > 0 and token_count + seq.num_token_ids >
                    self.cache_config.max_prefill_token_num):
                break

            # limit number of adapters
            if len(required_adapters) >= max_adapters:
                if seq.adapter_name not in required_adapters:
                    break

            if not block_manager.can_allocate(seq):
                if not _evict_until_can_append(seq):
                    break

            if eviction_helper.need_swap_in(seq):
                if not eviction_helper.try_swap_in(seq, swap_in_map):
                    break
            # allocate session memory
            block_manager.allocate(seq)
            _active_adapter(seq.adapter_name)
            self.waiting.pop(0)
            _to_running(seq)

        deactive_adapters = self.actived_adapters.difference(required_adapters)
        for adapter_name in deactive_adapters:
            _deactive_adapter(adapter_name)

        self.actived_adapters = required_adapters

        self.running += running
        return running, swap_in_map, swap_out_map, copy_map

    @logging_timer('ScheduleDecoding', logger)
    def _schedule_decoding(self, prealloc_size: int = 0):
        """schedule decoding."""
        assert len(self.running) != 0

        block_manager = self.block_manager
        eviction_helper = self.eviction_helper
        swap_out_map: Dict[int, int] = dict()
        swap_in_map: Dict[int, int] = dict()
        copy_map: Dict[int, int] = dict()
        running: SeqList = []

        def _to_running(seq: SchedulerSequence):
            """to running."""
            self._set_message_status(seq, MessageStatus.RUNNING)
            running.append(seq)

        def _try_append_slot(seq):
            """try append slot."""
            if self.block_manager.num_required_blocks(seq, prealloc_size) == 0:
                _to_running(seq)
                return True
            if block_manager.can_append_slot(seq, prealloc_size):
                block_manager.append_slot(seq, prealloc_size)
                _to_running(seq)
                return True
            return False

        def _evict_until_can_append(seq: SchedulerSequence):
            """evict until can append."""
            while eviction_helper.try_swap_out_unused(self.hanging,
                                                      self.waiting,
                                                      swap_out_map):
                if block_manager.can_append_slot(seq, prealloc_size):
                    return True
            return False

        # 1. running
        for seq in self.running:
            # token + n

            if len(seq.logical_blocks) > self.block_manager.num_gpu_blocks:
                # Reach max gpu cache size.
                logger.warning(f'session[{seq.session_id}] '
                               f'sequence[{seq.seq_id}] '
                               'reach max gpu size.')
                self._set_message_status(seq, MessageStatus.ABORTED)
                self.block_manager.free(seq)

            if not _try_append_slot(seq):
                # try free unused cache from waiting
                if _evict_until_can_append(seq):
                    _try_append_slot(seq)
                else:
                    # move to waiting
                    self._set_message_status(seq, MessageStatus.WAITING)
                    self.waiting.insert(0, seq)

        self.running = running
        return running, swap_in_map, swap_out_map, copy_map

    @classmethod
    def _get_adapter_list(cls, adapter_names: List[str]):
        adapters = [
            ADAPTER_MANAGER.get_adapter(name) for name in adapter_names
        ]
        return adapters

    def schedule(self, is_prefill: bool, prealloc_size: int = 0):
        """Schedule inputs for next steps."""
        if is_prefill:
            output = self._schedule_prefill()
        else:
            output = self._schedule_decoding(prealloc_size)
        running, swap_in_map, swap_out_map, copy_map = output

        adapters = self._get_adapter_list(self.actived_adapters)

        return SchedulerOutput(running=running,
                               swap_in_map=swap_in_map,
                               swap_out_map=swap_out_map,
                               copy_map=copy_map,
                               adapters=adapters)

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

    def has_running(self):
        return len(self.running) > 0

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

    def get_block_tables(self, seqs: Union[SeqList, AdapterList]):
        """get block table of the sequences."""
        return [self.block_manager.get_block_table(seq) for seq in seqs]

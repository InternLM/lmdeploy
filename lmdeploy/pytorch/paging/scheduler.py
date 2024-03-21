# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Set, Union

from lmdeploy.utils import get_logger, logging_timer

from ..adapter.adapter import ADAPTER_MANAGER, SchedulerAdapter
from ..config import CacheConfig, SchedulerConfig
from ..messages import (MessageStatus, SchedulerSequence, SchedulerSession,
                        SequenceManager)
from .block_manager import build_block_manager
from .radix_tree_manager import RadixTreeManager

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

        self.sessions: Dict[int, SchedulerSession] = OrderedDict()
        self.actived_adapters: Set[str] = set()

        self.block_manager = build_block_manager(cache_config)

        self.seq_manager = SequenceManager()

        self.rtree_manager = RadixTreeManager(self.cache_config,
                                              self.add_session(-1))

        self.eviction_helper = self.build_eviction_helper(
            self.scheduler_config.eviction_type)

    @property
    def waiting(self):
        """get waiting sequence."""
        seq_map = self.seq_manager.get_sequences(MessageStatus.WAITING)
        return list(seq_map.values())

    @property
    def running(self):
        """get waiting sequence."""
        seq_map = self.seq_manager.get_sequences(MessageStatus.RUNNING)
        return list(seq_map.values())

    @property
    def hanging(self):
        """get waiting sequence."""
        seq_map = self.seq_manager.get_sequences(MessageStatus.STOPPED)
        return list(seq_map.values())

    def build_eviction_helper(self, eviction_type: str):
        if eviction_type == 'copy':
            logger.warning('Copy eviction has been deprecated.')
            eviction_type = 'recompute'
        if eviction_type == 'recompute':
            from .eviction_helper import RecomputeEvictionHelper
            return RecomputeEvictionHelper(self)
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
        session = SchedulerSession(session_id,
                                   self.cache_config.block_size,
                                   seq_manager=self.seq_manager)
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

        current_running = self.running
        max_batches = self.scheduler_config.max_batches - len(current_running)
        block_manager = self.block_manager
        eviction_helper = self.eviction_helper
        swap_out_map: Dict[int, int] = dict()
        swap_in_map: Dict[int, int] = dict()
        copy_map: Dict[int, int] = dict()
        running: SeqList = []
        required_adapters = set(seq.adapter_name for seq in current_running)
        max_adapters = self.scheduler_config.max_active_adapters - len(
            required_adapters)
        token_count = 0
        sorted_nodes = self.rtree_manager.sort_nodes(
            max_visit_time=self.rtree_manager.step_time - 2)

        def _to_running(seq: SchedulerSequence):
            """to running."""
            seq.status = MessageStatus.RUNNING
            running.append(seq)
            nonlocal token_count
            token_count += seq.num_token_ids
            self.rtree_manager.update_sequence(seq)

        def _evict_until_can_append(seq: SchedulerSequence):
            """evict until can append."""
            if block_manager.can_allocate(seq):
                return True
            if len(sorted_nodes) == 0:
                return False
            return eviction_helper.evict_for_seq(seq, sorted_nodes)

        def _reorder_waiting():
            """reorder waiting."""
            return sorted(self.waiting, key=lambda seq: seq.arrive_time)

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

        num_waiting = self.seq_manager.num_sequences(MessageStatus.WAITING)
        if (len(running) >= max_batches or num_waiting == 0):
            return running, swap_in_map, swap_out_map, copy_map

        waiting = _reorder_waiting()
        while len(waiting) > 0 and len(running) < max_batches:
            seq = waiting.pop(0)

            if (len(running) > 0 and token_count + seq.num_token_ids >
                    self.cache_config.max_prefill_token_num):
                break

            # limit number of adapters
            if len(required_adapters) >= max_adapters:
                if seq.adapter_name not in required_adapters:
                    break

            if seq.seq_id not in self.rtree_manager.seq_node_map:
                self.rtree_manager.add_sequence(seq,
                                                self.cache_config.shared_cache)

            if not _evict_until_can_append(seq):
                break

            # allocate session memory
            block_manager.allocate(seq)
            _active_adapter(seq.adapter_name)
            _to_running(seq)

        deactive_adapters = self.actived_adapters.difference(required_adapters)
        for adapter_name in deactive_adapters:
            _deactive_adapter(adapter_name)

        self.actived_adapters = required_adapters

        return running, swap_in_map, swap_out_map, copy_map

    @logging_timer('ScheduleDecoding', logger)
    def _schedule_decoding(self, prealloc_size: int = 0):
        """schedule decoding."""

        running = self.running
        assert len(running) != 0

        block_manager = self.block_manager
        eviction_helper = self.eviction_helper
        swap_out_map: Dict[int, int] = dict()
        swap_in_map: Dict[int, int] = dict()
        copy_map: Dict[int, int] = dict()
        sorted_nodes = self.rtree_manager.sort_nodes(
            max_visit_time=self.rtree_manager.step_time - 1)

        def _try_append_slot(seq):
            """try append slot."""
            if self.block_manager.num_required_blocks(seq, prealloc_size) == 0:
                return True
            if block_manager.can_append_slot(seq, prealloc_size):
                block_manager.append_slot(seq, prealloc_size)
                return True
            return False

        def _evict_until_can_append(seq: SchedulerSequence):
            """evict until can append."""
            if block_manager.can_allocate(seq):
                return True
            if len(sorted_nodes) == 0:
                return False
            return eviction_helper.evict_for_seq(seq, sorted_nodes)

        # 1. running
        for seq in running:
            # token + n

            if len(seq.logical_blocks) > self.block_manager.num_gpu_blocks:
                # Reach max gpu cache size.
                logger.warning(f'session[{seq.session_id}] '
                               f'sequence[{seq.seq_id}] '
                               'reach max gpu size.')
                self._set_message_status(seq, MessageStatus.ABORTED)
                node = self.rtree_manager.seq_node_map[seq.seq_id]
                self.block_manager.free(seq, node.num_blocks)
                self.rtree_manager.remove_sequence(seq)

            if not _evict_until_can_append(seq):
                self._set_message_status(seq, MessageStatus.WAITING)
                continue

            _try_append_slot(seq)
            self.rtree_manager.update_sequence(seq)

        return self.running, swap_in_map, swap_out_map, copy_map

    @classmethod
    def _get_adapter_list(cls, adapter_names: List[str]):
        adapters = [
            ADAPTER_MANAGER.get_adapter(name) for name in adapter_names
        ]
        return adapters

    def schedule(self, is_prefill: bool, prealloc_size: int = 0):
        """Schedule inputs for next steps."""
        self.rtree_manager.step_time += 1
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
        node = self.rtree_manager.seq_node_map[seq.seq_id]
        self.block_manager.free(seq, node.num_blocks)
        seq.session.remove_sequence(seq)
        self.rtree_manager.remove_sequence(seq)

    def end_session(self, session_id: int):
        """End session.

        Args:
            session_id (int): The session id.
        """
        session = self.sessions[session_id]
        seqs = list(session.sequences.values())
        for seq in seqs:
            seq.status = MessageStatus.ENDED
            self._remove_sequence(seq)
        self.sessions.pop(session_id)

    def has_unfinished(self):
        """Check if there are any unfinished message."""
        return self.has_running() or self.has_waiting()

    def has_running(self):
        return self.seq_manager.num_sequences(MessageStatus.RUNNING) > 0

    def has_waiting(self):
        return self.seq_manager.num_sequences(MessageStatus.WAITING) > 0

    def get_block_tables(self, seqs: Union[SeqList, AdapterList]):
        """get block table of the sequences."""
        return [self.block_manager.get_block_table(seq) for seq in seqs]

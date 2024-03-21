# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from lmdeploy.pytorch.block import LogicalTokenBlocks
from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.messages import (MessageStatus, SchedulerSequence,
                                       SchedulerSession)


def _np_empty(dtype=np.int64):
    return np.empty((0, ), dtype=dtype)


@dataclass
class TreeNode:
    node_id: int
    sequence: SchedulerSequence
    token_slice: slice
    block_slice: slice
    visit_time: int = 0
    _parent: 'TreeNode' = field(default=None, init=False, repr=False)
    _children: Dict[int, 'TreeNode'] = field(init=False, default_factory=dict)

    @property
    def token_ids(self):
        """token_ids."""
        return self.sequence.history_cache[self.token_slice]

    @property
    def blocks(self):
        """blocks."""
        return self.sequence.logical_blocks[self.block_slice]

    @property
    def parent(self):
        """parent getter."""
        return self._parent

    @parent.setter
    def parent(self, value: 'TreeNode'):
        """parent setter."""

        def __clear_old_parent():
            old_parent = self._parent
            if old_parent is not None:
                old_parent.children.pop(self.node_id, None)

        def __update_new_parent():
            if value is not None:
                value.children[self.node_id] = self

        __clear_old_parent()
        __update_new_parent()
        self._parent = value

    @property
    def children(self):
        """parent getter."""
        return self._children

    @property
    def num_blocks(self):
        stop = self.block_slice.stop
        if stop is None:
            stop = self.block_slice.indices(self.sequence.num_blocks)[1]
        return stop - self.block_slice.start

    @property
    def num_token_ids(self):
        stop = self.token_slice.stop
        if stop is None:
            stop = self.token_slice.indices(self.sequence.num_all_ids)[1]
        return stop - self.token_slice.start

    @property
    def num_cum_tokens(self):
        stop = self.token_slice.stop
        if stop is None:
            return self.sequence.num_all_ids
        return stop

    @property
    def num_cum_blocks(self):
        stop = self.block_slice.stop
        if stop is None:
            return self.sequence.num_blocks
        return stop

    @property
    def num_parent_tokens(self):
        return self.token_slice.start

    @property
    def num_parent_blocks(self):
        return self.block_slice.start

    def update_visit_time(self, time: int, window_size: int = -1):
        """update visit time."""
        self.visit_time = max(time, self.visit_time)
        if self.parent is not None:
            if window_size < 0:
                self.parent.update_visit_time(time)
                return
            next_window_size = window_size - self.num_token_ids
            if next_window_size > 0:
                self.parent.update_visit_time(time, next_window_size)

    def __str__(self):
        children_str = [str(child) for child in self.children]
        children_str = ', '.join(children_str)
        parent = 'None' if self.parent is None else f'<{self.parent.node_id}>'
        return f"""TreeNode(
node_id={self.node_id},
token_ids={self.token_ids},
blocks={self.blocks},
parent={parent},
children=[{children_str}],
visit_time={self.visit_time})"""

    def __repr__(self):
        return str(self)


class RadixTreeManager:
    """Radix tree manager."""

    def __init__(self, cache_config: CacheConfig,
                 session: SchedulerSession) -> None:

        self.cache_config = cache_config
        self._block_size = self.cache_config.block_size
        self._session = session

        self._roots: Dict[str, TreeNode] = dict()
        self.nodes: Dict[int, TreeNode] = dict()
        self.seq_node_map: Dict[int, TreeNode] = dict()

        self._max_node_id = 0
        self.step_time = 0

    @property
    def block_size(self):
        return self._block_size

    def _get_next_node_id(self):
        """get next node id."""
        ret = self._max_node_id
        self._max_node_id += 1
        return ret

    def _session_add_sequence(self,
                              token_ids: np.ndarray,
                              adapter_name: str = None,
                              blocks: LogicalTokenBlocks = None,
                              step: int = 0):
        """session add sequence."""
        seq = self._session.add_sequence(token_ids, adapter_name=adapter_name)
        if step > 0:
            seq.set_step(step)
        if blocks is not None:
            seq.logical_blocks = blocks
        seq.status = MessageStatus.PROTECTED
        return seq

    def get_root(self, adapter_name: str):
        """get root."""
        if adapter_name not in self._roots:
            seq = self._session_add_sequence(_np_empty(), adapter_name)
            node = TreeNode(self._get_next_node_id(), seq, slice(0, 0),
                            slice(0, 0))
            node.parent = None
            self._roots[adapter_name] = node
            self.nodes[node.node_id] = node
        return self._roots[adapter_name]

    def new_node(self,
                 seq: SchedulerSequence,
                 token_slice: slice = None,
                 block_slice: slice = None,
                 parent: TreeNode = None,
                 is_leaf: bool = False):
        """new node."""
        if token_slice is None:
            token_slice = slice(0, None)
        if block_slice is None:
            block_slice = slice(0, None)
        if parent is None:
            parent = self.get_root(seq.adapter_name)

        node = TreeNode(self._get_next_node_id(),
                        seq,
                        token_slice=token_slice,
                        block_slice=block_slice)
        node.parent = parent
        self.nodes[node.node_id] = node
        if is_leaf:
            self.seq_node_map[seq.seq_id] = node

        return node

    def match_sequence(self, seq: SchedulerSequence):
        """match sequence."""

        def __match_tokens(tokens0: np.ndarray, tokens1: np.ndarray):
            match_size = min(len(tokens0), len(tokens1))
            if match_size == 0:
                return 0
            diff = tokens0[:match_size] == tokens1[:match_size]
            ret = np.argmin(diff)
            if ret == 0 and diff[0]:
                ret = match_size
            return ret

        def __match_children(node: TreeNode, token_ids: np.ndarray,
                             num_matched: int):
            """match children."""
            best_match_node = node
            best_match_len = num_matched
            if len(token_ids) < self.block_size:
                return best_match_node, best_match_len

            for child in node.children.values():
                if child.num_token_ids < self.block_size:
                    continue
                tmp_num_matched = __match_tokens(token_ids, child.token_ids)
                if tmp_num_matched < self.block_size:
                    continue
                tmp_match_node, tmp_match_len = __match_children(
                    child, token_ids[tmp_num_matched:],
                    num_matched + tmp_num_matched)
                if tmp_match_len > best_match_len:
                    best_match_len = tmp_match_len
                    best_match_node = tmp_match_node
            return best_match_node, best_match_len

        return __match_children(self.get_root(seq.adapter_name), seq.all_ids,
                                0)

    def add_sequence(self, seq: SchedulerSequence, share_cache: bool = True):
        """add sequence."""

        def __parent_exists(matched_node: TreeNode):
            """parent exists."""
            matched_seq = matched_node.sequence
            seq.logical_blocks = matched_seq.logical_blocks.clone()
            num_cum_tokens = matched_node.num_cum_tokens
            num_cum_blocks = matched_node.num_cum_blocks
            if num_cum_tokens > 0:
                seq.set_step(num_cum_tokens)
            node = self.new_node(seq,
                                 slice(num_cum_tokens, None),
                                 slice(num_cum_blocks, None),
                                 parent=matched_node,
                                 is_leaf=True)
            return node

        def __parent_not_exists(matched_node: TreeNode, match_len: int):
            """parent not exists."""
            num_blocks = match_len // seq.block_size
            parent = self.split_node(matched_node, num_blocks)
            node = __parent_exists(parent)
            return node

        share_cache = seq.num_blocks == 0 and share_cache

        if share_cache:
            matched_node, matched_len = self.match_sequence(seq)
        else:
            matched_node = self.get_root(seq.adapter_name)
            matched_len = 0

        if matched_node.sequence.session == self._session:
            node = __parent_exists(matched_node)
        else:
            node = __parent_not_exists(matched_node, matched_len)
        node.update_visit_time(max(0, self.step_time - 2),
                               self.cache_config.window_size)

    def split_node(self, node: TreeNode, num_blocks: int):
        """split node."""
        old_parent = node.parent
        old_parent_seq = node.sequence
        assert num_blocks >= old_parent.num_cum_blocks
        num_tokens = num_blocks * self.block_size
        blocks = LogicalTokenBlocks(old_parent_seq.logical_blocks[:num_blocks])
        token_ids = old_parent_seq.all_ids[:num_tokens]
        new_parent_seq = self._session_add_sequence(
            token_ids,
            adapter_name=old_parent_seq.adapter_name,
            blocks=blocks,
            step=num_tokens)
        new_parent = self.new_node(new_parent_seq,
                                   slice(node.token_slice.start, num_tokens),
                                   slice(node.block_slice.start, num_blocks),
                                   parent=old_parent,
                                   is_leaf=False)
        node.token_slice = slice(num_tokens, node.token_slice.stop)
        node.block_slice = slice(num_blocks, node.block_slice.stop)
        node.parent = new_parent

        return new_parent

    def update_sequence(self, seq: SchedulerSequence):
        """update sequence."""
        assert seq.seq_id in self.seq_node_map
        node = self.seq_node_map[seq.seq_id]
        node.update_visit_time(self.step_time,
                               window_size=self.cache_config.window_size)

    def remove_sequence(self, seq: SchedulerSequence):
        """remove sequence."""
        assert seq.seq_id in self.seq_node_map
        node = self.seq_node_map[seq.seq_id]
        self.seq_node_map.pop(seq.seq_id)
        self.remove_node(node)

    def remove_node(self, node: TreeNode):
        """remove node."""

        def __remove_from_manager(node: TreeNode):
            """remove from manager."""
            node.parent = None
            if node.sequence.session == self._session:
                self._session.sequences.pop(node.sequence.seq_id)
            self.nodes.pop(node.node_id)

        def __update_leaf(node: TreeNode):
            """update leaf seq."""
            if node.sequence.seq_id not in self.seq_node_map:
                __remove_from_manager(node)

        def __update_empty_child(node: TreeNode, remove_len: int):
            """update child sequence."""
            seq = node.sequence
            new_num_blocks = seq.num_blocks - remove_len
            new_num_tokens = new_num_blocks * self.block_size
            seq.logical_blocks.resize(new_num_blocks)
            seq.set_step(new_num_tokens)
            node.parent = node.parent.parent

        if len(node.children) == 0:
            __update_leaf(node)
            return

        num_remove_blocks = node.num_blocks
        for child in list(node.children.values()):
            if child.num_blocks == 0:
                __update_empty_child(child, num_remove_blocks)
            else:
                raise RuntimeError('Unsupported no empty child.')
        __remove_from_manager(node)

    def sort_nodes(self,
                   ignore_empty: bool = True,
                   max_visit_time: int = None):
        """sort nodes."""

        def __sort_key(n: TreeNode):
            """get sort key."""
            key = n.visit_time
            if len(n.children) == 0:
                # children first
                key -= 0.5
            return key

        nodes = self.nodes.values()
        if ignore_empty:
            nodes = filter(lambda n: n.num_blocks > 0, nodes)
        if max_visit_time is not None:
            nodes = filter(lambda n: n.visit_time <= max_visit_time, nodes)

        nodes = list(nodes)
        nodes = sorted(nodes, key=__sort_key)
        return nodes

    def get_all_nodes(self, seq: SchedulerSequence):
        """get all nodes."""
        if seq.seq_id not in self.seq_node_map:
            return []

        node = self.seq_node_map[seq.seq_id]
        ret = []
        while node.parent is not None:
            ret.append(node)
            node = node.parent

        return ret

    def get_seq_node(self, seq: SchedulerSequence):
        """get seq node."""
        return self.seq_node_map.get(seq.seq_id, None)

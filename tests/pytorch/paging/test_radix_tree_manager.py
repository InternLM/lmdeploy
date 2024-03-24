import numpy as np
import pytest

from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.messages import SchedulerSession, SequenceManager
from lmdeploy.pytorch.paging.radix_tree_manager import (RadixTreeManager,
                                                        TreeNode)


@pytest.fixture
def block_size():
    yield 16


@pytest.fixture
def seq_manager():
    yield SequenceManager()


def _rand_array(num: int):
    return np.random.randint(0, 100, (num, ))


class TestTreeNode:

    @pytest.fixture
    def session(self, block_size, seq_manager):
        yield SchedulerSession(-1, block_size, seq_manager)

    def test_set_parent(self, session):
        seq = session.add_sequence(_rand_array(3))
        node0 = TreeNode(0, seq, slice(0, None), slice(0, None))
        seq = session.add_sequence(_rand_array(3))
        node1 = TreeNode(1, seq, slice(0, None), slice(0, None))
        seq = session.add_sequence(_rand_array(3))
        node2 = TreeNode(2, seq, slice(0, None), slice(0, None))

        node1.parent = node0
        assert len(node0.children) == 1
        assert list(node0.children.values())[0] == node1

        node1.parent = node2
        assert len(node0.children) == 0
        assert len(node2.children) == 1
        assert list(node2.children.values())[0] == node1

    def test_update_visit_time(self, session, block_size):
        seq = session.add_sequence(_rand_array(2 * block_size))
        node0 = TreeNode(0, seq, slice(0, 2 * block_size), slice(0, 2))
        seq = session.add_sequence(_rand_array(4 * block_size))
        node1 = TreeNode(1, seq, slice(2 * block_size, 4 * block_size),
                         slice(2, 4))
        seq = session.add_sequence(_rand_array(6 * block_size))
        node2 = TreeNode(2, seq, slice(4 * block_size, 6 * block_size),
                         slice(4, 6))

        node0.parent = None
        node1.parent = node0
        node2.parent = node1

        visit_time = 5
        node2.update_visit_time(visit_time)
        assert node0.visit_time == visit_time
        assert node1.visit_time == visit_time
        assert node2.visit_time == visit_time

        old_visit_time = visit_time
        visit_time = 10
        node1.update_visit_time(visit_time)
        assert node0.visit_time == visit_time
        assert node1.visit_time == visit_time
        assert node2.visit_time == old_visit_time


class TestRadixTreeManager:

    @pytest.fixture
    def num_cpu_blocks(self):
        yield 64

    @pytest.fixture
    def num_gpu_blocks(self):
        yield 128

    @pytest.fixture
    def cache_config(
        self,
        block_size,
        num_cpu_blocks,
        num_gpu_blocks,
    ):
        yield CacheConfig(block_size=block_size,
                          num_cpu_blocks=num_cpu_blocks,
                          num_gpu_blocks=num_gpu_blocks)

    @pytest.fixture
    def session(self, block_size, seq_manager):
        yield SchedulerSession(1, block_size, seq_manager)

    @pytest.fixture
    def rt_session(self, block_size, seq_manager):
        yield SchedulerSession(-1, block_size, seq_manager)

    @pytest.fixture
    def rtree_manager(self, cache_config, rt_session):
        yield RadixTreeManager(cache_config, rt_session)

    def test_get_root(self, rtree_manager, rt_session):
        adapter_name0 = None
        adapter_name1 = 'adapter1'

        assert len(rt_session.sequences) == 0
        assert len(rtree_manager.nodes) == 0
        assert len(rtree_manager._roots) == 0

        root0 = rtree_manager.get_root(adapter_name0)

        assert len(rt_session.sequences) == 1
        assert len(rtree_manager.nodes) == 1
        assert len(rtree_manager._roots) == 1

        new_root0 = rtree_manager.get_root(adapter_name0)

        assert len(rt_session.sequences) == 1
        assert len(rtree_manager.nodes) == 1
        assert len(rtree_manager._roots) == 1
        assert root0 == new_root0

        root1 = rtree_manager.get_root(adapter_name1)

        assert len(rt_session.sequences) == 2
        assert len(rtree_manager.nodes) == 2
        assert len(rtree_manager._roots) == 2
        assert root0 != root1

    def test_new_node(self, rtree_manager, session):
        seq0 = session.add_sequence(_rand_array(4))

        node0 = rtree_manager.new_node(seq0, slice(0, 4), slice(0, 4))

        assert len(rtree_manager.nodes) == 2
        assert node0.sequence == seq0
        assert node0.token_slice == slice(0, 4)
        assert node0.block_slice == slice(0, 4)
        assert node0.parent is not None
        assert seq0.seq_id not in rtree_manager.seq_node_map

        seq1 = session.add_sequence(_rand_array(8))
        node1 = rtree_manager.new_node(seq1,
                                       slice(4, None),
                                       slice(4, None),
                                       parent=node0,
                                       is_leaf=True)

        assert len(rtree_manager.nodes) == 3
        assert node1.sequence == seq1
        assert node1.parent == node0
        assert seq1.seq_id in rtree_manager.seq_node_map

    def test_match_sequence(self, rtree_manager, session, block_size):
        seq = session.add_sequence(np.array([1] * (block_size * 2 + 1)))
        seq.logical_blocks.append(np.array([0, 1]))
        seq_node = rtree_manager.new_node(seq,
                                          slice(0, None),
                                          slice(0, None),
                                          is_leaf=True)

        # test no match
        seq1 = session.add_sequence(np.array([2] * (block_size * 2 + 1)))
        matched_node, matched_len = rtree_manager.match_sequence(seq1)
        assert matched_node == rtree_manager.get_root(seq1.adapter_name)
        assert matched_len == 0

        # test partial match
        seq2 = session.add_sequence(
            np.array([1] * (block_size + 1) + [2] * block_size))
        matched_node, matched_len = rtree_manager.match_sequence(seq2)
        assert matched_node == seq_node
        assert matched_len == block_size + 1

        # test match parent
        seq = session.add_sequence(np.array([3] * (block_size * 2)))
        seq.logical_blocks.append(np.array([2, 3]))
        seq_node = rtree_manager.new_node(seq,
                                          slice(0, block_size * 2),
                                          slice(0, 2),
                                          is_leaf=False)
        seq1 = session.add_sequence(np.array([], dtype=np.int64))
        rtree_manager.new_node(seq1,
                               slice(block_size * 2, None),
                               slice(2, None),
                               parent=seq_node,
                               is_leaf=True)
        seq2 = session.add_sequence(np.array([3] * (block_size * 2)))
        matched_node, matched_len = rtree_manager.match_sequence(seq2)
        assert matched_node == seq_node
        assert matched_len == block_size * 2

    def test_add_sequence(self, rtree_manager, session, block_size):
        # test add to leaf
        token_ids = np.array([1] * (block_size * 2 + 1))
        seq = session.add_sequence(token_ids)

        assert len(rtree_manager.nodes) == 0
        rtree_manager.step_time = 4
        rtree_manager.add_sequence(seq)
        blocks = np.array([0, 1])
        seq.logical_blocks.append(blocks)
        assert seq.num_blocks == 2
        seq_node = rtree_manager.seq_node_map[seq.seq_id]
        assert len(rtree_manager.nodes) == 2  # root + seq
        assert seq_node.visit_time == rtree_manager.step_time - 2

        # test match no exist parent
        token_ids = np.array([1] * block_size + [2] * block_size)
        seq = session.add_sequence(token_ids)
        rtree_manager.add_sequence(seq)
        seq_node = rtree_manager.seq_node_map[seq.seq_id]
        assert len(rtree_manager.nodes) == 4
        assert seq.num_blocks == 1
        assert seq_node.visit_time == rtree_manager.step_time - 2

        # parent exist
        token_ids = np.array([1] * block_size + [3] * block_size)
        seq = session.add_sequence(token_ids)
        rtree_manager.add_sequence(seq)
        seq_node = rtree_manager.seq_node_map[seq.seq_id]
        assert len(rtree_manager.nodes) == 5
        assert seq_node.visit_time == rtree_manager.step_time - 2

    def test_update_sequence(self, rtree_manager, session, block_size):
        token_ids = np.array([1] * (block_size * 2 + 1))
        seq = session.add_sequence(token_ids)
        rtree_manager.add_sequence(seq)
        blocks = np.array([0, 1])
        seq.logical_blocks.append(blocks)
        rtree_manager.step_time = 4

        rtree_manager.update_sequence(seq)
        seq_node = rtree_manager.seq_node_map[seq.seq_id]
        assert seq_node.visit_time == rtree_manager.step_time

    def test_remove_sequence(self, rtree_manager, session, block_size):
        token_ids = np.array([1] * (block_size * 2 + 1))
        seq = session.add_sequence(token_ids)
        rtree_manager.add_sequence(seq)
        blocks = np.array([0, 1])
        seq.logical_blocks.append(blocks)
        assert len(rtree_manager.nodes) == 2

        rtree_manager.remove_sequence(seq)
        assert len(rtree_manager.nodes) == 1
        assert seq.seq_id not in rtree_manager.seq_node_map

    def test_remove_node(self, rtree_manager, session, block_size):
        token_ids = np.array([1] * (block_size * 2))
        seq = session.add_sequence(token_ids)
        rtree_manager.add_sequence(seq)
        blocks = np.array([0, 1])
        seq.logical_blocks.append(blocks)

        token_ids = np.array([1] * (block_size * 2) + [2] * block_size)
        seq1 = session.add_sequence(token_ids)
        rtree_manager.add_sequence(seq1)
        assert len(rtree_manager.nodes) == 4

        # test remove leaf
        node = rtree_manager.seq_node_map[seq1.seq_id]
        rtree_manager.remove_node(node)
        assert len(rtree_manager.nodes) == 4
        assert seq1.num_blocks == 2
        assert seq1.num_history_ids == 2 * block_size

        # test remove parent
        parent = rtree_manager.seq_node_map[seq.seq_id].parent
        rtree_manager.remove_node(parent)
        assert len(rtree_manager.nodes) == 3
        assert seq.num_blocks == 0
        assert seq1.num_blocks == 0
        assert seq.num_history_ids == 0
        assert seq1.num_history_ids == 0

    def test_sort_node(self, rtree_manager, session, block_size):
        token_ids = np.array([1] * block_size)
        seq1 = session.add_sequence(token_ids)
        rtree_manager.add_sequence(seq1)
        blocks = np.array([0])
        seq1.logical_blocks.append(blocks)
        seq1_node = rtree_manager.get_seq_node(seq1)

        token_ids = np.array([1] * block_size + [2] * block_size)
        seq2 = session.add_sequence(token_ids)
        rtree_manager.add_sequence(seq2)
        blocks = np.array([1])
        seq2.logical_blocks.append(blocks)
        seq2_node = rtree_manager.get_seq_node(seq2)

        token_ids = np.array([1] * block_size + [3] * block_size)
        seq3 = session.add_sequence(token_ids)
        rtree_manager.add_sequence(seq3)
        blocks = np.array([2])
        seq3.logical_blocks.append(blocks)
        seq3_node = rtree_manager.get_seq_node(seq3)

        assert len(rtree_manager.nodes) == 5
        parent = seq1_node.parent
        assert parent == seq2_node.parent
        assert parent == seq3_node.parent

        step_time = 5
        rtree_manager.step_time = step_time
        parent.visit_time = step_time - 1
        seq2_node.visit_time = step_time - 1
        seq3_node.visit_time = step_time

        nodes = rtree_manager.sort_nodes()
        assert len(nodes) == 3
        assert nodes[0] == seq2_node
        assert nodes[1] == parent
        assert nodes[2] == seq3_node

    def test_get_all_nodes(self, rtree_manager, session, block_size):
        token_ids = np.array([1] * block_size)
        seq1 = session.add_sequence(token_ids)
        rtree_manager.add_sequence(seq1)
        blocks = np.array([0])
        seq1.logical_blocks.append(blocks)

        token_ids = np.array([1] * block_size + [2] * block_size)
        seq2 = session.add_sequence(token_ids)
        rtree_manager.add_sequence(seq2)

        nodes = rtree_manager.get_all_nodes(seq2)
        assert len(nodes) == 2
        assert nodes[0].sequence == seq2

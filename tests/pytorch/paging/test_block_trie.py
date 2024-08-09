import numpy as np
import pytest

from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.messages import SchedulerSession
from lmdeploy.pytorch.paging.block_manager import build_block_manager
from lmdeploy.pytorch.paging.block_trie import BlockTrie


class TestBlockTire:

    @pytest.fixture
    def block_size(self):
        yield 16

    @pytest.fixture
    def num_cpu_blocks(self):
        yield 4

    @pytest.fixture
    def num_gpu_blocks(self):
        yield 16

    @pytest.fixture
    def cache_config(self, block_size, num_cpu_blocks, num_gpu_blocks):
        yield CacheConfig(max_batches=256,
                          block_size=block_size,
                          num_cpu_blocks=num_cpu_blocks,
                          num_gpu_blocks=num_gpu_blocks,
                          enable_prefix_caching=True)

    @pytest.fixture
    def block_mgr(self, cache_config):
        yield build_block_manager(cache_config)

    @pytest.fixture
    def block_trie(self, cache_config, block_mgr):
        yield BlockTrie(cache_config, block_mgr)

    def test_allocate(self, block_trie, block_mgr, block_size):
        allocator = block_trie.allocator
        sess = SchedulerSession(0, block_size)
        token_ids = ([1] * block_size + [2] * block_size)
        token_ids += [3] * (block_size // 2)
        seq = sess.add_sequence(token_ids)

        # first allocate
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        logical_blocks = seq.logical_blocks
        assert len(logical_blocks) == 3
        ref_cnt = allocator.get_ref_count(logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [2, 2, 1])
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.num_matched == block_size * 2
        assert np.array_equal(node.tokens, [2] * block_size)
        assert np.array_equal(node.parent.tokens, [1] * block_size)
        assert node in block_trie.leaves
        assert node.parent not in block_trie.leaves

        # append
        seq.update_token_ids([4] * block_size)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        logical_blocks = seq.logical_blocks
        assert len(logical_blocks) == 4
        ref_cnt = allocator.get_ref_count(logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [2, 2, 2, 1])
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.num_matched == block_size * 3
        expect_tokens = [3] * (block_size // 2) + [4] * (block_size // 2)
        assert np.array_equal(node.tokens, expect_tokens)
        assert node in block_trie.leaves
        assert len(block_trie.leaves) == 1

    def test_match(self, block_trie, block_mgr, block_size):
        allocator = block_trie.allocator
        sess = SchedulerSession(0, block_size)

        # initialize cache
        token_ids = ([1] * block_size + [2] * block_size)
        token_ids += [3] * (block_size // 2)
        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)

        # test1
        token_ids = ([1] * block_size + [3] * block_size)
        seq = sess.add_sequence(token_ids)
        block_trie.match(seq)
        logical_blocks = seq.logical_blocks
        assert len(logical_blocks) == 1
        ref_cnt = allocator.get_ref_count(logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [3])
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.num_matched == block_size
        assert np.array_equal(node.tokens, [1] * block_size)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        assert len(block_trie.leaves) == 2

        # test2
        token_ids = ([1] * block_size + [2] * block_size)
        token_ids += [4] * (block_size // 2)
        seq = sess.add_sequence(token_ids)
        block_trie.match(seq)
        logical_blocks = seq.logical_blocks
        assert len(logical_blocks) == 2
        ref_cnt = allocator.get_ref_count(logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [4, 3])

    def test_evict(self, block_trie, block_size, num_gpu_blocks):
        block_mgr = block_trie.block_manager
        sess = SchedulerSession(0, block_size)
        token_ids = ([1] * block_size * (num_gpu_blocks - 1))
        token_ids += [2] * (block_size // 2)
        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        assert block_mgr.get_num_free_gpu_blocks() == 0

        # test free
        block_mgr.free(seq)
        seq.set_step(0)
        assert block_mgr.get_num_free_gpu_blocks() == 1

        # test evict
        leaf = next(iter(block_trie.leaves))
        block_trie.evict(4)
        new_leaf = next(iter(block_trie.leaves))
        assert leaf != new_leaf
        assert block_mgr.get_num_free_gpu_blocks() == 5

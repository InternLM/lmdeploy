import numpy as np
import pytest

from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.messages import SamplingParam, SchedulerSession, SequenceManager, SequenceMeta
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

    @pytest.fixture
    def num_moe_layers(self):
        yield 4

    @pytest.fixture
    def experts_topk(self):
        yield 4

    @pytest.fixture
    def seq_manager(self, block_size):
        from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
        strategy = ARSequenceStrategy()
        seq_meta = SequenceMeta(block_size, strategy=strategy)
        yield SequenceManager(seq_meta)

    def test_with_routed_experts(self, block_trie, block_mgr, seq_manager, num_moe_layers, experts_topk):

        def _get_routed_experts(size, value):
            return np.full((size, num_moe_layers, experts_topk), value, dtype=np.int32)

        sess = SchedulerSession(0, seq_manager)
        block_size = sess.seq_meta.block_size
        token_ids = ([1] * block_size + [2] * block_size)
        all_routed_experts = [_get_routed_experts(block_size, 1), _get_routed_experts(block_size, 2)]
        token_ids += [3] * (block_size // 2)
        all_routed_experts += [_get_routed_experts(block_size // 2, 3)]
        seq = sess.add_sequence(token_ids, sampling_param=SamplingParam(return_routed_experts=True))
        all_routed_experts += [_get_routed_experts(block_size - 1, 4)]
        routed_experts = np.concatenate(all_routed_experts, axis=0)
        seq.update_token_ids([4] * block_size, routed_experts=routed_experts)

        # test allocate
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.routed_experts is not None
        target_routed_experts = np.concatenate(
            [_get_routed_experts(block_size // 2, 3),
             _get_routed_experts(block_size // 2, 4)], axis=0)
        assert np.array_equal(node.routed_experts, target_routed_experts)

        # test match
        seq_query = sess.add_sequence(token_ids, sampling_param=SamplingParam(return_routed_experts=True))
        block_trie.match(seq_query)
        assert seq_query.all_routed_experts is not None
        assert len(seq_query.all_routed_experts) == block_size * 2
        assert np.array_equal(seq_query.all_routed_experts.get_real(), np.concatenate(all_routed_experts[:2], axis=0))

    def test_allocate(self, block_trie, block_mgr, seq_manager):
        allocator = block_trie.allocator
        sess = SchedulerSession(0, seq_manager)
        block_size = sess.seq_meta.block_size
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

    def test_match(self, block_trie, block_mgr, seq_manager):
        allocator = block_trie.allocator
        sess = SchedulerSession(0, seq_manager)
        block_size = sess.seq_meta.block_size

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

    def test_evict(self, block_trie, seq_manager, num_gpu_blocks):
        block_mgr = block_trie.block_manager
        sess = SchedulerSession(0, seq_manager)
        block_size = sess.seq_meta.block_size
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

    def test_reset(self, block_trie, block_mgr, seq_manager, num_gpu_blocks):
        allocator = block_trie.allocator
        sess = SchedulerSession(0, seq_manager)
        block_size = sess.seq_meta.block_size

        # initialize cache
        token_ids = ([1] * block_size + [2] * block_size)
        token_ids += [3] * (block_size // 2)
        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)

        token_ids = ([1] * block_size + [3] * block_size)
        seq1 = sess.add_sequence(token_ids)
        block_trie.match(seq1)
        block_mgr.allocate(seq1)
        block_trie.allocate(seq1)

        ref_cnt = allocator.get_ref_count(seq.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [3, 2, 1])
        ref_cnt = allocator.get_ref_count(seq1.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [3, 2])
        block_trie.reset()
        assert len(block_trie.leaves) == 0
        ref_cnt = allocator.get_ref_count(seq.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [2, 1, 1])
        ref_cnt = allocator.get_ref_count(seq1.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [2, 1])
        block_mgr.free(seq)
        ref_cnt = allocator.get_ref_count(seq1.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [1, 1])
        block_mgr.free(seq1)
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks

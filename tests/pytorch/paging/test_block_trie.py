import numpy as np
import pytest

from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.messages import SchedulerSession
from lmdeploy.pytorch.multimodal.data_type import MultiModalTensor
from lmdeploy.pytorch.paging.block_manager import build_block_manager
from lmdeploy.pytorch.paging.block_trie import BlockTrie


class TestBlockTrie:

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

    def test_allocate_multimodals(self, block_trie, block_mgr, block_size):
        allocator = block_trie.allocator
        sess = SchedulerSession(0, block_size)
        half_block_size = block_size // 2
        # test case 1 single block
        token_ids = [1] * block_size + [2] * half_block_size
        multimodals = dict(image=[
            MultiModalTensor(data=None, start=0, end=block_size, meta=dict(hash_value='image_0')),
        ])
        seq = sess.add_sequence(token_ids, multimodals=multimodals)

        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        logical_blocks = seq.logical_blocks
        assert len(logical_blocks) == 2
        ref_cnt = allocator.get_ref_count(logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [2, 1])
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.mm_hashes == tuple(['image_0'])
        assert node.num_matched == block_size
        assert np.array_equal(node.tokens, [1] * block_size)
        assert node in block_trie.leaves
        assert len(block_trie.leaves) == 1
        assert node.parent not in block_trie.leaves
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 2)
        block_mgr.free(seq)
        block_trie.evict(2)
        assert block_mgr.get_num_free_gpu_blocks() == block_mgr.num_gpu_blocks
        assert len(block_trie.leaves) == 0

        # test case 2 multi blocks, but last block not full

        token_ids = [1] * (block_size + half_block_size) + [2] * 2 * block_size
        multimodals = dict(image=[
            MultiModalTensor(data=None,
                             start=block_size + half_block_size,
                             end=3 * block_size + half_block_size,
                             meta=dict(hash_value='image_0')),
        ])
        seq = sess.add_sequence(token_ids, multimodals=multimodals)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        logical_blocks = seq.logical_blocks
        assert len(logical_blocks) == 4
        ref_cnt = allocator.get_ref_count(logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [2, 1, 1, 1])
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.mm_hashes is None
        assert node.num_matched == block_size
        assert np.array_equal(node.tokens, [1] * block_size)
        assert node in block_trie.leaves
        assert len(block_trie.leaves) == 1
        assert node.parent not in block_trie.leaves
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 4)
        block_mgr.free(seq)
        block_trie.evict(1)
        assert block_mgr.get_num_free_gpu_blocks() == block_mgr.num_gpu_blocks
        assert len(block_trie.leaves) == 0

        # append text token to make last block full
        seq.update_token_ids([3] * block_size)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        logical_blocks = seq.logical_blocks
        assert len(logical_blocks) == 5
        ref_cnt = allocator.get_ref_count(logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [2, 2, 2, 2, 1])
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.num_matched == block_size * 4
        assert node.mm_hashes == tuple(['image_0'])
        expect_tokens = [2] * half_block_size + [3] * half_block_size
        assert np.array_equal(node.tokens, expect_tokens)
        assert np.array_equal(node.parent.tokens, [2] * block_size)
        assert node in block_trie.leaves
        assert len(block_trie.leaves) == 1
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 5)
        block_mgr.free(seq)
        block_trie.evict(5)
        assert block_mgr.get_num_free_gpu_blocks() == block_mgr.num_gpu_blocks
        assert len(block_trie.leaves) == 0

        # test 3 multi images
        quarter_block_size = block_size // 4
        token_ids = [1] * quarter_block_size + [2] * half_block_size + [3] * quarter_block_size + [
            4
        ] * quarter_block_size
        multimodals = dict(image=[
            MultiModalTensor(data=None, start=0, end=quarter_block_size, meta=dict(hash_value='image_0')),
            MultiModalTensor(
                data=None, start=block_size - quarter_block_size, end=block_size, meta=dict(hash_value='image_1')),
        ])
        seq = sess.add_sequence(token_ids, multimodals=multimodals)

        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        logical_blocks = seq.logical_blocks
        assert len(logical_blocks) == 2
        ref_cnt = allocator.get_ref_count(logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [2, 1])
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.mm_hashes == tuple(['image_0', 'image_1'])
        assert node.num_matched == block_size
        expect_tokens = [1] * quarter_block_size + [2] * half_block_size + [3] * quarter_block_size
        assert np.array_equal(node.tokens, expect_tokens)
        assert node in block_trie.leaves
        assert len(block_trie.leaves) == 1
        assert node.parent not in block_trie.leaves

        # append image token, but last vision block is not full
        token_ids = [4] * block_size + [5] * half_block_size + [6] * block_size
        multimodals = dict(image=[
            MultiModalTensor(data=None, start=0, end=block_size, meta=dict(hash_value='image_2')),
            MultiModalTensor(data=None,
                             start=2 * block_size - half_block_size,
                             end=3 * block_size - half_block_size,
                             meta=dict(hash_value='image_3')),
        ])
        seq.update_token_ids(token_ids, multimodals=multimodals)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        logical_blocks = seq.logical_blocks
        assert len(logical_blocks) == 4
        ref_cnt = allocator.get_ref_count(logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [2, 1, 1, 1])
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.mm_hashes == tuple(['image_0', 'image_1'])
        assert node.num_matched == block_size
        expect_tokens = [1] * quarter_block_size + [2] * half_block_size + [3] * quarter_block_size
        assert np.array_equal(node.tokens, expect_tokens)

        # append text to make last vision block full
        seq.update_token_ids([7] * block_size)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        logical_blocks = seq.logical_blocks
        assert len(logical_blocks) == 5
        ref_cnt = allocator.get_ref_count(logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [2, 2, 2, 2, 1])
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.mm_hashes == tuple(['image_3'])
        assert node.num_matched == 4 * block_size
        expect_tokens = [6] * (block_size - quarter_block_size) + [7] * quarter_block_size
        assert np.array_equal(node.tokens, expect_tokens)
        parent = node.parent
        assert parent is not None
        assert parent.mm_hashes == tuple(['image_2', 'image_3'])
        assert parent.num_matched == 3 * block_size
        expect_tokens = [4] * quarter_block_size + [5] * half_block_size + [6] * quarter_block_size
        assert np.array_equal(parent.tokens, expect_tokens)

    @pytest.mark.test
    def test_match_multimodals(self, block_trie, block_mgr, block_size):
        allocator = block_trie.allocator
        sess = SchedulerSession(0, block_size)
        half_block_size = block_size // 2
        quarter_block_size = block_size // 4

        # initialize cache with single image

        token_ids = [1] * half_block_size  # text
        token_ids += [2] * block_size  # img0
        multimodals = dict(image=[
            MultiModalTensor(data=None, start=half_block_size, end=3 *
                             half_block_size, meta=dict(hash_value='image_0')),
        ])
        seq0 = sess.add_sequence(token_ids, multimodals=multimodals)

        block_mgr.allocate(seq0)
        block_trie.allocate(seq0)

        # test with last vision block unfull
        token_ids = [1] * half_block_size  # text
        token_ids += [2] * block_size  # img0
        multimodals = dict(image=[
            MultiModalTensor(data=None, start=half_block_size, end=3 *
                             half_block_size, meta=dict(hash_value='image_0')),
        ])
        seq_prob = sess.add_sequence(token_ids, multimodals=multimodals)

        block_trie.match(seq_prob)
        last_node = getattr(seq_prob.logical_blocks, 'last_shared_node', None)
        assert last_node.parent is None
        assert last_node.num_matched == 0
        assert last_node.mm_hashes is None

        seq0.update_token_ids(token_ids=[3] * block_size)
        block_mgr.allocate(seq0)
        block_trie.allocate(seq0)

        # prob seq last vision block is unfull
        block_trie.match(seq_prob)
        last_node = getattr(seq_prob.logical_blocks, 'last_shared_node', None)
        assert last_node.parent is None
        assert last_node.num_matched == 0
        assert last_node.mm_hashes is None

        # prob seq last vision block is full
        seq_prob.update_token_ids([3] * block_size + [0] * block_size)
        block_trie.match(seq_prob)
        last_node = getattr(seq_prob.logical_blocks, 'last_shared_node', None)
        assert last_node.parent is not None
        assert last_node.num_matched == block_size * 2
        assert last_node.mm_hashes == tuple(['image_0'])
        assert np.array_equal(last_node.tokens, [2] * half_block_size + [3] * half_block_size)
        parent = last_node.parent
        assert parent is not None
        assert parent.num_matched == block_size
        assert parent.mm_hashes == tuple(['image_0'])
        assert np.array_equal(parent.tokens, [1] * half_block_size + [2] * half_block_size)
        ref_cnt = allocator.get_ref_count(seq_prob.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [3, 3])
        block_mgr.free(seq_prob)
        assert len(seq_prob.logical_blocks) == 0
        ref_cnt = allocator.get_ref_count(seq0.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [2, 2, 1])

        # test with different image
        token_ids = [1] * half_block_size  # text
        token_ids += [2] * block_size  # img1
        token_ids += [3] * block_size
        multimodals = dict(image=[
            MultiModalTensor(data=None, start=half_block_size, end=3 *
                             half_block_size, meta=dict(hash_value='image_1')),
        ])
        seq_prob = sess.add_sequence(token_ids, multimodals=multimodals)

        block_trie.match(seq_prob)
        last_node = getattr(seq_prob.logical_blocks, 'last_shared_node', None)
        assert last_node.parent is None
        assert last_node.num_matched == 0
        assert last_node.mm_hashes is None

        # test with multi image
        block_mgr.free(seq0)
        block_trie.evict(3)
        assert block_mgr.get_num_free_gpu_blocks() == block_mgr.num_gpu_blocks

        # test multi images
        token_ids = [1] * half_block_size  # text
        token_ids += [100] * 3 * half_block_size  # img 0
        token_ids += [2] * block_size  # text
        token_ids += [200] * quarter_block_size  # img 1
        token_ids += [3] * half_block_size  # text
        token_ids += [300] * half_block_size  # img 2
        token_ids += [4] * half_block_size  # text

        multimodals = dict(image=[
            MultiModalTensor(data=None, start=half_block_size, end=2 * block_size, meta=dict(hash_value='image_0')),
            MultiModalTensor(data=None,
                             start=3 * block_size,
                             end=3 * block_size + quarter_block_size,
                             meta=dict(hash_value='image_1')),
            MultiModalTensor(data=None,
                             start=4 * block_size - quarter_block_size,
                             end=4 * block_size + quarter_block_size,
                             meta=dict(hash_value='image_2')),
        ])
        seq0 = sess.add_sequence(token_ids, multimodals=multimodals)
        block_mgr.allocate(seq0)
        block_trie.allocate(seq0)

        # test one same image
        token_ids = [1] * half_block_size  # text
        token_ids += [100] * 3 * half_block_size  # img 0
        token_ids += [2] * half_block_size  # haft text, not match

        multimodals = dict(image=[
            MultiModalTensor(data=None, start=half_block_size, end=2 * block_size, meta=dict(hash_value='image_0')),
        ])
        seq_prob = sess.add_sequence(token_ids, multimodals=multimodals)

        block_trie.match(seq_prob)
        last_node = getattr(seq_prob.logical_blocks, 'last_shared_node', None)
        assert last_node.parent is not None
        assert last_node.num_matched == 2 * block_size
        assert last_node.mm_hashes == tuple(['image_0'])
        assert np.array_equal(last_node.tokens, [100] * block_size)
        ref_cnt = allocator.get_ref_count(seq_prob.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [3, 3])
        block_mgr.free(seq_prob)

        # test with two same image
        token_ids = [1] * half_block_size  # text
        token_ids += [100] * 3 * half_block_size  # img 0
        token_ids += [2] * block_size  # text
        token_ids += [200] * quarter_block_size  # img 1
        token_ids += [6] * block_size  # diff text

        multimodals = dict(image=[
            MultiModalTensor(data=None, start=half_block_size, end=2 * block_size, meta=dict(hash_value='image_0')),
            MultiModalTensor(data=None,
                             start=3 * block_size,
                             end=3 * block_size + quarter_block_size,
                             meta=dict(hash_value='image_1')),
        ])
        seq_prob = sess.add_sequence(token_ids, multimodals=multimodals)

        block_trie.match(seq_prob)
        last_node = getattr(seq_prob.logical_blocks, 'last_shared_node', None)
        assert last_node.parent is not None
        assert last_node.num_matched == 3 * block_size
        assert last_node.mm_hashes is None
        assert np.array_equal(last_node.tokens, [2] * block_size)
        ref_cnt = allocator.get_ref_count(seq_prob.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [3, 3, 3])
        block_mgr.free(seq_prob)

        # test with two same image
        token_ids = [1] * half_block_size  # text
        token_ids += [100] * 3 * half_block_size  # img 0
        token_ids += [2] * block_size  # text
        token_ids += [200] * quarter_block_size  # img 1
        token_ids += [3] * half_block_size  # text
        token_ids += [300] * half_block_size  # img 2
        token_ids += [4] * half_block_size  # text
        token_ids += [5] * block_size  # text

        multimodals = dict(image=[
            MultiModalTensor(data=None, start=half_block_size, end=2 * block_size, meta=dict(hash_value='image_0')),
            MultiModalTensor(data=None,
                             start=3 * block_size,
                             end=3 * block_size + quarter_block_size,
                             meta=dict(hash_value='image_1')),
            MultiModalTensor(data=None,
                             start=4 * block_size - quarter_block_size,
                             end=4 * block_size + quarter_block_size,
                             meta=dict(hash_value='image_2')),
        ])
        seq_prob = sess.add_sequence(token_ids, multimodals=multimodals)

        block_trie.match(seq_prob)
        last_node = getattr(seq_prob.logical_blocks, 'last_shared_node', None)
        assert last_node.parent is not None
        assert last_node.num_matched == 3 * block_size
        assert last_node.mm_hashes is None
        assert np.array_equal(last_node.tokens, [2] * block_size)
        ref_cnt = allocator.get_ref_count(seq_prob.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [3, 3, 3])
        block_mgr.free(seq_prob)

        # test with all images match
        seq0.update_token_ids([5] * 2 * block_size)
        block_mgr.allocate(seq0)
        block_trie.allocate(seq0)

        block_trie.match(seq_prob)
        last_node = getattr(seq_prob.logical_blocks, 'last_shared_node', None)
        assert last_node.parent is not None
        assert last_node.num_matched == 5 * block_size
        assert last_node.mm_hashes == tuple(['image_2'])
        assert np.array_equal(last_node.tokens,
                              [300] * quarter_block_size + [4] * half_block_size + [5] * quarter_block_size)
        ref_cnt = allocator.get_ref_count(seq_prob.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [3, 3, 3, 3, 3])

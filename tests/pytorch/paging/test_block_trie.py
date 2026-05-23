import numpy as np
import pytest
import torch

from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.messages import SequenceMeta
from lmdeploy.pytorch.multimodal.data_type import MultiModalData
from lmdeploy.pytorch.paging import Scheduler
from lmdeploy.vl.constants import Modality


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
    def max_batch_size(self):
        yield 4

    @pytest.fixture
    def cache_config(self, block_size, num_cpu_blocks, num_gpu_blocks, max_batch_size):
        yield CacheConfig(max_batches=max_batch_size,
                          block_size=block_size,
                          num_cpu_blocks=num_cpu_blocks,
                          num_gpu_blocks=num_gpu_blocks,
                          enable_prefix_caching=True)

    @pytest.fixture
    def scheduler_config(self, max_batch_size):
        yield SchedulerConfig(max_batches=max_batch_size,
                              max_session_len=128,
                              max_request_output_len=64,
                              eviction_type='recompute')

    @pytest.fixture
    def seq_meta(self, block_size):
        from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
        strategy = ARSequenceStrategy()
        yield SequenceMeta(block_size, strategy=strategy)

    @pytest.fixture
    def scheduler(self, cache_config, scheduler_config, seq_meta):
        yield Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)

    @pytest.fixture
    def block_mgr(self, scheduler):
        yield scheduler.block_manager

    @pytest.fixture
    def block_trie(self, scheduler):
        yield scheduler.block_trie

    def _image_multimodals(self, start: int, end: int, value: float, image_token_id: int = 99):
        data = torch.full((2, 2), value, dtype=torch.float32)
        return dict(image=[MultiModalData(data=data,
                                          start=start,
                                          end=end,
                                          meta=dict(image_token_id=image_token_id))])

    def _modal_data(self, start: int, end: int, value: float, modality: Modality):
        data = torch.full((2, 2), value, dtype=torch.float32)
        return MultiModalData(data=data,
                              start=start,
                              end=end,
                              modality=modality,
                              meta=dict(token_id=int(value)))

    def test_allocate(self, block_trie, block_mgr, scheduler):
        allocator = block_trie.allocator
        sess = scheduler.add_session(0)
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

    def test_match(self, block_trie, block_mgr, scheduler):
        allocator = block_trie.allocator
        sess = scheduler.add_session(0)
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

    def test_match_multimodal_same_hash(self, block_trie, block_mgr, scheduler):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size + [99] * block_size + [2] * block_size + [3]

        seq = sess.add_sequence(token_ids, multimodals=self._image_multimodals(block_size, block_size * 2, 1.0))
        block_mgr.allocate(seq)
        block_trie.allocate(seq)

        seq = sess.add_sequence(token_ids, multimodals=self._image_multimodals(block_size, block_size * 2, 1.0))
        block_trie.match(seq)

        assert len(seq.logical_blocks) == 3
        assert seq.num_history_ids == block_size * 3
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.num_matched == block_size * 3

    def test_match_multimodal_different_hash(self, block_trie, block_mgr, scheduler):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size + [99] * block_size + [2] * block_size + [3]

        seq = sess.add_sequence(token_ids, multimodals=self._image_multimodals(block_size, block_size * 2, 1.0))
        block_mgr.allocate(seq)
        block_trie.allocate(seq)

        seq = sess.add_sequence(token_ids, multimodals=self._image_multimodals(block_size, block_size * 2, 2.0))
        block_trie.match(seq)

        assert len(seq.logical_blocks) == 1
        assert seq.num_history_ids == block_size
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.num_matched == block_size

    def test_match_multimodal_clamps_before_split_span(self, block_trie, block_mgr, scheduler):
        allocator = block_trie.allocator
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        start = block_size // 2
        end = block_size + block_size // 2
        token_ids = [99] * block_size + [99] * block_size + [3]

        seq = sess.add_sequence(token_ids, multimodals=self._image_multimodals(start, end, 1.0))
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        cached_blocks = seq.logical_blocks.get_real_blocks()[:1]

        token_ids = [99] * block_size + [98] * block_size + [3]
        seq = sess.add_sequence(token_ids, multimodals=self._image_multimodals(start, end, 1.0))
        block_trie.match(seq)

        assert len(seq.logical_blocks) == 0
        assert seq.num_history_ids == 0
        assert np.array_equal(allocator.get_ref_count(cached_blocks), [2])
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.num_matched == 0

    def test_match_multimodal_extra_hash_order_is_canonical(self, block_trie, block_mgr, scheduler):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [99] * block_size + [3]
        image = self._modal_data(2, 6, 1.0, Modality.IMAGE)
        video = self._modal_data(8, 12, 2.0, Modality.VIDEO)

        seq = sess.add_sequence(token_ids, multimodals=dict(image=[image], video=[video]))
        block_mgr.allocate(seq)
        block_trie.allocate(seq)

        image = self._modal_data(2, 6, 1.0, Modality.IMAGE)
        video = self._modal_data(8, 12, 2.0, Modality.VIDEO)
        seq = sess.add_sequence(token_ids, multimodals=dict(video=[video], image=[image]))
        block_trie.match(seq)

        assert len(seq.logical_blocks) == 1
        assert seq.num_history_ids == block_size
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.num_matched == block_size

    def test_prefix_cache_extra_hash_lookup_is_block_indexed(self, scheduler):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [99] * block_size * 4 + [3]
        multimodals = dict(image=[
            self._modal_data(1, block_size + 1, 1.0, Modality.IMAGE),
            self._modal_data(block_size * 2 + 1, block_size * 2 + 4, 2.0, Modality.IMAGE),
            self._modal_data(block_size * 3 + 2, block_size * 3 + 6, 3.0, Modality.IMAGE),
        ])
        seq = sess.add_sequence(token_ids, multimodals=multimodals)

        block0_hashes = seq.get_prefix_cache_extra_hashes(0, block_size)
        block1_hashes = seq.get_prefix_cache_extra_hashes(block_size, block_size * 2)
        block2_hashes = seq.get_prefix_cache_extra_hashes(block_size * 2, block_size * 3)
        block3_hashes = seq.get_prefix_cache_extra_hashes(block_size * 3, block_size * 4)

        assert len(block0_hashes) == 1
        assert block0_hashes == block1_hashes
        assert len(block2_hashes) == 1
        assert len(block3_hashes) == 1
        assert len(seq._prefix_cache_extra_hashes) == 4
        assert seq._num_indexed_prefix_cache_metas == 3

    def test_evict(self, block_trie, scheduler, num_gpu_blocks):
        block_mgr = block_trie.block_manager
        sess = scheduler.add_session(0)
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

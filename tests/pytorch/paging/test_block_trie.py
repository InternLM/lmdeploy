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
    def ssm_cache_config(self, block_size, num_cpu_blocks, num_gpu_blocks, max_batch_size):
        yield CacheConfig(max_batches=max_batch_size,
                          block_size=block_size,
                          num_cpu_blocks=num_cpu_blocks,
                          num_gpu_blocks=num_gpu_blocks,
                          enable_prefix_caching=True,
                          num_state_caches=max_batch_size + 1 + 8,
                          prefix_cache_state_budget=8,
                          states_shapes=[((1, ), torch.float32)])

    @pytest.fixture
    def ssm_scheduler(self, ssm_cache_config, scheduler_config, seq_meta):
        yield Scheduler(scheduler_config=scheduler_config, cache_config=ssm_cache_config, seq_meta=seq_meta)

    @pytest.fixture
    def block_mgr(self, scheduler):
        yield scheduler.block_manager

    @pytest.fixture
    def block_trie(self, scheduler):
        yield scheduler.block_trie

    def _image_multimodals(self,
                           start: int,
                           end: int,
                           value: float,
                           image_token_id: int = 99,
                           content_hash: str | None = None):
        data = torch.full((2, 2), value, dtype=torch.float32)
        return dict(image=[MultiModalData(data=data,
                                          start=start,
                                          end=end,
                                          meta=dict(image_token_id=image_token_id),
                                          content_hash=content_hash)])

    def _modal_data(self, start: int, end: int, value: float, modality: Modality):
        data = torch.full((2, 2), value, dtype=torch.float32)
        return MultiModalData(data=data,
                              start=start,
                              end=end,
                              modality=modality,
                              meta=dict(token_id=int(value)))

    def _multi_image_multimodals(self, spans: list[tuple[int, int, float]]):
        return dict(image=[
            MultiModalData(data=torch.full((2, 2), value, dtype=torch.float32),
                           start=start,
                           end=end,
                           modality=Modality.IMAGE,
                           meta=dict(image_token_id=99)) for start, end, value in spans
        ])

    def _add_ready_ssm_checkpoint(self, scheduler, token_ids):
        seq = scheduler.add_session(len(scheduler.sessions)).add_sequence(token_ids)
        scheduler.block_manager.allocate(seq)
        scheduler.block_trie.allocate(seq)
        state_idx = scheduler.block_trie.reserve_state_checkpoint_for_seq(seq)
        assert state_idx >= 0
        assert scheduler.block_trie.commit_state_checkpoint_for_seq(seq)
        return seq, getattr(seq.logical_blocks, 'last_shared_node'), state_idx

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

    def test_match_multimodal_uses_precomputed_content_hash(self, block_trie, block_mgr, scheduler):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size + [99] * block_size + [2] * block_size + [3]

        seq = sess.add_sequence(
            token_ids,
            multimodals=self._image_multimodals(block_size, block_size * 2, 1.0, content_hash='image-a'),
        )
        block_mgr.allocate(seq)
        block_trie.allocate(seq)

        seq = sess.add_sequence(
            token_ids,
            multimodals=self._image_multimodals(block_size, block_size * 2, 2.0, content_hash='image-a'),
        )
        block_trie.match(seq)

        assert len(seq.logical_blocks) == 3
        assert seq.num_history_ids == block_size * 3
        assert seq.prefix_cache_metas[0].content_hash == 'image-a'

    def test_match_multimodal_different_precomputed_content_hash(self, block_trie, block_mgr, scheduler):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size + [99] * block_size + [2] * block_size + [3]

        seq = sess.add_sequence(
            token_ids,
            multimodals=self._image_multimodals(block_size, block_size * 2, 1.0, content_hash='image-a'),
        )
        block_mgr.allocate(seq)
        block_trie.allocate(seq)

        seq = sess.add_sequence(
            token_ids,
            multimodals=self._image_multimodals(block_size, block_size * 2, 1.0, content_hash='image-b'),
        )
        block_trie.match(seq)

        assert len(seq.logical_blocks) == 1
        assert seq.num_history_ids == block_size
        assert seq.prefix_cache_metas[0].content_hash == 'image-b'

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

    def test_match_multimodal_clamp_keeps_previous_images(self, block_trie, block_mgr, scheduler):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [7] * (block_size * 7 + block_size // 2)
        image1 = (block_size, block_size * 2, 1.0)
        image2 = (block_size * 3, block_size * 4, 2.0)
        image3 = (block_size * 6, block_size * 7 + block_size // 4, 3.0)

        seq = sess.add_sequence(token_ids, multimodals=self._multi_image_multimodals([image1, image2, image3]))
        block_mgr.allocate(seq)
        block_trie.allocate(seq)

        seq = sess.add_sequence(token_ids, multimodals=self._multi_image_multimodals([image1, image2, image3]))
        block_trie.match(seq)
        assert len(seq.logical_blocks) == 6
        assert seq.num_history_ids == block_size * 6
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.num_matched == block_size * 6

        different_last_image = (image3[0], image3[1], 4.0)
        seq = sess.add_sequence(
            token_ids,
            multimodals=self._multi_image_multimodals([image1, image2, different_last_image]),
        )
        block_trie.match(seq)
        assert len(seq.logical_blocks) == 6
        assert seq.num_history_ids == block_size * 6

        different_middle_image = (image2[0], image2[1], 5.0)
        seq = sess.add_sequence(
            token_ids,
            multimodals=self._multi_image_multimodals([image1, different_middle_image, image3]),
        )
        block_trie.match(seq)
        assert len(seq.logical_blocks) == 3
        assert seq.num_history_ids == block_size * 3

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

    def test_match_ssm_requires_ready_state_checkpoint(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size * 2 + [2]

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        node = getattr(seq.logical_blocks, 'last_shared_node')

        seq = sess.add_sequence(token_ids)
        block_trie.match(seq)
        assert len(seq.logical_blocks) == 0
        assert seq.num_history_ids == 0
        assert seq.prefix_cache_restore_state == -1

        state_idx = block_trie.reserve_state_checkpoint(node)
        block_trie.mark_state_checkpoint_ready(node)

        seq = sess.add_sequence(token_ids)
        block_trie.match(seq)
        assert len(seq.logical_blocks) == 2
        assert seq.num_history_ids == block_size * 2
        assert seq.prefix_cache_restore_state == state_idx

    def test_match_ssm_clamps_to_deepest_ready_state_checkpoint(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size * 3 + [2]

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        leaf = getattr(seq.logical_blocks, 'last_shared_node')
        checkpoint_node = leaf.parent
        state_idx = block_trie.reserve_state_checkpoint(checkpoint_node)
        block_trie.mark_state_checkpoint_ready(checkpoint_node)

        seq = sess.add_sequence(token_ids)
        block_trie.match(seq)

        assert len(seq.logical_blocks) == 2
        assert seq.num_history_ids == block_size * 2
        assert seq.prefix_cache_restore_state == state_idx

    def test_match_ssm_sparse_index_misses_without_block_walk(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        num_blocks = 12
        token_ids = []
        for block_id in range(num_blocks):
            token_ids.extend([block_id + 1] * block_size)
        token_ids.append(99)

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        node = getattr(seq.logical_blocks, 'last_shared_node')
        block_trie.reserve_state_checkpoint(node)
        block_trie.mark_state_checkpoint_ready(node)

        miss_token_ids = token_ids.copy()
        miss_token_ids[(num_blocks - 1) * block_size:num_blocks * block_size] = [777] * block_size
        seq = sess.add_sequence(miss_token_ids)
        calls = 0
        get_hashes = seq.get_prefix_cache_extra_hashes

        def count_hashes(start, end):
            nonlocal calls
            calls += 1
            return get_hashes(start, end)

        seq.get_prefix_cache_extra_hashes = count_hashes
        block_trie.match(seq)

        assert len(seq.logical_blocks) == 0
        assert seq.prefix_cache_restore_state == -1
        assert calls == 1

    def test_match_ssm_sparse_index_verifies_hash_collision_exactly(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size + [2] * block_size + [3]

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        node = getattr(seq.logical_blocks, 'last_shared_node')
        block_trie.reserve_state_checkpoint(node)
        block_trie.mark_state_checkpoint_ready(node)

        miss_token_ids = [1] * block_size + [4] * block_size + [3]
        seq = sess.add_sequence(miss_token_ids)
        collision_key = block_trie._make_state_checkpoint_lookup_key(seq, block_size * 2)
        block_trie._state_checkpoint_index.setdefault(collision_key, []).append(node)
        block_trie._state_checkpoint_steps.setdefault(seq.adapter_name, set()).add(block_size * 2)

        block_trie.match(seq)

        assert len(seq.logical_blocks) == 0
        assert seq.prefix_cache_restore_state == -1

    def test_ssm_checkpoint_save_publishes_to_sparse_index(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size * 2

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        state_idx = block_trie.reserve_state_checkpoint_for_seq(seq)
        node = getattr(seq.logical_blocks, 'last_shared_node')

        assert state_idx >= 0
        assert seq.prefix_cache_save_state == state_idx
        assert not node.state_ready

        assert block_trie.commit_state_checkpoint_for_seq(seq)
        assert node.state_ready
        assert seq.prefix_cache_save_state == -1

        seq = sess.add_sequence(token_ids + [2])
        block_trie.match(seq)

        assert seq.num_history_ids == block_size * 2
        assert seq.prefix_cache_restore_state == state_idx

    def test_ssm_checkpoint_ready_index_is_idempotent(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size * 2

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        node = getattr(seq.logical_blocks, 'last_shared_node')
        block_trie.reserve_state_checkpoint(node)

        block_trie.mark_state_checkpoint_ready(node)
        block_trie.mark_state_checkpoint_ready(node)

        key = block_trie._make_state_checkpoint_node_key(node)
        assert block_trie._state_checkpoint_index[key] == [node]
        assert block_trie._state_checkpoint_steps[node.adapter_name] == {node.num_matched}

    def test_ssm_checkpoint_unindex_removes_duplicate_entries(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size * 2

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        node = getattr(seq.logical_blocks, 'last_shared_node')
        block_trie.reserve_state_checkpoint(node)
        block_trie.mark_state_checkpoint_ready(node)
        key = block_trie._make_state_checkpoint_node_key(node)
        block_trie._state_checkpoint_index[key].extend([node, node])

        block_trie.release_state_checkpoint(node)

        assert key not in block_trie._state_checkpoint_index
        assert node.adapter_name not in block_trie._state_checkpoint_steps
        assert node.state_idx == -1
        assert not node.state_ready

    def test_ssm_checkpoint_pending_save_discard_releases_slot(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size * 2

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        free_states = ssm_scheduler.state_manager.get_num_free_checkpoint()
        state_idx = block_trie.reserve_state_checkpoint_for_seq(seq)
        node = getattr(seq.logical_blocks, 'last_shared_node')

        assert state_idx >= 0
        assert node.state_idx == state_idx
        assert not node.state_ready
        assert ssm_scheduler.state_manager.get_num_free_checkpoint() == free_states - 1

        assert block_trie.discard_state_checkpoint_for_seq(seq)
        assert seq.prefix_cache_save_state == -1
        assert seq.prefix_cache_save_step == 0
        assert seq.prefix_cache_save_node is None
        assert node.state_idx == -1
        assert not node.state_ready
        assert ssm_scheduler.state_manager.get_num_free_checkpoint() == free_states

    def test_ssm_checkpoint_commit_failure_discards_pending_slot(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size * 2

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        free_states = ssm_scheduler.state_manager.get_num_free_checkpoint()
        state_idx = block_trie.reserve_state_checkpoint_for_seq(seq)
        node = getattr(seq.logical_blocks, 'last_shared_node')

        assert state_idx >= 0
        seq.logical_blocks.last_shared_node = node.parent

        assert not block_trie.commit_state_checkpoint_for_seq(seq)
        assert seq.prefix_cache_save_state == -1
        assert seq.prefix_cache_save_step == 0
        assert seq.prefix_cache_save_node is None
        assert node.state_idx == -1
        assert ssm_scheduler.state_manager.get_num_free_checkpoint() == free_states

    def test_ssm_checkpoint_save_uses_explicit_chunk_step(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        checkpoint_step = block_size * 2
        token_ids = [1] * block_size * 4 + [2]

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        state_idx = block_trie.reserve_state_checkpoint_for_seq(seq, step=checkpoint_step)

        assert state_idx >= 0
        assert seq.prefix_cache_save_state == state_idx
        assert seq.prefix_cache_save_step == checkpoint_step

        # Long-context chunking advances the sequence step before the executor
        # output is committed. The checkpoint should still attach to the
        # ancestor node for the chunk boundary.
        seq.set_step(checkpoint_step)
        assert block_trie.commit_state_checkpoint_for_seq(seq)

        seq = sess.add_sequence(token_ids[:checkpoint_step] + [3])
        block_trie.match(seq)

        assert seq.num_history_ids == checkpoint_step
        assert seq.prefix_cache_restore_state == state_idx

    def test_ssm_checkpoint_save_skips_partial_tail(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size * 2 + [2]

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)

        assert block_trie.reserve_state_checkpoint_for_seq(seq) == -1
        assert seq.prefix_cache_save_state == -1

    def test_ssm_checkpoint_save_skips_when_no_state_slot(self, ssm_cache_config, scheduler_config, seq_meta):
        cache_config = ssm_cache_config
        cache_config.num_state_caches = 1
        scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)
        block_mgr = scheduler.block_manager
        block_trie = scheduler.block_trie
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size * 2

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)

        assert block_trie.reserve_state_checkpoint_for_seq(seq) == -1
        assert seq.prefix_cache_save_state == -1

    def test_ssm_checkpoint_save_evicts_unpinned_state_only(self, ssm_cache_config, scheduler_config, seq_meta):
        cache_config = ssm_cache_config
        cache_config.prefix_cache_state_budget = 0
        cache_config.num_state_caches = 2
        scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)
        block_size = scheduler.seq_meta.block_size
        token_ids_a = [1] * block_size * 2
        token_ids_b = [2] * block_size * 2

        _, node_a, state_idx_a = self._add_ready_ssm_checkpoint(scheduler, token_ids_a)
        seq_b = scheduler.add_session(99).add_sequence(token_ids_b)
        scheduler.block_manager.allocate(seq_b)
        scheduler.block_trie.allocate(seq_b)
        state_idx_b = scheduler.block_trie.reserve_state_checkpoint_for_seq(seq_b)

        assert state_idx_b >= 0
        assert state_idx_b == state_idx_a
        assert node_a.state_idx == -1
        assert not node_a.state_ready
        assert scheduler.state_manager.get_num_free_checkpoint() == 0

        assert scheduler.block_trie.commit_state_checkpoint_for_seq(seq_b)

        seq_a = scheduler.add_session(100).add_sequence(token_ids_a + [3])
        scheduler.block_trie.match(seq_a)
        assert seq_a.prefix_cache_restore_state == -1

        seq_b = scheduler.add_session(101).add_sequence(token_ids_b + [3])
        scheduler.block_trie.match(seq_b)
        assert seq_b.prefix_cache_restore_state == state_idx_b

    def test_ssm_checkpoint_state_eviction_skips_pinned_restore(self, ssm_cache_config, scheduler_config, seq_meta):
        cache_config = ssm_cache_config
        cache_config.prefix_cache_state_budget = 0
        cache_config.num_state_caches = 3
        scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)
        block_size = scheduler.seq_meta.block_size
        token_ids_a = [1] * block_size * 2
        token_ids_b = [2] * block_size * 2
        token_ids_c = [3] * block_size * 2

        _, node_a, state_idx_a = self._add_ready_ssm_checkpoint(scheduler, token_ids_a)
        _, node_b, state_idx_b = self._add_ready_ssm_checkpoint(scheduler, token_ids_b)

        seq_a = scheduler.add_session(100).add_sequence(token_ids_a + [4])
        scheduler.block_trie.match(seq_a)
        assert seq_a.prefix_cache_restore_state == state_idx_a
        assert scheduler.block_trie.acquire_state_checkpoint_restore_for_seq(seq_a)
        assert node_a.state_ref_count == 1

        seq_c = scheduler.add_session(101).add_sequence(token_ids_c)
        scheduler.block_manager.allocate(seq_c)
        scheduler.block_trie.allocate(seq_c)
        state_idx_c = scheduler.block_trie.reserve_state_checkpoint_for_seq(seq_c)

        assert state_idx_c >= 0
        assert node_a.state_idx == state_idx_a
        assert node_a.state_ready
        assert node_b.state_idx == -1
        assert not node_b.state_ready
        assert state_idx_c == state_idx_b

        assert scheduler.block_trie.release_state_checkpoint_restore_for_seq(seq_a)
        assert node_a.state_ref_count == 0
        assert seq_a.prefix_cache_restore_state == -1

    def test_evict_ssm_releases_state_checkpoint(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size * 2 + [2]

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        node = getattr(seq.logical_blocks, 'last_shared_node')
        block_trie.reserve_state_checkpoint(node)
        block_trie.mark_state_checkpoint_ready(node)
        free_states = ssm_scheduler.state_manager.get_num_free_checkpoint()

        block_mgr.free(seq)
        seq.set_step(0)
        block_trie.evict(1)

        assert ssm_scheduler.state_manager.get_num_free_checkpoint() == free_states + 1

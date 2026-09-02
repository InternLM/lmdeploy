import numpy as np
import pytest

from lmdeploy.pytorch import messages as messages_module
from lmdeploy.pytorch.messages import SamplingParam, UpdateTokenMode
from lmdeploy.pytorch.paging import Scheduler
from lmdeploy.vl.constants import Modality

from ._utils import BlockTrieTestMixin


class TestBlockTrie(BlockTrieTestMixin):

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
        node = seq.prefix_cache.trie_cursor
        assert node is not None
        assert node.prefix_len == block_size * 2
        assert np.array_equal(node.token_ids, [2] * block_size)
        assert np.array_equal(node.parent.token_ids, [1] * block_size)
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
        node = seq.prefix_cache.trie_cursor
        assert node is not None
        assert node.prefix_len == block_size * 3
        expect_tokens = [3] * (block_size // 2) + [4] * (block_size // 2)
        assert np.array_equal(node.token_ids, expect_tokens)
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
        node = seq.prefix_cache.trie_cursor
        assert node is not None
        assert node.prefix_len == block_size
        assert np.array_equal(node.token_ids, [1] * block_size)
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

    def test_logprob_prefix_match_is_capped_at_scoring_start(self, block_trie, block_mgr, scheduler):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = list(range(block_size * 3 + 1))

        cached = sess.add_sequence(token_ids)
        block_mgr.allocate(cached)
        block_trie.allocate(cached)
        cached_blocks = cached.logical_blocks.get_real_blocks().copy()

        param = SamplingParam(num_logprobs=0, logprob_start_len=block_size + 3)
        scored = sess.add_sequence(token_ids, sampling_param=param)
        assert scored.get_prefix_cache_max_candidate_step() == block_size + 3
        assert scored.get_prefix_cache_max_match_step() == block_size

        block_trie.match(scored)

        assert scored.num_history_ids == block_size
        assert scored.prefix_cache.recompute_overlap.fresh_block_range == range(1, 3)

        block_mgr.allocate(scored)
        fresh_blocks = scored.logical_blocks.get_real_blocks()[1:3].copy()
        assert not np.array_equal(fresh_blocks, cached_blocks[1:3])

        block_trie.allocate(scored)

        assert np.array_equal(scored.logical_blocks.get_real_blocks()[1:3], fresh_blocks)
        assert scored.prefix_cache.recompute_overlap.fresh_block_range is None
        assert scored.prefix_cache.trie_cursor.prefix_len == block_size * 3

    def test_logprob_start_minus_one_preserves_prefix_match_limit(self, block_trie, scheduler):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = list(range(block_size * 3 + 1))

        cached = sess.add_sequence(token_ids)
        scheduler.block_manager.allocate(cached)
        block_trie.allocate(cached)

        default_seq = sess.add_sequence(token_ids)
        generated_only = sess.add_sequence(token_ids,
                                           sampling_param=SamplingParam(num_logprobs=0, logprob_start_len=-1))
        assert default_seq.get_prefix_cache_max_match_step() == generated_only.get_prefix_cache_max_match_step()

        block_trie.match(default_seq)
        block_trie.match(generated_only)
        assert default_seq.num_history_ids == generated_only.num_history_ids == block_size * 3

    def test_match_recompute_overlap_stays_private_during_allocate(self, block_trie, block_mgr, scheduler):
        allocator = block_trie.allocator
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size + [2] * block_size + [3] * block_size + [4]

        cached = sess.add_sequence(token_ids)
        block_mgr.allocate(cached)
        block_trie.allocate(cached)
        cached_blocks = cached.logical_blocks.get_real_blocks().copy()

        seq = sess.add_sequence(token_ids)
        seq.prefix_cache.recompute_overlap.recompute_blocks = 1
        block_trie.stats.reset()

        block_trie.match(seq)

        assert seq.num_history_ids == block_size * 2
        assert len(seq.logical_blocks) == 2
        assert seq.prefix_cache.recompute_overlap.fresh_block_range == range(2, 3)
        assert block_trie.stats.num_query_tokens == len(token_ids)
        assert block_trie.stats.num_hit_tokens == block_size * 2

        block_mgr.allocate(seq)
        fresh_overlap_block = seq.logical_blocks[2]
        assert fresh_overlap_block != cached_blocks[2]

        block_trie.allocate(seq)

        assert seq.logical_blocks[2] == fresh_overlap_block
        assert seq.prefix_cache.recompute_overlap.fresh_block_range is None
        assert allocator.get_ref_count(np.array([cached_blocks[2]])).item() == 2
        assert allocator.get_ref_count(np.array([fresh_overlap_block])).item() == 1
        assert seq.prefix_cache.trie_cursor.prefix_len == block_size * 3

    def test_recompute_overlap_cursor_rebuilds_after_eviction(self, block_trie, block_mgr, scheduler):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size + [2] * block_size + [3] * block_size + [4]

        cached = sess.add_sequence(token_ids)
        block_mgr.allocate(cached)
        block_trie.allocate(cached)
        cached_overlap_leaf = cached.prefix_cache.trie_cursor
        block_mgr.free(cached)

        seq = sess.add_sequence(token_ids)
        seq.prefix_cache.recompute_overlap.recompute_blocks = 1
        block_trie.match(seq)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)

        assert seq.prefix_cache.trie_cursor is cached_overlap_leaf
        assert block_trie.evict(1) == 1
        assert cached_overlap_leaf.parent is None

        seq.update_token_ids([5] * (block_size - 1), mode=UpdateTokenMode.INPUTS)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)

        new_node = seq.prefix_cache.trie_cursor
        assert new_node.prefix_len == block_size * 4
        assert new_node.parent is not cached_overlap_leaf
        assert block_trie._cursor_belongs_to_trie(new_node)

    @pytest.mark.parametrize('raw_match_blocks', [1, 2, 5])
    def test_match_recompute_overlap_boundary_cases(self, block_trie, block_mgr, scheduler, raw_match_blocks):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = []
        for block_id in range(raw_match_blocks):
            token_ids.extend([block_id + 1] * block_size)
        token_ids.append(99)

        cached = sess.add_sequence(token_ids)
        block_mgr.allocate(cached)
        block_trie.allocate(cached)
        cached_blocks = cached.logical_blocks.get_real_blocks().copy()

        seq = sess.add_sequence(token_ids)
        seq.prefix_cache.recompute_overlap.recompute_blocks = 1

        block_trie.match(seq)

        expected_history = max(0, raw_match_blocks - 1) * block_size
        expected_raw = raw_match_blocks * block_size
        assert seq.num_history_ids == expected_history
        assert seq.prefix_cache.recompute_overlap.fresh_block_range == range(expected_history // block_size,
                                                                             expected_raw // block_size)

        block_mgr.allocate(seq)
        fresh_overlap_blocks = seq.logical_blocks.get_real_blocks()[expected_history // block_size:raw_match_blocks]
        if len(fresh_overlap_blocks) > 0:
            assert not np.array_equal(fresh_overlap_blocks,
                                      cached_blocks[expected_history // block_size:raw_match_blocks])

        block_trie.allocate(seq)

        assert seq.prefix_cache.recompute_overlap.fresh_block_range is None
        assert seq.prefix_cache.trie_cursor.prefix_len == expected_raw

    def test_match_recompute_disabled_keeps_ar_full_hit(self, block_trie, block_mgr, scheduler):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size + [2] * block_size + [3] * block_size + [4]

        cached = sess.add_sequence(token_ids)
        block_mgr.allocate(cached)
        block_trie.allocate(cached)

        seq = sess.add_sequence(token_ids)
        block_trie.match(seq)

        assert seq.prefix_cache.recompute_overlap.recompute_blocks == 0
        assert seq.num_history_ids == block_size * 3
        assert seq.prefix_cache.recompute_overlap.fresh_block_range is None

    def test_match_recompute_overlap_expands_to_multimodal_boundary(self, block_trie, block_mgr, scheduler):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        image_start = block_size * 2 + block_size // 2
        image_end = block_size * 3 + block_size // 2
        token_ids = [99] * (block_size * 4 + 1)
        multimodals = self._image_multimodals(image_start, image_end, 1.0)

        cached = sess.add_sequence(token_ids, multimodals=multimodals)
        block_mgr.allocate(cached)
        block_trie.allocate(cached)
        cached_blocks = cached.logical_blocks.get_real_blocks().copy()

        seq = sess.add_sequence(token_ids, multimodals=self._image_multimodals(image_start, image_end, 1.0))
        seq.prefix_cache.recompute_overlap.recompute_blocks = 1

        block_trie.match(seq)

        assert seq.num_history_ids == block_size * 2
        assert seq.prefix_cache.recompute_overlap.fresh_block_range == range(2, 4)

        block_mgr.allocate(seq)
        fresh_overlap_blocks = seq.logical_blocks.get_real_blocks()[2:4].copy()
        assert not np.array_equal(fresh_overlap_blocks, cached_blocks[2:4])

        block_trie.allocate(seq)

        assert np.array_equal(seq.logical_blocks.get_real_blocks()[2:4], fresh_overlap_blocks)
        assert seq.prefix_cache.recompute_overlap.fresh_block_range is None
        assert seq.prefix_cache.trie_cursor.prefix_len == block_size * 4

    def test_ssm_match_recompute_overlap_extends_from_checkpoint_to_raw_hit(self, ssm_scheduler):
        block_trie = ssm_scheduler.block_trie
        block_mgr = ssm_scheduler.block_manager
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size

        checkpoint_tokens = [1] * block_size + [2] * block_size
        checkpoint_seq = sess.add_sequence(checkpoint_tokens)
        block_mgr.allocate(checkpoint_seq)
        block_trie.allocate(checkpoint_seq)
        state_idx = block_trie.state_checkpoints.reserve_save(checkpoint_seq)
        assert state_idx >= 0
        assert block_trie.state_checkpoints.publish_save(checkpoint_seq)

        token_ids = checkpoint_tokens + [3] * block_size + [4] * block_size + [5]
        cached = sess.add_sequence(token_ids)
        block_mgr.allocate(cached)
        block_trie.allocate(cached)
        cached_blocks = cached.logical_blocks.get_real_blocks().copy()

        seq = sess.add_sequence(token_ids)
        seq.prefix_cache.recompute_overlap.recompute_blocks = 1

        block_trie.match(seq)

        assert seq.num_history_ids == block_size * 2
        assert seq.prefix_cache.restore.slot == state_idx
        assert seq.prefix_cache.recompute_overlap.fresh_block_range == range(2, 4)

        block_mgr.allocate(seq)
        fresh_overlap_blocks = seq.logical_blocks.get_real_blocks()[2:4].copy()
        assert not np.array_equal(fresh_overlap_blocks, cached_blocks[2:4])

        block_trie.allocate(seq)

        assert np.array_equal(seq.logical_blocks.get_real_blocks()[2:4], fresh_overlap_blocks)
        assert seq.prefix_cache.recompute_overlap.fresh_block_range is None
        assert seq.prefix_cache.trie_cursor.prefix_len == block_size * 4

    def test_logprob_ssm_checkpoint_beyond_scoring_start_is_ignored(self, ssm_scheduler):
        block_trie = ssm_scheduler.block_trie
        block_mgr = ssm_scheduler.block_manager
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [token for token in range(1, 5) for _ in range(block_size)]
        token_ids.append(9)

        cached = sess.add_sequence(token_ids)
        block_mgr.allocate(cached)
        block_trie.allocate(cached)
        deep_state = block_trie.state_checkpoints.reserve_save(cached, step=block_size * 4)
        assert deep_state >= 0
        assert block_trie.state_checkpoints.publish_save(cached)

        param = SamplingParam(num_logprobs=0, logprob_start_len=block_size * 2 + 3)
        scored = sess.add_sequence(token_ids, sampling_param=param)
        assert scored.get_prefix_cache_max_candidate_step() == block_size * 2 + 3

        block_trie.match(scored)

        assert scored.num_history_ids == 0
        assert scored.prefix_cache.restore.slot == -1
        assert scored.prefix_cache.restore.node is None

    def test_ssm_match_recompute_falls_back_for_required_overlap(self, ssm_scheduler):
        block_trie = ssm_scheduler.block_trie
        block_mgr = ssm_scheduler.block_manager
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = ([1] * block_size + [2] * block_size + [3] * block_size + [4] * block_size)

        cached = sess.add_sequence(token_ids)
        block_mgr.allocate(cached)
        block_trie.allocate(cached)
        shallow_step = block_size * 3
        shallow_state = block_trie.state_checkpoints.reserve_save(cached, step=shallow_step)
        assert shallow_state >= 0
        assert block_trie.state_checkpoints.publish_save(cached)
        deep_state = block_trie.state_checkpoints.reserve_save(cached, step=block_size * 4)
        assert deep_state >= 0
        assert block_trie.state_checkpoints.publish_save(cached)

        seq = sess.add_sequence(token_ids + [5])
        seq.prefix_cache.recompute_overlap.recompute_blocks = 1
        block_trie.match(seq)

        assert seq.num_history_ids == shallow_step
        assert seq.prefix_cache.restore.slot == shallow_state
        assert seq.prefix_cache.restore.slot != deep_state
        assert seq.prefix_cache.recompute_overlap.fresh_block_range == range(shallow_step // block_size, 4)

    def test_ssm_match_recompute_misses_without_cached_suffix(self, ssm_scheduler):
        block_trie = ssm_scheduler.block_trie
        block_mgr = ssm_scheduler.block_manager
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        checkpoint_tokens = [1] * block_size + [2] * block_size
        cached_tokens = checkpoint_tokens + [3] * block_size

        cached = sess.add_sequence(cached_tokens)
        block_mgr.allocate(cached)
        block_trie.allocate(cached)
        state_idx = block_trie.state_checkpoints.reserve_save(cached, step=block_size * 2)
        assert state_idx >= 0
        assert block_trie.state_checkpoints.publish_save(cached)
        cached_blocks = cached.logical_blocks.get_real_blocks().copy()
        ref_counts = block_trie.allocator.get_ref_count(cached_blocks).copy()

        seq = sess.add_sequence(checkpoint_tokens + [4] * block_size + [5])
        seq.prefix_cache.recompute_overlap.recompute_blocks = 1
        block_trie.match(seq)

        assert seq.num_history_ids == 0
        assert len(seq.logical_blocks) == 0
        assert seq.prefix_cache.restore.slot == -1
        assert seq.prefix_cache.restore.node is None
        assert seq.prefix_cache.recompute_overlap.fresh_block_range is None
        assert np.array_equal(block_trie.allocator.get_ref_count(cached_blocks), ref_counts)

    @pytest.mark.parametrize('num_new_blocks', [0, 1])
    def test_ssm_checkpoint_after_recompute_overlap_uses_trie_block_map(self, ssm_scheduler, num_new_blocks):
        block_trie = ssm_scheduler.block_trie
        block_mgr = ssm_scheduler.block_manager
        block_size = ssm_scheduler.seq_meta.block_size
        cached_tokens = [token for token in range(1, 5) for _ in range(block_size)]

        cached_session = ssm_scheduler.add_session(0)
        cached = cached_session.add_sequence(cached_tokens)
        block_mgr.allocate(cached)
        block_trie.allocate(cached)
        checkpoint_step = block_size * 2
        assert block_trie.state_checkpoints.reserve_save(cached, step=checkpoint_step) >= 0
        assert block_trie.state_checkpoints.publish_save(cached)
        ssm_scheduler.end_session(cached_session.session_id)

        new_tokens = [5] * block_size * num_new_blocks
        request_tokens = cached_tokens + new_tokens + [9]
        producer_session = ssm_scheduler.add_session(1)
        producer = producer_session.add_sequence(request_tokens)
        producer.prefix_cache.recompute_overlap.recompute_blocks = 1
        block_trie.match(producer)
        block_mgr.allocate(producer)
        block_trie.allocate(producer)

        trie_block_map = producer.prefix_cache.recompute_overlap.trie_block_map
        assert set(trie_block_map) == {2, 3}
        producer_blocks = producer.logical_blocks.get_real_blocks().copy()
        assert np.all(producer_blocks[2:4] != [trie_block_map[2], trie_block_map[3]])

        save_step = block_size * (4 + num_new_blocks)
        assert block_trie.state_checkpoints.reserve_save(producer, step=save_step) >= 0
        save_node = producer.prefix_cache.pending_save.node
        assert block_trie.state_checkpoints.publish_save(producer)
        match_data = save_node.state_checkpoint.exact_match_data
        trie_blocks = np.array([node.block_id for node in save_node.path_from_root()])
        assert np.array_equal(match_data.block_ids, trie_blocks)

        fresh_overlap_blocks = producer_blocks[2:4]
        ssm_scheduler.end_session(producer_session.session_id)
        assert producer.prefix_cache.recompute_overlap.trie_block_map == {}
        assert np.all(block_trie.allocator.get_ref_count(fresh_overlap_blocks) == 0)
        assert np.all(block_trie.allocator.get_ref_count(trie_blocks) > 0)

        matched = ssm_scheduler.add_session(2).add_sequence(request_tokens)
        block_trie.match(matched)
        assert matched.num_history_ids == save_step
        assert np.array_equal(matched.logical_blocks.get_real_blocks(), trie_blocks)

    def test_match_after_sequence_blocks_are_freed(self, block_trie, block_mgr, scheduler):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size + [2] * block_size + [3] * (block_size // 2)
        seq = sess.add_sequence(token_ids)

        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        seq.state.free()

        assert seq.num_history_ids == 0
        assert len(seq.logical_blocks) == 0
        assert seq.prefix_cache.trie_cursor is None

        block_trie.match(seq)

        assert seq.num_history_ids == block_size * 2
        assert len(seq.logical_blocks) == 2
        node = seq.prefix_cache.trie_cursor
        assert node is not None
        assert node.prefix_len == block_size * 2

    def test_match_replays_cached_routed_experts(self, block_trie, block_mgr, scheduler):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size + [2] * block_size + [3]
        sampling_param = SamplingParam(return_routed_experts=True)
        seq = sess.add_sequence(token_ids, sampling_param=sampling_param)
        experts = self._routed_experts(block_size * 2)

        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        seq.append_routed_experts(experts)
        block_trie.cache_routed_experts_for_seq(seq)

        matched = sess.add_sequence(token_ids, sampling_param=sampling_param)
        block_trie.match(matched)

        assert matched.num_history_ids == block_size * 2
        assert np.array_equal(matched.all_routed_experts.get_real(), experts)

    def test_match_skips_routed_expert_replay_when_not_requested(self, block_trie, block_mgr, scheduler):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size + [2] * block_size + [3]
        seq = sess.add_sequence(token_ids, sampling_param=SamplingParam(return_routed_experts=True))

        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        seq.append_routed_experts(self._routed_experts(block_size * 2))
        block_trie.cache_routed_experts_for_seq(seq)

        matched = sess.add_sequence(token_ids)
        block_trie.match(matched)

        assert matched.num_history_ids == block_size * 2
        assert len(matched.all_routed_experts) == 0

    def test_existing_node_can_be_enriched_with_routed_experts(self, block_trie, block_mgr, scheduler):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size + [2] * block_size + [3]

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        node = seq.prefix_cache.trie_cursor
        assert node is not None
        assert node.routed_experts is None

        expert_seq = sess.add_sequence(token_ids, sampling_param=SamplingParam(return_routed_experts=True))
        experts = self._routed_experts(block_size * 2)
        expert_seq.append_routed_experts(experts)
        block_mgr.allocate(expert_seq)
        block_trie.allocate(expert_seq)

        assert node.routed_experts is not None
        matched = sess.add_sequence(token_ids, sampling_param=SamplingParam(return_routed_experts=True))
        block_trie.match(matched)
        assert np.array_equal(matched.all_routed_experts.get_real(), experts)

    def test_match_stops_before_block_missing_routed_experts(self, block_trie, block_mgr, scheduler):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size + [2] * block_size + [3]
        seq = sess.add_sequence(token_ids, sampling_param=SamplingParam(return_routed_experts=True))

        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        seq.append_routed_experts(self._routed_experts(block_size))
        block_trie.cache_routed_experts_for_seq(seq)

        matched = sess.add_sequence(token_ids, sampling_param=SamplingParam(return_routed_experts=True))
        block_trie.match(matched)

        assert matched.num_history_ids == block_size
        assert np.array_equal(matched.all_routed_experts.get_real(), self._routed_experts(block_size))
        assert matched.prefix_cache.recompute_overlap.fresh_block_range == range(1, 2)

    def test_missing_replay_does_not_enrich_from_misaligned_tail(self, block_trie, block_mgr, scheduler):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size + [2] * block_size + [3]
        seq = sess.add_sequence(token_ids)

        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        last_node = seq.prefix_cache.trie_cursor
        assert last_node is not None
        assert last_node.routed_experts is None

        matched = sess.add_sequence(token_ids, sampling_param=SamplingParam(return_routed_experts=True))
        block_trie.match(matched)

        assert matched.num_history_ids == 0
        assert len(matched.all_routed_experts) == 0

        matched.append_routed_experts(self._routed_experts(1, offset=1000))
        block_trie.cache_routed_experts_for_seq(matched)

        assert last_node.routed_experts is None

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
        node = seq.prefix_cache.trie_cursor
        assert node is not None
        assert node.prefix_len == block_size * 3

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
        node = seq.prefix_cache.trie_cursor
        assert node is not None
        assert node.prefix_len == block_size

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
        assert seq.prefix_cache.multimodal_spans[0].content_hash == 'image-a'

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
        assert seq.prefix_cache.multimodal_spans[0].content_hash == 'image-b'

    def test_multimodal_prefix_cache_meta_skips_hash_when_prefix_cache_disabled(self, cache_config, scheduler_config,
                                                                                seq_meta, monkeypatch):
        cache_config.enable_prefix_caching = False
        scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)

        def _fail_hash(*args, **kwargs):
            raise AssertionError('disabled prefix cache should not hash multimodal payloads')

        monkeypatch.setattr(messages_module, 'make_multimodal_content_hash', _fail_hash)

        sess = scheduler.add_session(0)
        seq = sess.add_sequence([99] * sess.seq_meta.block_size,
                                multimodals=self._image_multimodals(0, sess.seq_meta.block_size, 1.0))

        assert seq.prefix_cache.multimodal_spans == []
        assert not seq.history_multimodals.empty()

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
        node = seq.prefix_cache.trie_cursor
        assert node is not None
        assert node.prefix_len == 0

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
        node = seq.prefix_cache.trie_cursor
        assert node is not None
        assert node.prefix_len == block_size * 6

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

    def test_match_multimodal_clamp_rechecks_after_block_rounding(self, block_trie, block_mgr, scheduler):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [99] * (block_size * 7 + block_size // 2)
        image1 = (block_size // 2, block_size * 5 + block_size // 2, 1.0)
        image2 = (block_size * 5 + block_size // 2 + 2, block_size * 7 + block_size // 2, 2.0)

        seq = sess.add_sequence(token_ids, multimodals=self._multi_image_multimodals([image1, image2]))
        block_mgr.allocate(seq)
        block_trie.allocate(seq)

        seq = sess.add_sequence(token_ids, multimodals=self._multi_image_multimodals([image1, image2]))
        block_trie.match(seq)

        assert len(seq.logical_blocks) == 0
        assert seq.num_history_ids == 0
        node = seq.prefix_cache.trie_cursor
        assert node is not None
        assert node.prefix_len == 0

    def test_match_multimodal_identity_order_is_canonical(self, block_trie, block_mgr, scheduler):
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
        node = seq.prefix_cache.trie_cursor
        assert node is not None
        assert node.prefix_len == block_size

    def test_prefix_cache_extra_identity_lookup_is_block_indexed(self, scheduler):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [99] * block_size * 4 + [3]
        multimodals = dict(image=[
            self._modal_data(1, block_size + 1, 1.0, Modality.IMAGE),
            self._modal_data(block_size * 2 + 1, block_size * 2 + 4, 2.0, Modality.IMAGE),
            self._modal_data(block_size * 3 + 2, block_size * 3 + 6, 3.0, Modality.IMAGE),
        ])
        seq = sess.add_sequence(token_ids, multimodals=multimodals)

        block0_identity = seq.get_prefix_cache_extra_identity(0, block_size)
        block1_identity = seq.get_prefix_cache_extra_identity(block_size, block_size * 2)
        block2_identity = seq.get_prefix_cache_extra_identity(block_size * 2, block_size * 3)
        block3_identity = seq.get_prefix_cache_extra_identity(block_size * 3, block_size * 4)

        assert len(block0_identity) == 1
        assert block0_identity == block1_identity
        assert block0_identity[0] is seq.prefix_cache.multimodal_spans[0]
        assert len(block2_identity) == 1
        assert block2_identity[0] is seq.prefix_cache.multimodal_spans[1]
        assert len(block3_identity) == 1
        assert block3_identity[0] is seq.prefix_cache.multimodal_spans[2]
        assert len(seq.prefix_cache.block_extra_identity) == 4
        assert seq.prefix_cache.num_indexed_spans == 3

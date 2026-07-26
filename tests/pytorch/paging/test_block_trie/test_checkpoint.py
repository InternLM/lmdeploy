import numpy as np

from lmdeploy.pytorch.messages import SamplingParam

from ._utils import BlockTrieTestMixin


class TestStateCheckpointMatching(BlockTrieTestMixin):

    def test_match_ssm_requires_published_state_checkpoint(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size * 2 + [2]

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        node = seq.prefix_cache.last_shared_node

        seq = sess.add_sequence(token_ids)
        block_trie.match(seq)
        assert len(seq.logical_blocks) == 0
        assert seq.num_history_ids == 0
        assert seq.prefix_cache.restore.slot == -1

        state_idx = block_trie.state_checkpoints._reserve_node(node)
        block_trie.state_checkpoints._publish_node(node)

        seq = sess.add_sequence(token_ids)
        block_trie.match(seq)
        assert len(seq.logical_blocks) == 2
        assert seq.num_history_ids == block_size * 2
        assert seq.prefix_cache.restore.slot == state_idx

    def test_match_ssm_clamps_to_deepest_published_state_checkpoint(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size * 3 + [2]

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        leaf = seq.prefix_cache.last_shared_node
        checkpoint_node = leaf.parent
        state_idx = block_trie.state_checkpoints._reserve_node(checkpoint_node)
        block_trie.state_checkpoints._publish_node(checkpoint_node)

        seq = sess.add_sequence(token_ids)
        block_trie.match(seq)

        assert len(seq.logical_blocks) == 2
        assert seq.num_history_ids == block_size * 2
        assert seq.prefix_cache.restore.slot == state_idx

    def test_match_ssm_replays_cached_routed_experts(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size + [2] * block_size
        sampling_param = SamplingParam(return_routed_experts=True)
        seq = sess.add_sequence(token_ids, sampling_param=sampling_param)
        experts = self._routed_experts(block_size * 2)

        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        seq.append_routed_experts(experts)
        block_trie.cache_routed_experts_for_seq(seq)
        state_idx = block_trie.state_checkpoints.reserve_save(seq)
        assert state_idx >= 0
        assert block_trie.state_checkpoints.publish_save(seq)

        matched = sess.add_sequence(token_ids + [3], sampling_param=sampling_param)
        block_trie.match(matched)

        assert matched.prefix_cache.restore.slot == state_idx
        assert matched.num_history_ids == block_size * 2
        assert np.array_equal(matched.all_routed_experts.get_real(), experts)

    def test_match_ssm_misses_when_checkpoint_lacks_routed_experts(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size + [2] * block_size

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        state_idx = block_trie.reserve_state_checkpoint_for_seq(seq)
        assert state_idx >= 0
        assert block_trie.commit_state_checkpoint_for_seq(seq)

        sampling_param = SamplingParam(return_routed_experts=True)
        matched = sess.add_sequence(token_ids + [3], sampling_param=sampling_param)
        block_trie.match(matched)

        assert matched.num_history_ids == 0
        assert matched.prefix_cache.restore.slot == -1
        assert len(matched.all_routed_experts) == 0

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
        node = seq.prefix_cache.last_shared_node
        block_trie.state_checkpoints._reserve_node(node)
        block_trie.state_checkpoints._publish_node(node)

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
        assert seq.prefix_cache.restore.slot == -1
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
        node = seq.prefix_cache.last_shared_node
        block_trie.state_checkpoints._reserve_node(node)
        block_trie.state_checkpoints._publish_node(node)

        miss_token_ids = [1] * block_size + [4] * block_size + [3]
        seq = sess.add_sequence(miss_token_ids)
        collision_key = block_trie._checkpoint_index.make_request_key(seq, block_size * 2)
        block_trie._checkpoint_index._buckets.setdefault(collision_key, []).append(node)
        block_trie._checkpoint_index._steps_by_adapter.setdefault(seq.adapter_name, set()).add(block_size * 2)

        block_trie.match(seq)

        assert len(seq.logical_blocks) == 0
        assert seq.prefix_cache.restore.slot == -1

    def test_match_ssm_sparse_bucket_continues_after_candidate_mismatch(self, ssm_scheduler):
        block_size = ssm_scheduler.seq_meta.block_size
        bad_tokens = [9] * block_size + [2] * block_size
        good_tokens = [1] * block_size + [2] * block_size
        _, bad_node, _ = self._add_published_ssm_checkpoint(ssm_scheduler, bad_tokens)
        _, good_node, good_state = self._add_published_ssm_checkpoint(ssm_scheduler, good_tokens)
        block_trie = ssm_scheduler.block_trie
        key = block_trie._checkpoint_index.make_node_key(good_node)

        assert block_trie._checkpoint_index.make_node_key(bad_node) == key
        assert block_trie._checkpoint_index._buckets[key] == [bad_node, good_node]

        seq = ssm_scheduler.add_session(100).add_sequence(good_tokens + [3])
        block_trie.match(seq)

        assert seq.num_history_ids == block_size * 2
        assert seq.prefix_cache.restore.slot == good_state
        assert seq.prefix_cache.restore.node is good_node

    def test_match_ssm_from_checkpoint_cursor_appends_only_suffix(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size + [2] * block_size + [3] * block_size

        cached = sess.add_sequence(token_ids)
        block_mgr.allocate(cached)
        block_trie.allocate(cached)
        shallow_state = block_trie.state_checkpoints.reserve_save(cached, step=block_size)
        assert shallow_state >= 0
        assert block_trie.state_checkpoints.publish_save(cached)

        seq = sess.add_sequence(token_ids + [4])
        block_trie.match(seq)
        shallow_blocks = seq.logical_blocks.get_real_blocks().copy()
        assert len(shallow_blocks) == 1
        assert seq.prefix_cache.restore.slot == shallow_state

        deep_state = block_trie.state_checkpoints.reserve_save(cached, step=block_size * 3)
        assert deep_state >= 0
        assert block_trie.state_checkpoints.publish_save(cached)
        block_trie.match(seq)

        assert seq.num_history_ids == block_size * 3
        assert seq.prefix_cache.restore.slot == deep_state
        assert np.array_equal(seq.logical_blocks.get_real_blocks()[:1], shallow_blocks)
        assert len(seq.logical_blocks) == 3

    def test_match_ssm_exact_metadata_verifies_multimodal_identity(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size + [99] * block_size + [2] * block_size
        image_start = block_size
        image_end = block_size * 2

        checkpoint_seq = sess.add_sequence(
            token_ids,
            multimodals=self._image_multimodals(image_start, image_end, 1.0, content_hash='image-a'),
        )
        block_mgr.allocate(checkpoint_seq)
        block_trie.allocate(checkpoint_seq)
        state_idx = block_trie.state_checkpoints.reserve_save(checkpoint_seq)
        assert state_idx >= 0
        assert block_trie.state_checkpoints.publish_save(checkpoint_seq)

        matched = sess.add_sequence(
            token_ids + [3],
            multimodals=self._image_multimodals(image_start, image_end, 2.0, content_hash='image-a'),
        )
        block_trie.match(matched)
        assert matched.num_history_ids == block_size * 3
        assert matched.prefix_cache.restore.slot == state_idx

        mismatched = sess.add_sequence(
            token_ids + [3],
            multimodals=self._image_multimodals(image_start, image_end, 1.0, content_hash='image-b'),
        )
        block_trie.match(mismatched)
        assert mismatched.num_history_ids == 0
        assert mismatched.prefix_cache.restore.slot == -1

    def test_match_ssm_keeps_request_mismatch_checkpoint_candidate(self, ssm_scheduler):
        block_size = ssm_scheduler.seq_meta.block_size
        token_ids = [1] * block_size + [2] * block_size
        _, node, state_idx = self._add_published_ssm_checkpoint(ssm_scheduler, token_ids)
        key = ssm_scheduler.block_trie._checkpoint_index.make_node_key(node)
        free_states = ssm_scheduler.state_manager.get_num_free_checkpoint()

        # Same indexed step and last block, but a different earlier block. This
        # is a miss for this request, not proof that the cached checkpoint is
        # stale globally.
        miss_token_ids = [4] * block_size + [2] * block_size + [3]
        seq = ssm_scheduler.add_session(100).add_sequence(miss_token_ids)
        ssm_scheduler.block_trie.match(seq)

        assert len(seq.logical_blocks) == 0
        assert seq.prefix_cache.restore.slot == -1
        assert node.state_checkpoint.slot == state_idx
        assert node.state_checkpoint.published
        assert node in ssm_scheduler.block_trie._checkpoint_index._buckets[key]
        assert ssm_scheduler.state_manager.get_num_free_checkpoint() == free_states

    def test_match_ssm_drops_stale_sparse_index_entry_only(self, ssm_scheduler):
        block_size = ssm_scheduler.seq_meta.block_size
        token_ids = [1] * block_size + [2] * block_size
        _, node, state_idx = self._add_published_ssm_checkpoint(ssm_scheduler, token_ids)
        block_trie = ssm_scheduler.block_trie
        canonical_key = block_trie._checkpoint_index.make_node_key(node)

        miss_token_ids = [1] * block_size + [4] * block_size + [3]
        seq = ssm_scheduler.add_session(100).add_sequence(miss_token_ids)
        stale_key = block_trie._checkpoint_index.make_request_key(seq, block_size * 2)
        assert stale_key != canonical_key
        block_trie._checkpoint_index._buckets.setdefault(stale_key, []).append(node)
        block_trie._checkpoint_index._steps_by_adapter.setdefault(seq.adapter_name, set()).add(block_size * 2)
        free_states = ssm_scheduler.state_manager.get_num_free_checkpoint()

        block_trie.match(seq)

        assert len(seq.logical_blocks) == 0
        assert seq.prefix_cache.restore.slot == -1
        assert stale_key not in block_trie._checkpoint_index._buckets
        assert node in block_trie._checkpoint_index._buckets[canonical_key]
        assert block_trie._checkpoint_index._steps_by_adapter[node.adapter_name] == {node.num_matched}
        assert node.state_checkpoint.slot == state_idx
        assert node.state_checkpoint.published
        assert ssm_scheduler.state_manager.get_num_free_checkpoint() == free_states

    def test_match_ssm_releases_detached_stale_checkpoint_candidate(self, ssm_scheduler):
        block_size = ssm_scheduler.seq_meta.block_size
        token_ids = [1] * block_size + [2] * block_size
        _, node, _ = self._add_published_ssm_checkpoint(ssm_scheduler, token_ids)
        block_trie = ssm_scheduler.block_trie
        key = block_trie._checkpoint_index.make_node_key(node)
        free_states = ssm_scheduler.state_manager.get_num_free_checkpoint()

        node.parent = None
        seq = ssm_scheduler.add_session(100).add_sequence(token_ids + [3])
        block_trie.match(seq)

        assert len(seq.logical_blocks) == 0
        assert seq.prefix_cache.restore.slot == -1
        assert key not in block_trie._checkpoint_index._buckets
        assert node.adapter_name not in block_trie._checkpoint_index._steps_by_adapter
        assert node.state_checkpoint is None
        assert ssm_scheduler.state_manager.get_num_free_checkpoint() == free_states + 1

    def test_match_ssm_releases_checkpoint_with_detached_ancestor(self, ssm_scheduler):
        block_size = ssm_scheduler.seq_meta.block_size
        token_ids = [1] * block_size + [2] * block_size + [3] * block_size
        _, node, _ = self._add_published_ssm_checkpoint(ssm_scheduler, token_ids)
        block_trie = ssm_scheduler.block_trie
        key = block_trie._checkpoint_index.make_node_key(node)
        free_states = ssm_scheduler.state_manager.get_num_free_checkpoint()

        ancestor = node.parent
        ancestor.parent = None
        assert node.parent is ancestor
        assert node.state_checkpoint.exact_match_data is None

        seq = ssm_scheduler.add_session(100).add_sequence(token_ids + [4])
        block_trie.match(seq)

        assert len(seq.logical_blocks) == 0
        assert seq.prefix_cache.restore.slot == -1
        assert key not in block_trie._checkpoint_index._buckets
        assert node.state_checkpoint is None
        assert ssm_scheduler.state_manager.get_num_free_checkpoint() == free_states + 1

    def test_replacing_child_detaches_displaced_node(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size + [2] * block_size

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        displaced = seq.prefix_cache.last_shared_node
        parent = displaced.parent
        replacement = type(displaced)(hash_key=displaced.hash_key,
                                       block=displaced.block,
                                       tokens=displaced.tokens.copy(),
                                       num_matched=displaced.num_matched,
                                       extra_hashes=displaced.extra_hashes,
                                       adapter_name=displaced.adapter_name)
        replacement.parent = parent

        assert displaced.parent is None
        assert parent.children[replacement.hash_key] is replacement
        assert block_trie.state_checkpoints._reserve_node(replacement) >= 0
        block_trie.state_checkpoints._publish_node(replacement)
        match_data = replacement.state_checkpoint.exact_match_data

        displaced.parent = None
        assert parent.children[replacement.hash_key] is replacement
        assert replacement.state_checkpoint.exact_match_data is match_data

        matched = sess.add_sequence(token_ids + [3])
        block_trie.match(matched)
        assert matched.num_history_ids == block_size * 2
        assert matched.prefix_cache.restore.node is replacement

    def test_match_ssm_keeps_pinned_stale_checkpoint_candidate(self, ssm_scheduler):
        block_size = ssm_scheduler.seq_meta.block_size
        token_ids = [1] * block_size + [2] * block_size
        _, node, state_idx = self._add_published_ssm_checkpoint(ssm_scheduler, token_ids)
        block_trie = ssm_scheduler.block_trie
        key = block_trie._checkpoint_index.make_node_key(node)
        free_states = ssm_scheduler.state_manager.get_num_free_checkpoint()

        node.state_checkpoint.pin_count = 1
        node.parent = None
        seq = ssm_scheduler.add_session(100).add_sequence(token_ids + [3])
        block_trie.match(seq)

        assert len(seq.logical_blocks) == 0
        assert seq.prefix_cache.restore.slot == -1
        assert node.state_checkpoint.slot == state_idx
        assert node.state_checkpoint.published
        assert node.state_checkpoint.pin_count == 1
        assert node in block_trie._checkpoint_index._buckets[key]
        assert ssm_scheduler.state_manager.get_num_free_checkpoint() == free_states

    def test_match_ssm_releases_unpublished_indexed_checkpoint_candidate(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size + [2] * block_size

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        node = seq.prefix_cache.last_shared_node
        state_idx = block_trie.state_checkpoints._reserve_node(node)
        assert state_idx >= 0
        assert not node.state_checkpoint.published
        key = block_trie._checkpoint_index.make_node_key(node)
        block_trie._checkpoint_index._buckets.setdefault(key, []).append(node)
        block_trie._checkpoint_index._steps_by_adapter.setdefault(node.adapter_name, set()).add(node.num_matched)
        free_states = ssm_scheduler.state_manager.get_num_free_checkpoint()

        seq = sess.add_sequence(token_ids + [3])
        block_trie.match(seq)

        assert len(seq.logical_blocks) == 0
        assert seq.prefix_cache.restore.slot == -1
        assert key not in block_trie._checkpoint_index._buckets
        assert node.adapter_name not in block_trie._checkpoint_index._steps_by_adapter
        assert node.state_checkpoint is None
        assert ssm_scheduler.state_manager.get_num_free_checkpoint() == free_states + 1

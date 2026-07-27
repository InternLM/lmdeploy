import numpy as np
import pytest

from lmdeploy.pytorch.messages import UpdateTokenMode
from lmdeploy.pytorch.paging import Scheduler

from ._utils import BlockTrieTestMixin


class TestStateCheckpointLifecycle(BlockTrieTestMixin):

    def test_ssm_checkpoint_index_rejects_unpublished_node(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size * 2

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        node = seq.prefix_cache.last_shared_node
        assert block_trie.state_checkpoints._reserve_node(node) >= 0

        with pytest.raises(RuntimeError, match='unpublished SSM prefix-cache checkpoint'):
            block_trie.state_checkpoints._index_checkpoint(node)

    def test_ssm_checkpoint_save_publishes_to_sparse_index(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size * 2

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        state_idx = block_trie.state_checkpoints.reserve_save(seq)
        node = seq.prefix_cache.last_shared_node

        assert state_idx >= 0
        assert seq.prefix_cache.pending_save.slot == state_idx
        assert not node.state_checkpoint.published
        assert node.state_checkpoint.exact_match_data is None

        assert block_trie.state_checkpoints.publish_save(seq)
        assert node.state_checkpoint.published
        assert seq.prefix_cache.pending_save.slot == -1
        match_data = node.state_checkpoint.exact_match_data
        assert match_data is not None
        assert np.array_equal(match_data.token_ids, token_ids)
        assert not np.shares_memory(match_data.token_ids, seq.history_cache.get_real())
        assert not match_data.token_ids.flags.writeable
        assert not match_data.blocks.flags.writeable
        expected_blocks = seq.logical_blocks.get_real_blocks()[:len(token_ids) // block_size]
        assert np.array_equal(match_data.blocks, expected_blocks)
        assert not np.shares_memory(match_data.blocks, expected_blocks)

        seq = sess.add_sequence(token_ids + [2])
        block_trie.match(seq)

        assert seq.num_history_ids == block_size * 2
        assert seq.prefix_cache.restore.slot == state_idx

    def test_ssm_restore_pin_survives_tail_allocation(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        block_size = ssm_scheduler.seq_meta.block_size
        checkpoint_tokens = [1] * block_size * 2
        suffix_tokens = [2] * block_size * 3 + [3]

        _, checkpoint_node, state_idx = self._add_published_ssm_checkpoint(ssm_scheduler, checkpoint_tokens)

        seq = ssm_scheduler.add_session(100).add_sequence(checkpoint_tokens + suffix_tokens)
        block_trie.match(seq)
        assert seq.num_history_ids == block_size * 2
        assert seq.prefix_cache.restore.slot == state_idx
        assert seq.prefix_cache.restore.node is checkpoint_node

        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        assert seq.prefix_cache.last_shared_node is not checkpoint_node
        assert seq.prefix_cache.restore.node is checkpoint_node

        assert block_trie.state_checkpoints.pin_restore(seq)
        assert checkpoint_node.state_checkpoint.pin_count == 1
        assert block_trie.state_checkpoints.unpin_restore(seq)
        assert checkpoint_node.state_checkpoint.pin_count == 0
        assert seq.prefix_cache.restore.node is None

    def test_ssm_checkpoint_release_rejects_pinned_state(self, ssm_scheduler):
        block_trie = ssm_scheduler.block_trie
        block_size = ssm_scheduler.seq_meta.block_size
        token_ids = [1] * block_size * 2

        _, checkpoint_node, state_idx = self._add_published_ssm_checkpoint(ssm_scheduler, token_ids)

        seq = ssm_scheduler.add_session(100).add_sequence(token_ids + [2])
        block_trie.match(seq)
        assert seq.prefix_cache.restore.slot == state_idx
        assert block_trie.state_checkpoints.pin_restore(seq)

        with pytest.raises(RuntimeError, match='Cannot release a pinned'):
            block_trie.state_checkpoints.release_checkpoint(checkpoint_node)

        assert block_trie.state_checkpoints.unpin_restore(seq)
        block_trie.state_checkpoints.release_checkpoint(checkpoint_node)

    def test_ssm_checkpoint_restore_unpin_detects_lost_pin(self, ssm_scheduler):
        block_trie = ssm_scheduler.block_trie
        block_size = ssm_scheduler.seq_meta.block_size
        token_ids = [1] * block_size * 2

        _, checkpoint_node, state_idx = self._add_published_ssm_checkpoint(ssm_scheduler, token_ids)

        seq = ssm_scheduler.add_session(100).add_sequence(token_ids + [2])
        block_trie.match(seq)
        assert seq.prefix_cache.restore.slot == state_idx
        assert block_trie.state_checkpoints.pin_restore(seq)
        checkpoint_node.state_checkpoint.pin_count = 0

        with pytest.raises(RuntimeError, match='lost its node reference'):
            block_trie.state_checkpoints.unpin_restore(seq)

        checkpoint_node.state_checkpoint.pin_count = 1
        assert block_trie.state_checkpoints.unpin_restore(seq)

    def test_ssm_checkpoint_published_index_is_idempotent(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size * 2

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        node = seq.prefix_cache.last_shared_node
        block_trie.state_checkpoints._reserve_node(node)

        block_trie.state_checkpoints._publish_node(node, seq)
        match_data = node.state_checkpoint.exact_match_data
        block_trie.state_checkpoints._publish_node(node, seq)

        key = block_trie._checkpoint_index.make_node_key(node)
        assert node.state_checkpoint.exact_match_data is match_data
        assert block_trie._checkpoint_index._buckets[key] == [node]
        assert block_trie._checkpoint_index._steps_by_adapter[node.adapter_name] == {node.num_matched}

    def test_ssm_checkpoint_unindex_removes_duplicate_entries(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size * 2

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        node = seq.prefix_cache.last_shared_node
        block_trie.state_checkpoints._reserve_node(node)
        block_trie.state_checkpoints._publish_node(node, seq)
        key = block_trie._checkpoint_index.make_node_key(node)
        block_trie._checkpoint_index._buckets[key].extend([node, node])

        block_trie.state_checkpoints.release_checkpoint(node)

        assert key not in block_trie._checkpoint_index._buckets
        assert node.adapter_name not in block_trie._checkpoint_index._steps_by_adapter
        assert node.state_checkpoint is None

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
        state_idx = block_trie.state_checkpoints.reserve_save(seq)
        node = seq.prefix_cache.last_shared_node

        assert state_idx >= 0
        assert node.state_checkpoint.slot == state_idx
        assert not node.state_checkpoint.published
        assert ssm_scheduler.state_manager.get_num_free_checkpoint() == free_states - 1

        assert block_trie.state_checkpoints.discard_save(seq)
        assert seq.prefix_cache.pending_save.slot == -1
        assert seq.prefix_cache.pending_save.step == 0
        assert seq.prefix_cache.pending_save.node is None
        assert node.state_checkpoint is None
        assert ssm_scheduler.state_manager.get_num_free_checkpoint() == free_states

    def test_ssm_checkpoint_publish_allows_sequence_to_advance_past_save_step(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        save_step = block_size * 2
        token_ids = [1] * save_step

        seq = sess.add_sequence(token_ids)
        seq.set_step(save_step - 1)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        state_idx = block_trie.state_checkpoints.reserve_decode_save(seq, interval=block_size)
        node = seq.prefix_cache.pending_save.node

        assert state_idx >= 0
        seq.update_token_ids([2], mode=UpdateTokenMode.DECODE)

        assert block_trie.state_checkpoints.publish_save(seq)
        assert seq.prefix_cache.decode_checkpoint_node is node
        assert node.state_checkpoint.published

    def test_ssm_checkpoint_publish_failure_discards_detached_pending_slot(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size * 2

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        free_states = ssm_scheduler.state_manager.get_num_free_checkpoint()
        state_idx = block_trie.state_checkpoints.reserve_save(seq)
        node = seq.prefix_cache.last_shared_node

        assert state_idx >= 0
        node.detach_leaf()

        assert not block_trie.state_checkpoints.publish_save(seq)
        assert seq.prefix_cache.pending_save.slot == -1
        assert seq.prefix_cache.pending_save.step == 0
        assert seq.prefix_cache.pending_save.node is None
        assert node.state_checkpoint is None
        assert ssm_scheduler.state_manager.get_num_free_checkpoint() == free_states

    def test_ssm_checkpoint_publish_rejects_mismatched_sequence_cursor(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size * 2

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        free_states = ssm_scheduler.state_manager.get_num_free_checkpoint()
        state_idx = block_trie.state_checkpoints.reserve_save(seq)
        node = seq.prefix_cache.last_shared_node

        assert state_idx >= 0
        seq.history_cache = seq.history_cache.clone()
        assert not np.shares_memory(node.tokens, seq.history_cache.get_real())
        seq.history_cache[block_size * 2 - 1] = 9

        with pytest.raises(RuntimeError, match='mismatched sequence cursor'):
            block_trie.state_checkpoints.publish_save(seq)

        assert seq.prefix_cache.pending_save.slot == -1
        assert node.state_checkpoint is None
        assert ssm_scheduler.state_manager.get_num_free_checkpoint() == free_states

    def test_ssm_checkpoint_publish_index_error_rolls_back_publication(self, ssm_scheduler, monkeypatch):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size * 2

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        free_states = ssm_scheduler.state_manager.get_num_free_checkpoint()
        state_idx = block_trie.state_checkpoints.reserve_save(seq)
        node = seq.prefix_cache.last_shared_node
        key = block_trie._checkpoint_index.make_node_key(node)
        checkpoint_lifecycle = block_trie.state_checkpoints
        index_checkpoint = checkpoint_lifecycle._index_checkpoint

        def fail_after_index(checkpoint_node):
            index_checkpoint(checkpoint_node)
            raise RuntimeError('injected index failure')

        monkeypatch.setattr(checkpoint_lifecycle, '_index_checkpoint', fail_after_index)
        with pytest.raises(RuntimeError, match='injected index failure'):
            block_trie.state_checkpoints.publish_save(seq)

        assert state_idx >= 0
        assert seq.prefix_cache.pending_save.slot == -1
        assert seq.prefix_cache.pending_save.step == 0
        assert seq.prefix_cache.pending_save.node is None
        assert node.state_checkpoint is None
        assert key not in block_trie._checkpoint_index._buckets
        assert node.adapter_name not in block_trie._checkpoint_index._steps_by_adapter
        assert ssm_scheduler.state_manager.get_num_free_checkpoint() == free_states

    def test_ssm_decode_checkpoint_replaces_previous_unpinned_state(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        step = block_size * 2

        seq = sess.add_sequence([1] * step)
        seq.set_step(step - 1)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        state_idx_a = block_trie.state_checkpoints.reserve_decode_save(seq, interval=block_size)
        node_a = seq.prefix_cache.pending_save.node

        assert state_idx_a >= 0
        assert seq.prefix_cache.pending_save.is_decode
        assert block_trie.state_checkpoints.publish_save(seq)
        assert seq.prefix_cache.decode_checkpoint_node is node_a
        assert node_a.state_checkpoint.published

        seq.update_token_ids([2] * block_size, mode=UpdateTokenMode.DECODE)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        state_idx_b = block_trie.state_checkpoints.reserve_decode_save(seq, interval=block_size)
        node_b = seq.prefix_cache.pending_save.node

        assert state_idx_b >= 0
        assert node_a.state_checkpoint is None
        assert seq.prefix_cache.decode_checkpoint_node is None
        assert block_trie.state_checkpoints.publish_save(seq)
        assert seq.prefix_cache.decode_checkpoint_node is node_b
        assert node_b.state_checkpoint.published

    def test_ssm_decode_checkpoint_skip_replacement_when_previous_is_pinned(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        step = block_size * 2

        seq = sess.add_sequence([1] * step)
        seq.set_step(step - 1)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        state_idx = block_trie.state_checkpoints.reserve_decode_save(seq, interval=block_size)
        node = seq.prefix_cache.pending_save.node
        assert state_idx >= 0
        assert block_trie.state_checkpoints.publish_save(seq)
        node.state_checkpoint.pin_count = 1

        seq.update_token_ids([2] * block_size, mode=UpdateTokenMode.DECODE)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)

        assert block_trie.state_checkpoints.reserve_decode_save(seq, interval=block_size) == -1
        assert seq.prefix_cache.decode_checkpoint_node is node
        assert node.state_checkpoint.slot == state_idx
        assert node.state_checkpoint.published
        assert seq.prefix_cache.pending_save.slot == -1

    def test_ssm_decode_checkpoint_keeps_old_state_when_new_node_is_pending(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        step = block_size * 2
        old_tokens = [1] * step
        new_tokens = [2] * block_size

        seq = sess.add_sequence(old_tokens)
        seq.set_step(step - 1)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        old_state_idx = block_trie.state_checkpoints.reserve_decode_save(seq, interval=block_size)
        old_node = seq.prefix_cache.pending_save.node

        assert old_state_idx >= 0
        assert block_trie.state_checkpoints.publish_save(seq)
        assert seq.prefix_cache.decode_checkpoint_node is old_node
        assert old_node.state_checkpoint.published

        seq.update_token_ids(new_tokens, mode=UpdateTokenMode.DECODE)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        new_node = seq.prefix_cache.last_shared_node

        pending_seq = sess.add_sequence(old_tokens + new_tokens)
        block_mgr.allocate(pending_seq)
        block_trie.allocate(pending_seq)
        assert pending_seq.prefix_cache.last_shared_node is new_node
        pending_state_idx = block_trie.state_checkpoints.reserve_save(pending_seq)
        assert pending_state_idx >= 0
        assert new_node.state_checkpoint.slot == pending_state_idx
        assert not new_node.state_checkpoint.published

        assert block_trie.state_checkpoints.reserve_decode_save(seq, interval=block_size) == -1
        assert seq.prefix_cache.decode_checkpoint_node is old_node
        assert old_node.state_checkpoint.slot == old_state_idx
        assert old_node.state_checkpoint.published
        assert seq.prefix_cache.pending_save.slot == -1

    def test_ssm_decode_checkpoint_keeps_old_state_when_new_step_is_not_allocated(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        step = block_size * 2

        seq = sess.add_sequence([1] * step)
        seq.set_step(step - 1)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        state_idx = block_trie.state_checkpoints.reserve_decode_save(seq, interval=block_size)
        node = seq.prefix_cache.pending_save.node
        assert state_idx >= 0
        assert block_trie.state_checkpoints.publish_save(seq)

        seq.update_token_ids([2] * block_size, mode=UpdateTokenMode.DECODE)
        block_mgr.allocate(seq)

        assert block_trie.state_checkpoints.reserve_decode_save(seq, interval=block_size) == -1
        assert seq.prefix_cache.decode_checkpoint_node is node
        assert node.state_checkpoint.slot == state_idx
        assert node.state_checkpoint.published

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
        state_idx = block_trie.state_checkpoints.reserve_save(seq, step=checkpoint_step)

        assert state_idx >= 0
        assert seq.prefix_cache.pending_save.slot == state_idx
        assert seq.prefix_cache.pending_save.step == checkpoint_step

        # Long-context chunking advances the sequence step before the executor
        # output is completed. The checkpoint should still attach to the
        # ancestor node for the chunk boundary.
        seq.set_step(checkpoint_step)
        assert block_trie.state_checkpoints.publish_save(seq)

        seq = sess.add_sequence(token_ids[:checkpoint_step] + [3])
        block_trie.match(seq)

        assert seq.num_history_ids == checkpoint_step
        assert seq.prefix_cache.restore.slot == state_idx

    def test_ssm_checkpoint_save_skips_partial_tail(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size * 2 + [2]

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)

        assert block_trie.state_checkpoints.reserve_save(seq) == -1
        assert seq.prefix_cache.pending_save.slot == -1

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

        assert block_trie.state_checkpoints.reserve_save(seq) == -1
        assert seq.prefix_cache.pending_save.slot == -1

    def test_ssm_checkpoint_save_skips_duplicate_unpublished_node(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        block_size = ssm_scheduler.seq_meta.block_size
        token_ids = [1] * block_size * 2

        seq_a = ssm_scheduler.add_session(0).add_sequence(token_ids)
        block_mgr.allocate(seq_a)
        block_trie.allocate(seq_a)

        seq_b = ssm_scheduler.add_session(1).add_sequence(token_ids)
        block_mgr.allocate(seq_b)
        block_trie.allocate(seq_b)

        node = seq_a.prefix_cache.last_shared_node
        assert node is seq_b.prefix_cache.last_shared_node

        state_idx_a = block_trie.state_checkpoints.reserve_save(seq_a)
        state_idx_b = block_trie.state_checkpoints.reserve_save(seq_b)

        assert state_idx_a >= 0
        assert state_idx_b == -1
        assert node.state_checkpoint.slot == state_idx_a
        assert not node.state_checkpoint.published
        assert seq_a.prefix_cache.pending_save.slot == state_idx_a
        assert seq_a.prefix_cache.pending_save.node is node
        assert seq_b.prefix_cache.pending_save.slot == -1
        assert seq_b.prefix_cache.pending_save.node is None

        assert block_trie.state_checkpoints.publish_save(seq_a)
        matched = ssm_scheduler.add_session(2).add_sequence(token_ids + [2])
        block_trie.match(matched)
        assert matched.prefix_cache.restore.slot == state_idx_a
        assert matched.num_history_ids == block_size * 2

    def test_ssm_checkpoint_save_evicts_unpinned_state_only(self, ssm_cache_config, scheduler_config, seq_meta):
        cache_config = ssm_cache_config
        cache_config.prefix_cache_state_budget = 0
        cache_config.num_state_caches = 2
        scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)
        block_size = scheduler.seq_meta.block_size
        token_ids_a = [1] * block_size * 2
        token_ids_b = [2] * block_size * 2

        _, node_a, state_idx_a = self._add_published_ssm_checkpoint(scheduler, token_ids_a)
        seq_b = scheduler.add_session(99).add_sequence(token_ids_b)
        scheduler.block_manager.allocate(seq_b)
        scheduler.block_trie.allocate(seq_b)
        state_idx_b = scheduler.block_trie.state_checkpoints.reserve_save(seq_b)

        assert state_idx_b >= 0
        assert state_idx_b == state_idx_a
        assert node_a.state_checkpoint is None
        assert scheduler.state_manager.get_num_free_checkpoint() == 0

        assert scheduler.block_trie.state_checkpoints.publish_save(seq_b)

        seq_a = scheduler.add_session(100).add_sequence(token_ids_a + [3])
        scheduler.block_trie.match(seq_a)
        assert seq_a.prefix_cache.restore.slot == -1

        seq_b = scheduler.add_session(101).add_sequence(token_ids_b + [3])
        scheduler.block_trie.match(seq_b)
        assert seq_b.prefix_cache.restore.slot == state_idx_b

    def test_ssm_checkpoint_save_producer_pin_blocks_eviction_until_unpin(self, ssm_cache_config, scheduler_config,
                                                                          seq_meta):
        cache_config = ssm_cache_config
        cache_config.prefix_cache_state_budget = 0
        cache_config.num_state_caches = 2
        scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)
        block_size = scheduler.seq_meta.block_size
        token_ids_a = [1] * block_size * 2
        token_ids_b = [2] * block_size * 2

        seq_a = scheduler.add_session(0).add_sequence(token_ids_a)
        scheduler.block_manager.allocate(seq_a)
        scheduler.block_trie.allocate(seq_a)
        state_idx_a = scheduler.block_trie.state_checkpoints.reserve_save(seq_a)
        node_a = seq_a.prefix_cache.pending_save.node

        assert state_idx_a >= 0
        assert scheduler.block_trie.state_checkpoints.publish_save(seq_a, pin_save=True)
        assert node_a.state_checkpoint.published
        assert node_a.state_checkpoint.pin_count == 1

        matched = scheduler.add_session(1).add_sequence(token_ids_a + [3])
        scheduler.block_trie.match(matched)
        assert matched.prefix_cache.restore.slot == state_idx_a

        seq_b = scheduler.add_session(2).add_sequence(token_ids_b)
        scheduler.block_manager.allocate(seq_b)
        scheduler.block_trie.allocate(seq_b)

        assert scheduler.block_trie.state_checkpoints.reserve_save(seq_b) == -1
        assert node_a.state_checkpoint.slot == state_idx_a
        assert node_a.state_checkpoint.published
        assert node_a.state_checkpoint.pin_count == 1

        assert scheduler.block_trie.state_checkpoints.unpin_save(seq_a)
        assert node_a.state_checkpoint.pin_count == 0

        state_idx_b = scheduler.block_trie.state_checkpoints.reserve_save(seq_b)
        assert state_idx_b == state_idx_a
        assert node_a.state_checkpoint is None

    def test_ssm_checkpoint_state_eviction_skips_pinned_restore(self, ssm_cache_config, scheduler_config, seq_meta):
        cache_config = ssm_cache_config
        cache_config.prefix_cache_state_budget = 0
        cache_config.num_state_caches = 3
        scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)
        block_size = scheduler.seq_meta.block_size
        token_ids_a = [1] * block_size * 2
        token_ids_b = [2] * block_size * 2
        token_ids_c = [3] * block_size * 2

        _, node_a, state_idx_a = self._add_published_ssm_checkpoint(scheduler, token_ids_a)
        _, node_b, state_idx_b = self._add_published_ssm_checkpoint(scheduler, token_ids_b)

        seq_a = scheduler.add_session(100).add_sequence(token_ids_a + [4])
        scheduler.block_trie.match(seq_a)
        assert seq_a.prefix_cache.restore.slot == state_idx_a
        assert scheduler.block_trie.state_checkpoints.pin_restore(seq_a)
        assert node_a.state_checkpoint.pin_count == 1

        seq_c = scheduler.add_session(101).add_sequence(token_ids_c)
        scheduler.block_manager.allocate(seq_c)
        scheduler.block_trie.allocate(seq_c)
        state_idx_c = scheduler.block_trie.state_checkpoints.reserve_save(seq_c)

        assert state_idx_c >= 0
        assert node_a.state_checkpoint.slot == state_idx_a
        assert node_a.state_checkpoint.published
        assert node_b.state_checkpoint is None
        assert state_idx_c == state_idx_b

        assert scheduler.block_trie.state_checkpoints.unpin_restore(seq_a)
        assert node_a.state_checkpoint.pin_count == 0
        assert seq_a.prefix_cache.restore.slot == -1

    def test_evict_ssm_releases_state_checkpoint(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size * 2 + [2]

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        node = seq.prefix_cache.last_shared_node
        block_trie.state_checkpoints._reserve_node(node)
        block_trie.state_checkpoints._publish_node(node, seq)
        assert node.state_checkpoint.exact_match_data is not None
        free_states = ssm_scheduler.state_manager.get_num_free_checkpoint()

        block_mgr.free(seq)
        seq.set_step(0)
        block_trie.evict(1)

        assert ssm_scheduler.state_manager.get_num_free_checkpoint() == free_states + 1
        assert node.state_checkpoint is None

    def test_evict_ssm_skips_pinned_state_checkpoint(self, ssm_scheduler):
        block_mgr = ssm_scheduler.block_manager
        block_trie = ssm_scheduler.block_trie
        sess = ssm_scheduler.add_session(0)
        block_size = sess.seq_meta.block_size
        token_ids = [1] * block_size * 2 + [2]

        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        node = seq.prefix_cache.last_shared_node
        assert block_trie.state_checkpoints.reserve_save(seq, step=block_size * 2) >= 0
        assert block_trie.state_checkpoints.publish_save(seq, pin_save=True)
        assert node.state_checkpoint.published
        assert node.state_checkpoint.pin_count == 1
        free_states = ssm_scheduler.state_manager.get_num_free_checkpoint()

        block_mgr.free(seq)
        seq.set_step(0)

        assert block_trie.evict(1) == 0
        assert node in block_trie.leaves
        assert node.state_checkpoint.published
        assert node.state_checkpoint.pin_count == 1
        assert ssm_scheduler.state_manager.get_num_free_checkpoint() == free_states

        assert block_trie.state_checkpoints.unpin_save(seq)
        assert block_trie.evict(1) == 1
        assert ssm_scheduler.state_manager.get_num_free_checkpoint() == free_states + 1

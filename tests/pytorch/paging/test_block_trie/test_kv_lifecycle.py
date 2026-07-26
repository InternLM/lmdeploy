def test_evict(block_trie, scheduler, num_gpu_blocks):
    block_mgr = scheduler.block_manager
    sess = scheduler.add_session(0)
    block_size = sess.seq_meta.block_size
    token_ids = ([1] * block_size * (num_gpu_blocks - 1))
    token_ids += [2] * (block_size // 2)
    seq = sess.add_sequence(token_ids)
    block_mgr.allocate(seq)
    block_trie.allocate(seq)
    assert block_mgr.get_num_free_gpu_blocks() == 0

    # Release the sequence refs, leaving the trie as sole block owner.
    block_mgr.free(seq)
    seq.set_step(0)
    assert block_mgr.get_num_free_gpu_blocks() == 1

    leaf = next(iter(block_trie.leaves))
    block_trie.evict(4)
    new_leaf = next(iter(block_trie.leaves))
    assert leaf != new_leaf
    assert block_mgr.get_num_free_gpu_blocks() == 5


def test_evict_prunes_stale_non_leaf_entry(block_trie, scheduler, num_gpu_blocks):
    block_mgr = scheduler.block_manager
    allocator = block_trie.allocator
    sess = scheduler.add_session(0)
    block_size = sess.seq_meta.block_size
    token_ids = [1] * block_size + [2] * block_size
    seq = sess.add_sequence(token_ids)

    block_mgr.allocate(seq)
    block_trie.allocate(seq)
    leaf = seq.prefix_cache.last_shared_node
    parent = leaf.parent
    assert parent.parent is not None
    assert leaf in block_trie.leaves
    assert parent not in block_trie.leaves

    block_trie.leaves.add(parent)
    allocator._log_mem.access_time[leaf.block] = 0
    allocator._log_mem.access_time[parent.block] = 1
    block_mgr.free(seq)
    seq.set_step(0)

    assert block_trie.evict(2) == 2
    assert len(block_trie.leaves) == 0
    assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks

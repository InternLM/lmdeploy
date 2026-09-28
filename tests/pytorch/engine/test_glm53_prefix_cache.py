# Copyright (c) OpenMMLab. All rights reserved.
"""GLM KPool owner, aligned checkpoint production and unsupported-mode gates."""
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.hf_configs.configuration_glm5_next import Glm5NextConfig
from lmdeploy.pytorch.backends.default.cache import DefaultCacheBackend
from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.configurations.glm5_next import Glm5NextModelConfigBuilder, update_cache_config
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.engine.cache_engine.layout import CacheAllocation, CachePool
from lmdeploy.pytorch.engine.cache_engine.schema import BlockCacheGeometry, BlockCacheRequestContext
from lmdeploy.pytorch.engine.executor.base import ExecutorBase
from lmdeploy.pytorch.engine.inputs_maker import LongContextChunker
from lmdeploy.pytorch.messages import SequenceMeta
from lmdeploy.pytorch.nn.kpool import KPoolIndexer, kpool_pooled_write_locations
from lmdeploy.pytorch.paging import Scheduler
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy


def _cache(**kwargs):
    return CacheConfig(max_batches=2, block_size=64, num_cpu_blocks=0, num_gpu_blocks=64, **kwargs)


def _scheduler(length_limit=8192, prefix=True, budget=4):
    cache = _cache(enable_prefix_caching=prefix, max_prefill_token_num=length_limit,
                   states_shapes=[((1,), torch.float32)], num_state_caches=8,
                   prefix_cache_state_budget=budget)
    update_cache_config(cache)
    return Scheduler(SchedulerConfig(max_batches=2, max_session_len=16384), cache,
                     SequenceMeta(64, strategy=ARSequenceStrategy()))


@pytest.mark.parametrize('prefix', [False, True])
def test_geometry_normalizes_target_and_draft_before_allocation(prefix):
    cache = _cache(enable_prefix_caching=prefix)
    draft = _cache()
    executor = ExecutorBase.__new__(ExecutorBase)
    executor.model_config = SimpleNamespace(update_cache_config_func=update_cache_config)
    executor.cache_config = cache
    executor.specdecode_config = SimpleNamespace(cache_config=draft)
    executor._adjust_block_size()
    executor._sync_spec_cache_block_size()
    assert (cache.block_size, cache.kernel_block_size) == (64, 64)
    assert (draft.block_size, draft.kernel_block_size) == (64, 64)
    assert executor.model_config.block_size == 64
    indexer = KPoolIndexer.__new__(KPoolIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.index_kpool, indexer.head_dim = 4, 128
    requests = indexer.get_block_cache_requests(BlockCacheRequestContext(BlockCacheGeometry(64, 64)))
    assert len(requests) == 1
    with pytest.raises(ValueError, match='logical block_size=64'):
        indexer.get_block_cache_requests(BlockCacheRequestContext(BlockCacheGeometry(256, 64)))


@pytest.mark.parametrize('draft,depth', [(False, 0), (False, 2), (True, 2)])
def test_config_enables_ar_and_mtp_prefix(monkeypatch, draft, depth):
    monkeypatch.setattr('lmdeploy.pytorch.configurations.deepseek_v2.flash_mla_available', lambda: True)
    hf = Glm5NextConfig()
    model = Glm5NextModelConfigBuilder.build(
        hf, tp=1, device_type='cpu', is_draft_model=draft, num_spec_tokens=depth,
        spec_method='deepseek_mtp' if depth else None)
    cache = _cache()
    model.update_cache_config_func(cache)
    assert cache.prefix_cache_token_lookahead == int(depth > 0)
    assert model.prefix_caching_unsupported_reason is None
    if draft:
        assert not model.state_cache_specs  # PR's pageable raw-token tail, not a second state pool.


def test_geometry_rejects_pd_and_invalid_decode_checkpoint_interval():
    with pytest.raises(ValueError, match='PD migration'):
        update_cache_config(_cache(role=EngineRole.Prefill))
    cache = _cache()
    cache.prefix_cache_decode_state_interval = 32
    with pytest.raises(ValueError, match='multiple of 64'):
        update_cache_config(cache)


def test_pooled_suffix_is_private_and_full_owner_copy_includes_named_cache():
    groups = torch.arange(32)
    a = kpool_pooled_write_locations(torch.tensor([0, 1]), groups, 4, page_size=16)
    b = kpool_pooled_write_locations(torch.tensor([0, 2]), groups, 4, page_size=16)
    assert torch.equal(a[:16], b[:16])
    assert not torch.isin(a[16:], b[16:]).any()
    mla = torch.arange(4 * 64).reshape(4, 64).float()
    pooled = torch.arange(4 * 16).reshape(4, 16).clone()
    allocation = CacheAllocation(pools=(CachePool(mla, 0), CachePool(pooled, 0)),
                                 tensor_views=(mla, pooled))
    copy = DefaultCacheBackend.build_block_copy(allocation, 4, 1)
    expected = [x[:1].clone() for x in (mla, pooled)]
    copy.copy(torch.tensor([0]), torch.tensor([3]))
    for x, ref in zip((mla, pooled), expected):
        assert torch.equal(x[3:4], ref)


@pytest.mark.parametrize('length,limit', [(255, 8192), (256, 8192), (257, 8192),
                                         (511, 8192), (512, 8192), (513, 300), (1301, 500)])
def test_scheduler_and_chunker_capture_reusable_exact_boundary(length, limit):
    scheduler = _scheduler(limit)
    seq = scheduler.add_session(0).add_sequence(list(range(length)))
    chunker = LongContextChunker(limit)
    restore_step = (length - 1) // 64 * 64
    cuts = []
    chunker.set_seq(seq)
    while True:
        chunk, _ = chunker.next_chunk_size()
        scheduled_end = scheduler._prefill_scheduler._next_long_context_chunk_end(seq)
        assert scheduled_end == seq.num_history_ids + chunk
        cuts.append(scheduled_end)
        if chunker.is_last_chunk():
            break
        chunker.update_step(SimpleNamespace(is_chunk=True, max_q_seqlen=chunk))
    assert cuts[-1] == length
    if restore_step:
        assert restore_step in cuts[:-1]
    assert all(a < b for a, b in zip([0] + cuts, cuts))


@pytest.mark.parametrize('common', [64, 128, 192, 256])
def test_aligned_state_restore_and_divergent_suffix_ownership(common):
    scheduler = _scheduler()
    trie, manager = scheduler.block_trie, scheduler.block_manager
    producer = scheduler.add_session(0).add_sequence(list(range(256)))
    manager.allocate(producer)
    trie.allocate(producer)
    assert trie.state_checkpoints.reserve_save(producer, step=191) == -1
    slot = trie.state_checkpoints.reserve_save(producer, step=256)
    assert slot >= 0
    assert trie.state_checkpoints.publish_save(producer)
    refs = []
    for request, suffix in [(1, 9000), (2, 9001)]:
        seq = scheduler.add_session(request).add_sequence(list(range(common)) + [suffix] * 300)
        trie.match(seq)
        assert seq.num_history_ids == (256 if common == 256 else 0)
        manager.allocate(seq)
        refs.append(seq.logical_blocks.get_real_blocks().copy())
    if common == 256:
        assert refs[0][0] == refs[1][0]
        assert refs[0][4] != refs[1][4]
    else:
        assert refs[0][0] != refs[1][0]


def test_disabled_prefix_keeps_original_chunking():
    scheduler = _scheduler(prefix=False)
    seq = scheduler.add_session(0).add_sequence(list(range(513)))
    chunker = LongContextChunker(8192)
    assert not chunker.is_long_context(seq)
    assert seq._seq_meta.prefix_cache_checkpoint_block_size == 0


@pytest.mark.parametrize('exhausted', [False, True])
def test_zero_reserved_budget_borrows_idle_slots_or_skips_save(exhausted):
    scheduler = _scheduler(budget=0)
    trie, states = scheduler.block_trie, scheduler.state_manager
    seq = scheduler.add_session(0).add_sequence(list(range(257)))
    scheduler.block_manager.allocate(seq)
    trie.allocate(seq)
    states.allocate(seq)
    allocated = []
    if exhausted:
        while states.get_num_free_runtime():
            allocated.append(states.allocate_state())
    slot = trie.state_checkpoints.reserve_save(seq, step=256)
    assert (slot < 0) == exhausted
    if not exhausted:
        assert trie.state_checkpoints.publish_save(seq)
        assert trie.state_checkpoints.evict(1) == 1
        assert states.get_num_allocated_checkpoint_states() == 0
        consumer = scheduler.add_session(1).add_sequence(list(range(257)))
        trie.match(consumer)
        assert consumer.num_history_ids == 0
    for state in allocated:
        states.free_state(state)
    states.free(seq)


@pytest.mark.parametrize('checkpoint', [False, True])
def test_recompute_does_not_deduplicate_writable_suffix_into_trie(checkpoint):
    scheduler = _scheduler()
    trie, manager = scheduler.block_trie, scheduler.block_manager
    tokens = list(range(512)) + [1]
    producer = scheduler.add_session(0).add_sequence(tokens)
    manager.allocate(producer)
    trie.allocate(producer)
    if checkpoint:
        assert trie.state_checkpoints.reserve_save(producer, step=256) >= 0
        assert trie.state_checkpoints.publish_save(producer)
    seq = scheduler.add_session(1).add_sequence(tokens)
    trie.match(seq)
    assert seq.num_history_ids == (256 if checkpoint else 0)
    manager.allocate(seq)
    before = seq.logical_blocks.get_real_blocks().copy()
    trie.allocate(seq)
    assert (seq.logical_blocks.get_real_blocks() == before).all()
    start = 4 if checkpoint else 0
    assert all(seq.logical_blocks[i] != producer.logical_blocks[i] for i in range(start, 8))


@pytest.mark.parametrize('different_lookahead', [False, True])
def test_mtp_checkpoint_exact_lookahead_identity(different_lookahead):
    from lmdeploy.pytorch.strategies.ar_spec.sequence import ARSpecSequenceStrategy
    scheduler = _scheduler()
    scheduler.seq_meta.prefix_cache_token_lookahead = 1
    scheduler.seq_meta.strategy = ARSpecSequenceStrategy()
    trie = scheduler.block_trie
    # Force collisions: exact verification must cover x[256], not just hash.
    trie._hash_block = lambda tokens, extras: 7
    trie._checkpoint_index._hash_block = trie._hash_block
    tokens = list(range(256)) + [300, 301]
    producer = scheduler.add_session(0).add_sequence(tokens)
    assert producer.prefix_cache.recompute_overlap.recompute_blocks == 0
    scheduler.block_manager.allocate(producer)
    trie.allocate(producer)
    assert trie.state_checkpoints.reserve_save(producer, step=256) >= 0
    assert trie.state_checkpoints.publish_save(producer)
    other = tokens.copy()
    if different_lookahead:
        other[256] = 9000
    consumer = scheduler.add_session(1).add_sequence(other)
    trie.match(consumer)
    assert consumer.num_history_ids == (0 if different_lookahead else 256)


def test_mtp_prompt_end_and_scoring_never_publish_incomplete_draft():
    from lmdeploy.pytorch.messages import SamplingParam
    scheduler = _scheduler()
    scheduler.seq_meta.prefix_cache_token_lookahead = 1
    trie = scheduler.block_trie
    seq = scheduler.add_session(0).add_sequence(list(range(512)))
    scheduler.block_manager.allocate(seq)
    trie.allocate(seq)
    assert seq.prefix_cache.trie_cursor.prefix_len == 448
    assert trie.state_checkpoints.reserve_save(seq, step=512) < 0
    assert trie.state_checkpoints.reserve_save(seq, step=256, is_decode=True) < 0
    scoring = scheduler.add_session(1).add_sequence(
        list(range(512)), sampling_param=SamplingParam(num_logprobs=1, logprob_start_len=0))
    scheduler.block_manager.allocate(scoring)
    trie.allocate(scoring)
    assert scoring.prefix_cache.trie_cursor.prefix_len == 0
    assert trie.state_checkpoints.reserve_save(scoring, step=256) < 0


@pytest.mark.parametrize('boundary', [64, 128, 192, 256, 320, 512])
@pytest.mark.parametrize('mtp', [False, True])
def test_small_owner_exact_checkpoint_and_lookahead(boundary, mtp):
    scheduler = _scheduler()
    scheduler.seq_meta.prefix_cache_token_lookahead = int(mtp)
    trie, manager = scheduler.block_trie, scheduler.block_manager
    tokens = list(range(boundary + 2))
    producer = scheduler.add_session(0).add_sequence(tokens)
    manager.allocate(producer)
    trie.allocate(producer)
    assert trie.state_checkpoints.reserve_save(producer, step=boundary) >= 0
    assert trie.state_checkpoints.publish_save(producer)
    for sid, change_next in [(1, False), (2, True)]:
        other = tokens.copy()
        other[boundary + int(not change_next)] = 9999
        consumer = scheduler.add_session(sid).add_sequence(other)
        trie.match(consumer)
        assert consumer.num_history_ids == (0 if mtp and change_next else boundary)
        manager.allocate(consumer)
        if consumer.num_history_ids:
            assert consumer.logical_blocks[boundary//64-1] == producer.logical_blocks[boundary//64-1]
        assert consumer.logical_blocks[boundary//64] != producer.logical_blocks[boundary//64]

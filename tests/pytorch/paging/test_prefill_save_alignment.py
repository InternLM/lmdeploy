import pytest
import torch

from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.long_context import get_long_context_chunk_limit, plan_long_context_chunk
from lmdeploy.pytorch.messages import SequenceMeta
from lmdeploy.pytorch.paging import Scheduler
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy


def _make_scheduler(alignment=512, budget=2048, lookahead=0):
    cache_config = CacheConfig(max_batches=4,
                               block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=1024,
                               max_prefill_token_num=budget,
                               enable_prefix_caching=True,
                               prefix_cache_block_aligned=True,
                               prefill_save_alignment=alignment,
                               prefix_cache_token_lookahead=lookahead,
                               num_state_caches=21,
                               prefix_cache_state_budget=16,
                               states_shapes=[((1, ), torch.float32)])
    scheduler_config = SchedulerConfig(max_batches=4,
                                       max_session_len=65536,
                                       max_request_output_len=128,
                                       eviction_type='recompute')
    return Scheduler(scheduler_config, cache_config, SequenceMeta(64, strategy=ARSequenceStrategy()))


def _allocate_sequence(scheduler, length, session_id=0):
    seq = scheduler.add_session(session_id).add_sequence([session_id + 1] * length)
    scheduler.block_manager.allocate(seq)
    scheduler.block_trie.allocate(seq)
    scheduler.state_manager.allocate(seq)
    return seq


@pytest.mark.parametrize(('length', 'alignment', 'budget', 'expected_chunks', 'expected_saves'), [
    (65, 64, 2048, [64, 1], [64]),
    (65, 512, 2048, [65], []),
    (128, 512, 2048, [128], []),
    (129, 64, 2048, [128, 1], [128]),
    (129, 512, 2048, [129], []),
    (512, 512, 2048, [512], [512]),
    (513, 512, 2048, [512, 1], [512]),
    (576, 512, 2048, [512, 64], [512]),
    (1024, 512, 2048, [512, 512], [512, 1024]),
    (1024, 1024, 2048, [1024], [1024]),
    (1088, 512, 2048, [1024, 64], [1024]),
    (3000, 64, 2048, [2048, 896, 56], [2048, 2944]),
    (3000, 512, 2048, [2048, 512, 440], [2048, 2560]),
    (3000, 8192, 2048, [2048, 952], []),
    (3000, 8192, 8192, [3000], []),
    (4096, 512, 2048, [2048, 1536, 512], [2048, 3584, 4096]),
    (16384, 8192, 16384, [8192, 8192], [8192, 16384]),
    (16385, 8192, 32768, [16384, 1], [16384]),
    (16384, 64, 16384, [16320, 64], [16320, 16384]),
])
def test_prefill_saves_only_at_configured_alignment(length, alignment, budget, expected_chunks, expected_saves):
    scheduler = _make_scheduler(alignment, budget)
    seq = scheduler.add_session(0).add_sequence(list(range(length)))
    assert scheduler.schedule(is_prefill=True).running == [seq]
    chunks = []
    saved_steps = []
    while seq.num_history_ids < length:
        plan = plan_long_context_chunk(seq, get_long_context_chunk_limit(seq, budget))
        assert 0 < plan.chunk_size <= budget
        if chunks:
            assert scheduler.reserve_long_context_chunk(seq, plan.chunk_size, is_last_chunk=plan.is_last_chunk)
        else:
            assert (seq.kv_token_limit or seq.num_valid_ids) == plan.chunk_end
        copy_plan = scheduler.state_checkpoints.reserve_prefill_save_batch(
            [seq], None if plan.is_last_chunk else (plan.chunk_end, ))
        pending = seq.prefix_cache.pending_save
        if plan.chunk_end % alignment == 0:
            assert pending.is_pending
            assert pending.step == plan.chunk_end
            assert copy_plan.state_pairs == ((seq.logical_state, pending.slot), )
            saved_steps.append(pending.step)
            assert scheduler.state_checkpoints.publish_save(seq)
        else:
            assert not pending.is_pending
            assert not copy_plan.state_pairs
            assert not copy_plan.kv_block_pairs
        chunks.append(plan.chunk_size)
        seq.set_step(plan.chunk_end)
    assert chunks == expected_chunks
    assert saved_steps == expected_saves
    published = sorted(node.state_checkpoint.step for node in scheduler.block_trie._checkpoint_index.unique_nodes()
                       if node.state_checkpoint.published)
    assert published == expected_saves


@pytest.mark.parametrize('explicit_steps', [False, True])
def test_prefill_alignment_filters_mixed_batch(explicit_steps):
    scheduler = _make_scheduler()
    seqs = [_allocate_sequence(scheduler, length, session_id)
            for session_id, length in enumerate((128, 512, 576))]
    steps = tuple(seq.num_valid_ids for seq in seqs) if explicit_steps else None
    copy_plan = scheduler.state_checkpoints.reserve_prefill_save_batch(seqs, steps)
    assert copy_plan.state_pairs == ((seqs[1].logical_state, seqs[1].prefix_cache.pending_save.slot), )
    assert not copy_plan.kv_block_pairs
    assert not seqs[0].prefix_cache.pending_save.is_pending
    assert seqs[1].prefix_cache.pending_save.step == 512
    assert not seqs[2].prefix_cache.pending_save.is_pending


def test_prefill_alignment_rejection_does_not_allocate_or_evict(monkeypatch):
    scheduler = _make_scheduler()
    producer = _allocate_sequence(scheduler, 512)
    scheduler.state_checkpoints.reserve_prefill_save_batch([producer])
    node = producer.prefix_cache.pending_save.node
    assert scheduler.state_checkpoints.publish_save(producer)
    seq = _allocate_sequence(scheduler, 128, 1)
    free_states = scheduler.state_manager.get_num_free_checkpoint()
    free_blocks = scheduler.block_manager.get_num_free_gpu_blocks()

    def unexpected_reservation(*args):
        pytest.fail('An unaligned prefill must not try to allocate or evict a checkpoint.')

    monkeypatch.setattr(scheduler.state_checkpoints, '_reserve_checkpoint', unexpected_reservation)
    copy_plan = scheduler.state_checkpoints.reserve_prefill_save_batch([seq])
    assert not copy_plan.state_pairs
    assert not copy_plan.kv_block_pairs
    assert node.state_checkpoint.published
    assert scheduler.state_manager.get_num_free_checkpoint() == free_states
    assert scheduler.block_manager.get_num_free_gpu_blocks() == free_blocks


def test_prefill_alignment_rejection_discards_old_unpublished_reservation():
    scheduler = _make_scheduler()
    seq = _allocate_sequence(scheduler, 1024)
    free_states = scheduler.state_manager.get_num_free_checkpoint()
    scheduler.state_checkpoints.reserve_prefill_save_batch([seq], (512, ))
    node = seq.prefix_cache.pending_save.node
    assert seq.prefix_cache.pending_save.is_pending
    assert scheduler.state_manager.get_num_free_checkpoint() == free_states - 1

    copy_plan = scheduler.state_checkpoints.reserve_prefill_save_batch([seq], (768, ))

    assert not copy_plan.state_pairs
    assert not copy_plan.kv_block_pairs
    assert not seq.prefix_cache.pending_save.is_pending
    assert node.state_checkpoint is None
    assert scheduler.state_manager.get_num_free_checkpoint() == free_states


def test_prefill_alignment_rejection_preserves_published_checkpoint():
    scheduler = _make_scheduler()
    seq = _allocate_sequence(scheduler, 1024)
    scheduler.state_checkpoints.reserve_prefill_save_batch([seq], (512, ))
    node = seq.prefix_cache.pending_save.node
    scheduler.state_checkpoints.finish_forward_dispatch([seq], has_save_plan=True)
    scheduler.state_checkpoints.unpin_saves([seq])
    free_states = scheduler.state_manager.get_num_free_checkpoint()

    copy_plan = scheduler.state_checkpoints.reserve_prefill_save_batch([seq], (768, ))

    assert not copy_plan.state_pairs
    assert not seq.prefix_cache.pending_save.is_pending
    assert node.state_checkpoint.published
    assert node.state_checkpoint.step == 512
    assert scheduler.state_manager.get_num_free_checkpoint() == free_states


@pytest.mark.parametrize(('length', 'history', 'saved'), [(1024, 768, True), (768, 512, False)])
def test_prefill_alignment_uses_absolute_prefix_position(length, history, saved):
    scheduler = _make_scheduler()
    seq = _allocate_sequence(scheduler, length)
    seq.set_step(history)
    assert seq.num_token_ids == 256
    copy_plan = scheduler.state_checkpoints.reserve_prefill_save_batch([seq])
    assert bool(copy_plan.state_pairs) == saved
    assert seq.prefix_cache.pending_save.is_pending == saved
    if saved:
        assert seq.prefix_cache.pending_save.step == length


@pytest.mark.parametrize(('lookahead', 'length', 'saved'), [(0, 512, True), (1, 512, False), (1, 513, True)])
def test_prefill_alignment_preserves_lookahead_guard(lookahead, length, saved):
    scheduler = _make_scheduler(lookahead=lookahead)
    seq = _allocate_sequence(scheduler, length)
    copy_plan = scheduler.state_checkpoints.reserve_prefill_save_batch([seq], (512, ))
    assert bool(copy_plan.state_pairs) == saved
    assert seq.prefix_cache.pending_save.is_pending == saved


def test_prefill_alignment_does_not_filter_decode_checkpoints():
    scheduler = _make_scheduler()
    seq = _allocate_sequence(scheduler, 768)
    copy_plan = scheduler.state_checkpoints.reserve_decode_save_batch([seq], interval=64)
    assert copy_plan.state_pairs == ((seq.logical_state, seq.prefix_cache.pending_save.slot), )
    assert seq.prefix_cache.pending_save.step == 768
    assert seq.prefix_cache.pending_save.is_decode


@pytest.mark.parametrize('steps', [(), (512, 1024)])
def test_prefill_alignment_preserves_steps_length_validation(steps):
    scheduler = _make_scheduler()
    seq = _allocate_sequence(scheduler, 512)
    with pytest.raises(ValueError, match='one entry per prefill sequence'):
        scheduler.state_checkpoints.reserve_prefill_save_batch([seq], steps)


def test_prefill_alignment_preserves_empty_batch():
    scheduler = _make_scheduler()
    copy_plan = scheduler.state_checkpoints.reserve_prefill_save_batch([])
    assert not copy_plan.state_pairs
    assert not copy_plan.kv_block_pairs

# Copyright (c) OpenMMLab. All rights reserved.
"""Admission and actual forward extents must agree after prefix matching."""
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.configurations.glm5_next import update_cache_config
from lmdeploy.pytorch.engine.inputs_maker import InputsMakerAsync, LongContextChunker, _ForwardInputsTask
from lmdeploy.pytorch.messages import SequenceMeta
from lmdeploy.pytorch.paging import Scheduler
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
from lmdeploy.pytorch.strategies.ar_spec.sequence import ARSpecSequenceStrategy


def make_case(mtp, prealloc):
    cache = CacheConfig(max_batches=4, block_size=64, num_cpu_blocks=0, num_gpu_blocks=64,
                        enable_prefix_caching=True, max_prefill_token_num=512,
                        states_shapes=[((1,), torch.float32)], num_state_caches=8,
                        prefix_cache_state_budget=4)
    update_cache_config(cache, token_lookahead=int(mtp))
    strategy = ARSpecSequenceStrategy() if mtp else ARSequenceStrategy()
    scheduler = Scheduler(SchedulerConfig(max_batches=4, max_session_len=4096), cache,
                          SequenceMeta(64, strategy=strategy))
    maker = InputsMakerAsync.__new__(InputsMakerAsync)
    maker.scheduler = scheduler
    maker.executor = SimpleNamespace(device_type='cuda')
    maker.config = SimpleNamespace(role=cache.role, is_ssm=True, use_mrope=False)
    maker.engine_strategy = SimpleNamespace(get_prealloc_size=lambda _: prealloc)
    maker.long_context_chunker = LongContextChunker(512)
    maker._is_long_context_chunk_turn_due = lambda: True
    maker.kernel_blocks_per_kv = 1
    maker._set_adapter_ids = lambda *args: None
    maker._prepare_prefill_cache_inputs = lambda *args, **kwargs: None
    maker.create_model_inputs_delta_valid_only = lambda: (None, [], [])
    maker.model_agent_strategy = SimpleNamespace(make_extra_inputs=lambda *args: None)
    return scheduler, maker


def cache_prefix(scheduler, tokens, step):
    producer = scheduler.add_session(99).add_sequence(tokens)
    scheduler.block_manager.allocate(producer)
    scheduler.block_trie.allocate(producer)
    assert scheduler.block_trie.state_checkpoints.reserve_save(producer, step=step) >= 0
    assert scheduler.block_trie.state_checkpoints.publish_save(producer)
    scheduler.end_session(99)


def assert_extents(result):
    inputs = result.inputs
    for row, seq in enumerate(result.running):
        end = seq.num_history_ids + int(inputs.seq_length[row])
        assert end <= seq.num_blocks * seq.block_size
        if seq.kv_token_limit is not None:
            assert len(result.running) == 1 and inputs.is_chunk
            assert end == seq.kv_token_limit
        else:
            assert int(inputs.seq_length[row]) == seq.num_token_ids


@pytest.mark.parametrize('mtp', [False, True])
@pytest.mark.parametrize('prealloc', [0, 4, 65])
@pytest.mark.parametrize('order', ['hit-first', 'miss-first', 'both-hit'])
def test_checkpoint_cut_is_exclusive_after_real_prefix_match(mtp, prealloc, order):
    scheduler, maker = make_case(mtp, prealloc)
    a = list(range(129))
    b = list(range(1000, 1257))
    cache_prefix(scheduler, a, 128)
    if order == 'both-hit':
        cache_prefix(scheduler, b, 256)
    prompts = [b, a] if order == 'miss-first' else [a, b]
    seqs = [scheduler.add_session(i).add_sequence(tokens) for i, tokens in enumerate(prompts)]
    task = _ForwardInputsTask(maker, True)
    task._select_prefill_work()
    result = task.result
    assert_extents(result)
    if order == 'both-hit':
        assert result.running == seqs
        assert result.inputs.seq_length.tolist() == [1, 1]
        assert not result.inputs.is_chunk
        return
    first, second = seqs
    assert result.running == [first]
    assert second.num_blocks == 0 and second.logical_state < 0
    assert second.num_history_ids == 0 and second.kv_token_limit is None
    assert second.prefix_cache.restore.node is None  # no tentative references leaked
    if order == 'hit-first':
        assert result.inputs.seq_length.tolist() == [1]
        scheduler.end_session(first.session_id)
        task = _ForwardInputsTask(maker, True)
        task._select_prefill_work()
        result = task.result
        assert result.running == [second]
        assert_extents(result)
    chunk_seq = result.running[0]
    assert result.inputs.is_chunk and result.inputs.is_first_chunk
    assert result.inputs.seq_length.tolist() == [256]
    # Simulate completed forward, then use the production continuation path.
    assert scheduler.block_trie.state_checkpoints.reserve_save(chunk_seq, step=256) >= 0
    assert scheduler.block_trie.state_checkpoints.publish_save(chunk_seq)
    chunk_seq.set_step(256)
    final = task._build_active_chunk()
    assert_extents(final)
    assert final.inputs.is_chunk and final.inputs.is_last_chunk
    assert final.inputs.seq_length.tolist() == [1]
    assert chunk_seq.kv_token_limit is None and chunk_seq.num_blocks >= 5
    scheduler.end_session(chunk_seq.session_id)
    repeat = scheduler.add_session(10).add_sequence(b)
    # The completed checkpoint must be reusable, not lost by rejected admission.
    scheduler.block_trie.match(repeat)
    assert repeat.num_history_ids == 256


@pytest.mark.parametrize('mtp', [False, True])
def test_full_prefill_rejects_chunk_allocation_before_tensor_construction(mtp):
    scheduler, maker = make_case(mtp, 4)
    seq = scheduler.add_session(0).add_sequence(list(range(257)))
    scheduler.schedule(is_prefill=True)
    assert seq.kv_token_limit == 256
    with pytest.raises(AssertionError, match='exclusive chunk'):
        _ForwardInputsTask(maker, True)._build_prefill_inputs([seq])


@pytest.mark.parametrize('mtp', [False, True])
@pytest.mark.parametrize('failure', ['pin', 'capacity', 'state'])
def test_batch_candidate_losing_final_prefix_match_is_rolled_back(monkeypatch, mtp, failure):
    scheduler, maker = make_case(mtp, 4)
    a, b = list(range(129)), list(range(1000, 1257))
    cache_prefix(scheduler, a, 128)
    cache_prefix(scheduler, b, 256)
    first = scheduler.add_session(0).add_sequence(a)
    second = scheduler.add_session(1).add_sequence(b)
    checkpoints = scheduler.block_trie.state_checkpoints
    if failure == 'pin':
        original = checkpoints.pin_restore
        monkeypatch.setattr(checkpoints, 'pin_restore', lambda seq: False if seq is second else original(seq))
    elif failure == 'capacity':
        original = scheduler.eviction_helper.try_make_capacity_for
        monkeypatch.setattr(scheduler.eviction_helper, 'try_make_capacity_for',
                            lambda seq, *args: False if seq is second else original(seq, *args))
    else:
        monkeypatch.setattr(scheduler.eviction_helper, 'try_make_capacity_for', lambda *args: True)
        answers = iter([True, False])
        monkeypatch.setattr(checkpoints, 'make_runtime_state_available', lambda: next(answers))
    task = _ForwardInputsTask(maker, True)
    task._select_prefill_work()
    assert task.result.running == [first]
    assert_extents(task.result)
    assert second.num_history_ids == 0 and second.num_blocks == 0
    assert second.logical_state < 0 and second.kv_token_limit is None
    assert not second.prefix_cache.restore.pinned
    assert second.prefix_cache.restore.node is None

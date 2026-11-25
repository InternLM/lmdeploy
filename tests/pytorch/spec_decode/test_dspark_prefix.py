# Copyright (c) OpenMMLab. All rights reserved.
"""DSpark aligned checkpoint admission and request-private suffixes."""
from types import SimpleNamespace

import numpy as np
import pytest

from lmdeploy.pytorch.config import DistConfig, SSMStateCapacity
from lmdeploy.pytorch.spec_decode.dspark_utils import validate_dspark_cache_config
from lmdeploy.pytorch.strategies.ar_spec import ARSpecStrategyFactory
from tests.pytorch.engine.test_compact_allocation_selection import _configs, _executor
from tests.pytorch.paging.test_block_trie.test_compact_checkpoint import _environment, _sequence, _submit


def _prefix_configs(budget=2):
    model, cache, spec = _configs(4)
    cache.enable_prefix_caching = True
    cache.prefix_cache_state_budget = budget
    cache.compact_state_capacity = SSMStateCapacity.from_cache_config(cache, 4)
    spec.method = 'dspark'
    spec.model_config.hf_config = SimpleNamespace(architectures=['Qwen3DSparkModel'])
    return model, cache, spec


@pytest.mark.parametrize('budget', [0, 2])
def test_dspark_prefix_admission(budget):
    model, cache, spec = _prefix_configs(budget)
    capacity = cache.compact_state_capacity
    validate_dspark_cache_config(cache)
    executor = _executor(model, cache, spec)
    assert executor.cache_config.enable_prefix_caching
    assert cache.compact_state_capacity is capacity
    assert cache.prefix_cache_token_lookahead == 0
    assert not spec.cache_config.enable_prefix_caching


@pytest.mark.parametrize('case', ['family', 'compact', 'indexed', 'gdn', 'attn_tp', 'draft', 'replay'])
def test_dspark_prefix_rejects_unqualified_pairs(case):
    model, cache, spec = _prefix_configs()
    if case == 'family':
        model.hf_config.model_type = 'glm5_next'
    elif case == 'compact':
        cache.compact_state_capacity = None
    elif case == 'indexed':
        model.state_cache_is_indexed = False
    elif case == 'gdn':
        model.is_gated_delta = False
    elif case == 'draft':
        spec.model_config.hf_config.architectures = ['DFlashDraftModel']
    elif case == 'replay':
        model.enable_replayssm = True
        cache.compact_state_capacity = SSMStateCapacity.from_cache_config(cache, 4, runtime_is_auxiliary=True)
    if case == 'attn_tp':
        from lmdeploy.pytorch.config import BackendConfig, MiscConfig
        from lmdeploy.pytorch.engine.executor.base import ExecutorBase
        with pytest.raises((ValueError, NotImplementedError)):
            ExecutorBase('', model, cache, BackendConfig(), DistConfig(tp=2), MiscConfig(), specdecode_config=spec)
    else:
        with pytest.raises((ValueError, NotImplementedError)):
            _executor(model, cache, spec)


@pytest.mark.parametrize('method,shift,recompute,lookahead', [
    ('dspark', 0, 0, 3), ('dflash', 0, 1, 4), ('qwen3_5_mtp', 1, 1, 3)])
def test_dspark_prefix_strategy_contract(method, shift, recompute, lookahead):
    factory = ARSpecStrategyFactory(SimpleNamespace(bos_token_id=0), SimpleNamespace(
        method=method, num_speculative_tokens=3, dspark=SimpleNamespace(draft_query_len=3)))
    strategy = factory.build_sequence_strategy()
    assert strategy.prefill_input_shift == shift
    assert strategy.recompute_blocks == recompute
    engine = factory.build_engine_strategy(None, SimpleNamespace(prefill_interval=1))
    assert engine.draft_kv_lookahead == lookahead
    assert engine.get_num_required_tokens() == 4 + lookahead


@pytest.mark.parametrize('length,diverge,private_end', [(1160, 1024, 1024), (1200, 1152, 1152),
                                                     (1088, None, 1088), (1160, None, 1152)])
def test_dspark_checkpoint_hit_keeps_recomputed_suffix_private(length, diverge, private_end):
    scheduler, owner, trie = _environment()
    scheduler.seq_meta.strategy = ARSpecStrategyFactory(SimpleNamespace(bos_token_id=0),
                                                        SimpleNamespace(method='dspark')).build_sequence_strategy()
    scheduler.seq_meta.prefill_save_alignment = 1024
    tokens = list(range(1200))
    producer = _sequence(scheduler, owner, trie, tokens)
    slot = trie.state_checkpoints.reserve_save(producer, step=1024)
    node = producer.prefix_cache.pending_save.node
    _, access = _submit(owner, producer)
    assert trie.state_checkpoints.publish_save(producer)
    owner.acknowledge(access.access_id)
    canonical = producer.logical_blocks.get_real_blocks().copy()
    readers = []
    for _ in range(2):
        prompt = tokens[:length]
        if diverge is not None:
            prompt[diverge:] = [9999 + len(readers)] * (length - diverge)
        reader = scheduler.add_session(len(scheduler.sessions)).add_sequence(prompt)
        trie.match(reader)
        assert reader.num_history_ids == 1024
        fresh = reader.prefix_cache.recompute_overlap.fresh_block_range
        assert list(fresh or []) == list(range(16, private_end // 64))
        owner.allocate(reader)
        scheduler.block_manager.allocate(reader)
        trie.allocate(reader)
        assert trie.state_checkpoints.bind_restore(reader) == slot
        ids = reader.logical_blocks.get_real_blocks()
        np.testing.assert_array_equal(ids[:16], canonical[:16])
        assert set(ids[16:]).isdisjoint(canonical)
        for previous in readers:
            assert set(ids[16:]).isdisjoint(previous.logical_blocks.get_real_blocks()[16:])
        for idx in range(16, private_end // 64):
            assert reader.prefix_cache.recompute_overlap.trie_block_map[idx] == canonical[idx]
        readers.append(reader)
    assert node.state_checkpoint.pin_count == 2


@pytest.mark.parametrize('budget', [0, 2])
def test_dspark_prefix_real_input_packet(budget):
    from tests.pytorch.engine.test_compact_inputs_maker import _maker

    maker = _maker(4, budget=2048, requests=1, prefix=True, block_size=64,
                   checkpoint_budget=budget, save_alignment=1024)
    factory = ARSpecStrategyFactory(SimpleNamespace(bos_token_id=0), SimpleNamespace(
        method='dspark', num_speculative_tokens=3, dspark=SimpleNamespace(draft_query_len=3)))
    scheduler = maker.scheduler
    scheduler.seq_meta.strategy = factory.build_sequence_strategy()
    maker.engine_strategy = factory.build_engine_strategy(scheduler.cache_config, scheduler.scheduler_config)
    producer = scheduler.add_session(0).add_sequence(list(range(1100)))
    packet = maker._make_forward_inputs(True)
    assert packet['inputs'].history_lengths.tolist() == [0]
    assert packet['inputs'].seq_length.tolist() == [1100]
    assert producer.prefix_cache.pending_save.is_pending == bool(budget)
    assert maker.state_checkpoints.publish_save(producer) == bool(budget)
    ticket = packet['cache_inputs'].ssm_state_access_id
    maker.compact_state_manager.acknowledge(ticket)
    maker.state_checkpoints.complete_forward(ticket)
    scheduler.end_session(0)
    reader = scheduler.add_session(1).add_sequence(list(range(1024)) + [9000] * 64)
    packet = maker._make_forward_inputs(True)
    expected = 1024 if budget else 0
    assert reader.num_history_ids == expected
    assert packet['inputs'].history_lengths.tolist() == [expected]
    assert packet['inputs'].seq_length.tolist() == [1088 - expected]
    assert reader.prefix_cache.recompute_overlap.recompute_blocks == 0
    assert maker.engine_strategy.get_prealloc_size(False) == 3
    assert producer.session_id not in scheduler.sessions

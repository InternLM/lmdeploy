# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import lmdeploy.pytorch.engine.cache_engine.state as state_module
from lmdeploy.pytorch.config import CacheConfig, StateCacheSpec
from lmdeploy.pytorch.engine.cache_engine import StateCacheEngine
from lmdeploy.pytorch.engine.cache_engine.layout import (
    CacheAllocation,
    CachePool,
    ContiguousStateCacheLayout,
    PackedStateCacheLayout,
)
from lmdeploy.pytorch.engine.cache_engine.schema import build_state_cache_tensor_specs


def _allocate_test_state_caches(num_caches: int,
                                state_shapes,
                                device: torch.device | str = 'cpu',
                                state_specs=None) -> CacheAllocation:
    tensor_specs = build_state_cache_tensor_specs(state_shapes, state_specs=state_specs)
    return state_module._allocate_state_caches(tensor_specs, num_caches=num_caches, device=device)


def test_layered_state_cache_specs_do_not_require_total_layer_count():
    state_specs = [StateCacheSpec('subset', (96, ), torch.float32, layer_ids=[1, 9])]
    state_shapes = [(spec.shape, spec.dtype) for spec in state_specs]

    allocation = _allocate_test_state_caches(num_caches=2,
                                             state_shapes=state_shapes,
                                             state_specs=state_specs)
    mem_pool = allocation.pools[0].tensor
    caches = allocation.tensor_views

    assert isinstance(allocation, CacheAllocation)
    assert tuple(mem_pool.shape) == (2, 768)
    assert tuple(caches[0].shape) == (2, 2, 96)
    assert StateCacheEngine.get_state_slot_nbytes(state_shapes, state_specs=state_specs) == 768


def _make_multi_pool_state_allocation(num_caches: int, device: torch.device | str = 'cpu'):
    first = torch.zeros((num_caches, 2), dtype=torch.float32, device=device)
    second = torch.zeros((3, num_caches, 2), dtype=torch.float16, device=device)
    return CacheAllocation(
        pools=(CachePool(first, entry_axis=0), CachePool(second, entry_axis=1)),
        tensor_views=(first, second),
    )


def test_state_cache_engine_accepts_multi_pool_layout(monkeypatch):
    layout = SimpleNamespace(
        allocate=lambda num_caches, device: _make_multi_pool_state_allocation(num_caches, device),
    )
    cache_backend = SimpleNamespace(build_state_layout=lambda tensor_specs: layout)
    ops_backend = SimpleNamespace(get_cache_backend=lambda: cache_backend)
    monkeypatch.setattr(state_module, 'get_backend', lambda: ops_backend)

    allocation = _allocate_test_state_caches(num_caches=2, state_shapes=[])

    assert [pool.entry_axis for pool in allocation.pools] == [0, 1]
    assert allocation.nbytes == 2 * (2 * 4 + 3 * 2 * 2)
    assert StateCacheEngine.get_state_slot_nbytes([]) == 2 * 4 + 3 * 2 * 2


def test_layer_scoped_state_cache_specs_reject_invalid_layer_ids():
    negative_layer = [StateCacheSpec('bad', (1, ), torch.float32, layer_ids=[-1])]
    with pytest.raises(ValueError, match='non-negative'):
        _allocate_test_state_caches(num_caches=1,
                                    state_shapes=[((1, ), torch.float32)],
                                    state_specs=negative_layer,
                                    device='meta')

    empty_state_layers = [StateCacheSpec('empty', (1, ), torch.float32, layer_ids=[])]
    with pytest.raises(ValueError, match='must not be empty'):
        _allocate_test_state_caches(num_caches=1,
                                    state_shapes=[((1, ), torch.float32)],
                                    state_specs=empty_state_layers,
                                    device='meta')


def _make_state_cache_engine(num_caches: int = 4):
    cache_engine = object.__new__(StateCacheEngine)
    cache_engine.cache_config = CacheConfig(max_batches=1,
                                            block_size=64,
                                            num_cpu_blocks=0,
                                            num_gpu_blocks=0,
                                            num_state_caches=num_caches,
                                            states_shapes=[((2, 3), torch.float32), ((2, ), torch.float16)])
    cache_engine.allocation = _allocate_test_state_caches(num_caches=num_caches,
                                                          state_shapes=cache_engine.cache_config.states_shapes)
    cache_engine._cache_tensors = list(cache_engine.allocation.tensor_views)
    cache_engine._slot_tensors = tuple((pool.tensor, pool.entry_axis) for pool in cache_engine.allocation.pools)
    return cache_engine


def _make_multi_pool_state_cache_engine(num_caches: int = 4):
    cache_engine = object.__new__(StateCacheEngine)
    cache_engine.cache_config = CacheConfig(max_batches=1,
                                            block_size=64,
                                            num_cpu_blocks=0,
                                            num_gpu_blocks=0,
                                            num_state_caches=num_caches,
                                            states_shapes=[])
    cache_engine.allocation = _make_multi_pool_state_allocation(num_caches)
    cache_engine._cache_tensors = list(cache_engine.allocation.tensor_views)
    cache_engine._slot_tensors = tuple((pool.tensor, pool.entry_axis) for pool in cache_engine.allocation.pools)
    return cache_engine


def test_connector_state_cache_pools_expose_owning_pools_and_axes():
    cache_engine = _make_multi_pool_state_cache_engine()

    pools = cache_engine.connector_state_cache_pools

    assert [pool.entry_axis for pool in pools] == [0, 1]
    assert all(pool.tensor is allocation_pool.tensor
               for pool, allocation_pool in zip(pools, cache_engine.allocation.pools))


@pytest.fixture(params=['cpu'])
def glm_state_cache_engine(monkeypatch, request):
    from lmdeploy.hf_configs.configuration_glm5_next import Glm5NextConfig
    from lmdeploy.pytorch.configurations.glm5_next import Glm5NextModelConfigBuilder

    monkeypatch.setattr('lmdeploy.pytorch.configurations.deepseek_v2.flash_mla_available', lambda: True)
    monkeypatch.setattr(state_module, '_allocate_state_caches',
                        lambda specs, num_caches, device: PackedStateCacheLayout(tuple(specs)).allocate(
                            num_caches, device=request.param))
    hf_config = Glm5NextConfig(text_config=dict(
        num_hidden_layers=2, layer_types=['linear_attention', 'full_attention'],
        linear_num_heads=2, linear_head_dim=8, index_head_dim=8))
    model_config = Glm5NextModelConfigBuilder.build(hf_config, num_spec_tokens=3)
    cache_config = CacheConfig(max_batches=1, block_size=64, num_cpu_blocks=0,
                               num_gpu_blocks=0, num_state_caches=4, states_shapes=model_config.states_shapes)
    engine = StateCacheEngine(cache_config, model_config)
    return engine, model_config


def test_glm_store_projection_preserves_local_state_copies(glm_state_cache_engine):
    engine, model_config = glm_state_cache_engine
    assert [spec.store_cacheable for spec in model_config.state_cache_specs] == [True, True, False, False]
    checkpoint_pool, runtime_pool = engine.connector_state_cache_pools
    assert checkpoint_pool.transfer_ranges is None
    assert runtime_pool.transfer_ranges == ()
    assert checkpoint_pool.tensor is engine.allocation.pools[0].tensor
    assert runtime_pool.tensor is engine.allocation.pools[1].tensor
    for value, cache in enumerate(engine.state_caches, start=1):
        cache[1].fill_(value)
    engine.copy_slots(1, 3)
    for value, cache in enumerate(engine.state_caches, start=1):
        assert (cache[3] == value).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('glm_state_cache_engine', ['cuda'], indirect=True)
@pytest.mark.parametrize('boundary_offset', [1, 2, 4])
def test_glm_checkpoint_restores_the_selected_verification_position(
        monkeypatch, glm_state_cache_engine, boundary_offset):
    from lmdeploy.pytorch.backends import gated_delta_rule as meta_module

    engine, _ = glm_state_cache_engine
    conv, recurrent, tail_keys, tail_scores = engine.state_caches
    for cache in engine.state_caches:
        cache.fill_(-10)
    # Mark every entry by its absolute position, including the unaccepted tail.
    boundary, num_spec_tokens = 256, 3
    history = boundary - boundary_offset
    ring_size = num_spec_tokens + 1
    conv_length = conv.shape[-1]
    conv_width = conv_length - num_spec_tokens
    for position in range(history + 1, history + ring_size + 1):
        recurrent[1, 0, position % ring_size].fill_(position)
    for position in range(history + ring_size - conv_length, history + ring_size):
        conv[1, 0, :, position % conv_length].fill_(position)
    plan = torch.tensor([[0, 0], [boundary, boundary], [1, 1], [2, 3]], device='cuda')
    engine.copy_decode_slots(plan, torch.tensor([True, False], device='cuda'))
    assert (recurrent[3] == -10).all()
    for cache in engine.state_caches:
        cache[1].fill_(999)  # Later forwards reuse the runtime state.
    engine.copy_slots(2, 3, checkpoint_only=True)

    # Use the actual prefill restore metadata to select the saved boundary.
    monkeypatch.setattr(meta_module, 'get_step_ctx_manager',
                        lambda: SimpleNamespace(build_ctx=SimpleNamespace(num_spec_tokens=num_spec_tokens)))
    meta = meta_module.GatedDeltaMeta(1, conv_width, torch.tensor([3]), SimpleNamespace(
        is_decoding=False, cu_seqlens_q=torch.tensor([0, 1]),
        q_seqlens=torch.tensor([1]), kv_seqlens=torch.tensor([boundary + 1])))
    restored_state = recurrent[3, 0, meta.spec_state_offsets[0][0]]
    torch.testing.assert_close(restored_state, torch.full_like(restored_state, boundary))
    restored_conv = conv[3, 0, :, meta.spec_conv_offsets[0][0]]
    expected_conv = torch.arange(boundary - conv_width + 1, boundary, dtype=conv.dtype, device=conv.device)
    torch.testing.assert_close(restored_conv, expected_conv.expand_as(restored_conv))
    assert (tail_keys[3] == -10).all() and (tail_scores[3] == -10).all()


def test_state_cache_engine_zero_slots_uses_each_pool_entry_axis():
    cache_engine = _make_multi_pool_state_cache_engine()
    first, second = cache_engine.state_caches
    first.fill_(1)
    second.fill_(1)

    cache_engine.zero_slots(torch.tensor([1, 3]), torch.tensor([True, False]))

    assert torch.count_nonzero(first[1]) == 0
    assert torch.count_nonzero(second[:, 1]) == 0
    assert torch.all(first[3] == 1)
    assert torch.all(second[:, 3] == 1)


def test_state_cache_engine_copy_slots_uses_each_pool_entry_axis():
    cache_engine = _make_multi_pool_state_cache_engine()
    first, second = cache_engine.state_caches
    first[0].fill_(3)
    first[1].fill_(5)
    second[:, 0].fill_(7)
    second[:, 1].fill_(9)

    cache_engine.copy_slots((1, 0), (3, 2))

    assert torch.equal(first[2], first[0])
    assert torch.equal(first[3], first[1])
    assert torch.equal(second[:, 2], second[:, 0])
    assert torch.equal(second[:, 3], second[:, 1])


def test_state_cache_engine_copy_slots_copies_all_state_views():
    cache_engine = _make_state_cache_engine()
    conv_state, recurrent_state = cache_engine.state_caches

    conv_state[1].fill_(3.0)
    recurrent_state[1].fill_(5.0)

    cache_engine.copy_slots(1, 2)

    assert torch.equal(conv_state[2], conv_state[1])
    assert torch.equal(recurrent_state[2], recurrent_state[1])


def test_state_cache_engine_copy_slots_supports_batched_indices():
    cache_engine = _make_state_cache_engine()
    conv_state, recurrent_state = cache_engine.state_caches

    conv_state[0].fill_(1.0)
    recurrent_state[0].fill_(2.0)
    conv_state[1].fill_(3.0)
    recurrent_state[1].fill_(4.0)

    cache_engine.copy_slots((1, 0), (3, 2))

    assert torch.equal(conv_state[2], conv_state[0])
    assert torch.equal(recurrent_state[2], recurrent_state[0])
    assert torch.equal(conv_state[3], conv_state[1])
    assert torch.equal(recurrent_state[3], recurrent_state[1])


def test_state_cache_engine_copy_slots_accepts_host_integer_scalars():
    cache_engine = _make_state_cache_engine()
    conv_state, recurrent_state = cache_engine.state_caches

    conv_state[1].fill_(7.0)
    recurrent_state[1].fill_(9.0)

    cache_engine.copy_slots(np.int64(1), np.int64(2))

    assert torch.equal(conv_state[2], conv_state[1])
    assert torch.equal(recurrent_state[2], recurrent_state[1])


def test_state_cache_engine_copy_slots_coalesces_contiguous_ranges():
    ranges = list(StateCacheEngine._coalesce_copy_ranges([4, 1, 5, 0, 6, 9], [20, 11, 21, 10, 22, 30]))

    assert ranges == [(0, 10, 2), (4, 20, 3), (9, 30, 1)]
    assert list(StateCacheEngine._coalesce_copy_ranges([], [])) == []


def test_state_cache_engine_copy_slots_rejects_mismatched_indices():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(ValueError, match='same number of elements'):
        cache_engine.copy_slots([0, 1], [2])


def test_state_cache_engine_copy_slots_rejects_tensor_indices():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(TypeError, match='host integers'):
        cache_engine.copy_slots(torch.tensor([0, 1]), torch.tensor([2, 3]))


def test_state_cache_engine_copy_slots_rejects_tensor_sequence_items():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(TypeError, match='host integers'):
        cache_engine.copy_slots([torch.tensor(0)], [2])


def test_state_cache_engine_copy_slots_rejects_out_of_range_indices():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(ValueError, match='out of range'):
        cache_engine.copy_slots([-1], [2])

    with pytest.raises(ValueError, match='out of range'):
        cache_engine.copy_slots([0], [4])


def test_state_cache_engine_copy_slots_rejects_overlapping_indices():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(ValueError, match='must not overlap'):
        cache_engine.copy_slots([0, 1], [1, 2])


def test_state_cache_engine_copy_slots_rejects_duplicate_destinations():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(ValueError, match='duplicate'):
        cache_engine.copy_slots([0, 1], [2, 2])


@pytest.mark.parametrize('layout_cls', [PackedStateCacheLayout, ContiguousStateCacheLayout])
@pytest.mark.parametrize('flags', [(True, True, True), (False, True, False), (False, False, False)])
def test_state_cache_engine_checkpoint_policy_preserves_runtime_state(monkeypatch, layout_cls, flags):
    specs = [StateCacheSpec(f'state_{i}', (2, 3), torch.float32,
                            layer_ids=[1, 3] if i == 1 else None, prefix_checkpoint=flag)
             for i, flag in enumerate(flags)]
    shapes = [(spec.shape, spec.dtype) for spec in specs]
    config = CacheConfig(max_batches=1, block_size=64, num_cpu_blocks=0,
                         num_gpu_blocks=0, num_state_caches=4, states_shapes=shapes)

    def allocate(tensor_specs, num_caches, device):
        return layout_cls(tuple(tensor_specs)).allocate(num_caches, 'cpu' if device == 'cuda' else device)

    monkeypatch.setattr(state_module, '_allocate_state_caches', allocate)
    engine = StateCacheEngine(config, SimpleNamespace(state_cache_specs=specs))
    assert engine.allocation.nbytes == 4 * engine.get_state_slot_nbytes(shapes, specs)
    # The middle declaration is layered; its model-facing slot axis is 1.
    views = [view.movedim(1, 0) if i == 1 else view for i, view in enumerate(engine.state_caches)]
    for i, view in enumerate(views):
        assert view.shape == ((4, 2, 2, 3) if i == 1 else (4, 2, 3))
        view.fill_(-11)
        view[0].fill_(i + 1)

    engine.copy_slots(0, 2, checkpoint_only=True)
    for i, (flag, view) in enumerate(zip(flags, views)):
        assert torch.all(view[2] == (i + 1 if flag else -11))
        view[0].fill_(-7)  # Producer reuse must not change the frozen copy.
    engine.copy_slots(2, 1, checkpoint_only=True)
    for i, (flag, view) in enumerate(zip(flags, views)):
        assert torch.all(view[1] == (i + 1 if flag else -11))

    engine.copy_slots(0, 3)  # General movement still includes runtime-only state.
    for view in views:
        assert torch.all(view[3] == -7)
    engine.zero_slots(torch.tensor([0, 3]), torch.tensor([True, False]))
    for view in views:
        assert torch.count_nonzero(view[0]) == 0
        assert torch.all(view[3] == -7)

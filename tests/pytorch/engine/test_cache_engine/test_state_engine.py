# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import lmdeploy.pytorch.engine.cache_engine.state as state_module
from lmdeploy.pytorch.config import CacheConfig, StateCacheSpec
from lmdeploy.pytorch.engine.cache_engine import StateCacheEngine
from lmdeploy.pytorch.engine.cache_engine.layout import CacheAllocation, CachePool
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

# Copyright (c) OpenMMLab. All rights reserved.
import asyncio

import numpy as np
import pytest
import torch

from lmdeploy.pytorch.config import BlockCacheSpec, CacheConfig, ModelConfig, StateCacheSpec
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol
from lmdeploy.pytorch.disagg.messages import MigrationExecutionBatch
from lmdeploy.pytorch.engine.cache_engine import CacheEngine, NamedCacheView, StateCacheEngine


def _make_model_config(**kwargs):
    model_config = ModelConfig(hidden_size=16,
                               num_layers=4,
                               num_attention_heads=2,
                               num_key_value_heads=2,
                               bos_token_id=1,
                               eos_token_id=[2],
                               head_dim=8)
    for key, value in kwargs.items():
        setattr(model_config, key, value)
    return model_config


def test_allocate_caches_requires_block_size_divisible_by_kernel_block_size():
    cache_config = CacheConfig(max_batches=1,
                               block_size=96,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)

    with pytest.raises(ValueError, match='block_size 96 must be divisible by kernel_block_size 64'):
        CacheEngine.allocate_caches(num_blocks=1,
                                    model_config=None,
                                    cache_config=cache_config,
                                    world_size=1,
                                    device='meta')


def test_pd_migration_rejects_split_kernel_blocks():
    cache_engine = object.__new__(CacheEngine)
    cache_engine.cache_config = CacheConfig(max_batches=1,
                                            block_size=96,
                                            kernel_block_size=64,
                                            num_cpu_blocks=0,
                                            num_gpu_blocks=0)
    migration_inputs = MigrationExecutionBatch(protocol=MigrationProtocol.RDMA, requests=[])

    with pytest.raises(RuntimeError, match='PD migration does not support block_size != kernel_block_size'):
        asyncio.run(cache_engine.migrate(migration_inputs))


def test_named_block_cache_specs_allocate_only_declared_layers():
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)
    model_config = _make_model_config(
        use_standard_kv_cache=False,
        block_cache_specs=[
            BlockCacheSpec('r4', [1, 3], (40, ), torch.float32),
            BlockCacheSpec('r128', [2], (96, ), torch.float32),
        ],
    )

    mem_pool, caches = CacheEngine.allocate_caches(num_blocks=3,
                                                   model_config=model_config,
                                                   cache_config=cache_config,
                                                   world_size=1,
                                                   device='cpu')

    assert [tuple(pool.shape) for pool in mem_pool] == [(2, 3, 256), (1, 3, 512)]
    assert [tuple(cache.shape) for cache in caches] == [(2, 3, 40), (1, 3, 96)]
    assert CacheEngine.get_cache_block_size(cache_config, model_config) == 1024
    assert CacheEngine._get_block_cache_layer_maps(model_config) == {
        'r4': {
            1: 0,
            3: 1,
        },
        'r128': {
            2: 0,
        },
    }


def test_layered_state_cache_specs_allocate_only_declared_layers():
    state_specs = [StateCacheSpec('subset', (96, ), torch.float32, layer_ids=[1, 3])]
    state_shapes = [(spec.shape, spec.dtype) for spec in state_specs]

    mem_pool, caches = StateCacheEngine.allocate_caches(num_caches=2,
                                                        state_shapes=state_shapes,
                                                        state_specs=state_specs,
                                                        num_layers=4,
                                                        device='cpu')

    assert tuple(mem_pool.shape) == (2, 768)
    assert tuple(caches[0].shape) == (2, 2, 96)
    assert StateCacheEngine.get_cache_state_size(state_shapes, state_specs=state_specs, num_layers=4) == 768
    assert StateCacheEngine._get_state_cache_layer_maps(state_specs, 4) == {'subset': {1: 0, 3: 1}}


def test_layer_scoped_cache_specs_reject_invalid_layer_ids():
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)

    duplicate_layer = _make_model_config(
        use_standard_kv_cache=False,
        block_cache_specs=[BlockCacheSpec('dup', [1, 1], (1, ), torch.float32)],
    )
    with pytest.raises(ValueError, match='duplicated'):
        CacheEngine.allocate_caches(num_blocks=1,
                                    model_config=duplicate_layer,
                                    cache_config=cache_config,
                                    world_size=1,
                                    device='meta')

    out_of_range = [StateCacheSpec('bad', (1, ), torch.float32, layer_ids=[4])]
    with pytest.raises(ValueError, match='out of range'):
        StateCacheEngine.allocate_caches(num_caches=1,
                                         state_shapes=[((1, ), torch.float32)],
                                         state_specs=out_of_range,
                                         num_layers=4,
                                         device='meta')


def test_deepseek_v4_cache_accessors_resolve_layer_scoped_rows():
    from lmdeploy.pytorch.models.deepseek_v4 import V4Caches

    state_cache = torch.arange(24).view(2, 3, 4)
    block_cache = torch.arange(40).view(2, 5, 4)
    caches = V4Caches(
        named_state_caches=NamedCacheView({'state': state_cache}, {'state': {1: 0, 3: 1}}),
        block_caches=NamedCacheView({'block': block_cache}, {'block': {1: 0, 3: 1}}),
    )

    assert torch.equal(caches.state_cache('state', 3), state_cache[1])
    assert torch.equal(caches.block_cache('block', 1), block_cache[0])
    with pytest.raises(RuntimeError, match='does not own cache'):
        caches.state_cache('state', 2)


def test_named_cache_properties_return_dict_without_layer_maps():
    block_cache_engine = object.__new__(CacheEngine)
    block_cache_engine._cache_names = ['k_cache']
    block_cache_engine._cache_list = [torch.empty(1)]
    block_cache_engine._block_cache_layer_maps = {}

    state_cache_engine = object.__new__(StateCacheEngine)
    state_cache_engine._state_cache_names = ['state_0']
    state_cache_engine._state_caches = [torch.empty(1)]
    state_cache_engine._state_cache_layer_maps = {}

    assert type(block_cache_engine.block_caches) is dict
    assert type(state_cache_engine.named_state_caches) is dict


def _make_state_cache_engine(num_caches: int = 4):
    cache_engine = object.__new__(StateCacheEngine)
    cache_engine.cache_config = CacheConfig(max_batches=1,
                                            block_size=64,
                                            num_cpu_blocks=0,
                                            num_gpu_blocks=0,
                                            num_state_caches=num_caches,
                                            states_shapes=[((2, 3), torch.float32), ((2, ), torch.float16)])
    cache_engine.mem_pool, cache_engine._state_caches = StateCacheEngine.allocate_caches(
        num_caches=num_caches,
        state_shapes=cache_engine.cache_config.states_shapes,
        device='cpu',
    )
    return cache_engine


def test_state_cache_engine_copy_caches_copies_all_state_views():
    cache_engine = _make_state_cache_engine()
    conv_state, recurrent_state = cache_engine.state_caches

    conv_state[1].fill_(3.0)
    recurrent_state[1].fill_(5.0)

    cache_engine.copy_caches(1, 2)

    assert torch.equal(conv_state[2], conv_state[1])
    assert torch.equal(recurrent_state[2], recurrent_state[1])


def test_state_cache_engine_copy_caches_supports_batched_indices():
    cache_engine = _make_state_cache_engine()
    conv_state, recurrent_state = cache_engine.state_caches

    conv_state[0].fill_(1.0)
    recurrent_state[0].fill_(2.0)
    conv_state[1].fill_(3.0)
    recurrent_state[1].fill_(4.0)

    cache_engine.copy_caches((1, 0), (3, 2))

    assert torch.equal(conv_state[2], conv_state[0])
    assert torch.equal(recurrent_state[2], recurrent_state[0])
    assert torch.equal(conv_state[3], conv_state[1])
    assert torch.equal(recurrent_state[3], recurrent_state[1])


def test_state_cache_engine_copy_caches_accepts_host_integer_scalars():
    cache_engine = _make_state_cache_engine()
    conv_state, recurrent_state = cache_engine.state_caches

    conv_state[1].fill_(7.0)
    recurrent_state[1].fill_(9.0)

    cache_engine.copy_caches(np.int64(1), np.int64(2))

    assert torch.equal(conv_state[2], conv_state[1])
    assert torch.equal(recurrent_state[2], recurrent_state[1])


def test_state_cache_engine_copy_caches_coalesces_contiguous_ranges():
    ranges = list(StateCacheEngine._copy_ranges([4, 1, 5, 0, 6, 9], [20, 11, 21, 10, 22, 30]))

    assert ranges == [(0, 10, 2), (4, 20, 3), (9, 30, 1)]
    assert list(StateCacheEngine._copy_ranges([], [])) == []


def test_state_cache_engine_copy_caches_rejects_mismatched_indices():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(ValueError, match='same number of elements'):
        cache_engine.copy_caches([0, 1], [2])


def test_state_cache_engine_copy_caches_rejects_tensor_indices():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(TypeError, match='host integers'):
        cache_engine.copy_caches(torch.tensor([0, 1]), torch.tensor([2, 3]))


def test_state_cache_engine_copy_caches_rejects_tensor_sequence_items():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(TypeError, match='host integers'):
        cache_engine.copy_caches([torch.tensor(0)], [2])


def test_state_cache_engine_copy_caches_rejects_out_of_range_indices():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(ValueError, match='out of range'):
        cache_engine.copy_caches([-1], [2])

    with pytest.raises(ValueError, match='out of range'):
        cache_engine.copy_caches([0], [4])


def test_state_cache_engine_copy_caches_rejects_overlapping_indices():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(ValueError, match='must not overlap'):
        cache_engine.copy_caches([0, 1], [1, 2])


def test_state_cache_engine_copy_caches_rejects_duplicate_destinations():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(ValueError, match='duplicate'):
        cache_engine.copy_caches([0, 1], [2, 2])

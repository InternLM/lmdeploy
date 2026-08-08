# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import lmdeploy.pytorch.engine.cache_engine.engine as cache_engine_module
from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch.backends.dlinfer.op_backend import DlinferOpsBackend
from lmdeploy.pytorch.config import BlockCacheSpec, CacheConfig, ModelConfig, StateCacheSpec
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol
from lmdeploy.pytorch.disagg.messages import MigrationExecutionBatch
from lmdeploy.pytorch.engine.cache_engine import CacheEngine, NamedCacheView, StateCacheEngine
from lmdeploy.pytorch.engine.cache_engine.layout import CacheAllocation, CachePool
from lmdeploy.pytorch.engine.cache_engine.plan import BlockCachePlan
from lmdeploy.pytorch.engine.cache_engine.schema import CacheDesc, CacheResource


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


def test_bf16_sparse_mla_cache_layout():
    model_config = ModelConfig(hidden_size=6144,
                               num_layers=2,
                               num_attention_heads=64,
                               num_key_value_heads=1,
                               bos_token_id=None,
                               eos_token_id=[1],
                               head_dim=576,
                               k_head_dim=576,
                               v_head_dim=0,
                               dtype=torch.bfloat16,
                               use_flash_mla=True,
                               mla_kv_cache_dtype='bfloat16',
                               mla_index_topk=2048)
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)

    k_desc = CacheEngine.get_k_cache_desc(model_config, cache_config)
    v_desc = CacheEngine.get_v_cache_desc(model_config, cache_config)

    assert k_desc.shape == [64, 1, 576]
    assert k_desc.dtype == torch.bfloat16
    assert v_desc.shape == [64, 1, 0]
    assert v_desc.dtype == torch.bfloat16


def test_fp8_sparse_mla_cache_layout():
    model_config = ModelConfig(hidden_size=6144,
                               num_layers=2,
                               num_attention_heads=64,
                               num_key_value_heads=1,
                               bos_token_id=None,
                               eos_token_id=[1],
                               head_dim=576,
                               k_head_dim=576,
                               v_head_dim=0,
                               dtype=torch.bfloat16,
                               use_flash_mla=True,
                               mla_kv_cache_dtype='bfloat16',
                               mla_index_topk=2048)
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0,
                               quant_policy=QuantPolicy.FP8)
    CacheEngine.get_cache_block_size(cache_config, model_config)

    assert model_config.mla_kv_cache_dtype == 'fp8_ds_mla'

    k_desc = CacheEngine.get_k_cache_desc(model_config, cache_config)
    v_desc = CacheEngine.get_v_cache_desc(model_config, cache_config)

    assert k_desc.shape == [64, 1, 656]
    assert k_desc.dtype == torch.float8_e4m3fn
    assert v_desc.shape == [64, 1, 0]
    assert v_desc.dtype == torch.float8_e4m3fn


@pytest.mark.parametrize('quant_policy',
                         [QuantPolicy.INT4, QuantPolicy.INT8, QuantPolicy.FP8_E5M2, QuantPolicy.TURBO_QUANT])
def test_sparse_mla_rejects_other_cache_policies(quant_policy):
    model_config = SimpleNamespace(mla_index_topk=2048)
    cache_config = SimpleNamespace(quant_policy=quant_policy)

    with pytest.raises(ValueError, match='Sparse MLA does not support quant_policy'):
        CacheEngine.get_cache_block_size(cache_config, model_config)


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


def test_standard_cache_layout_preserves_pool_bytes_strides_and_tuple_order():
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)
    model_config = _make_model_config(dtype=torch.bfloat16)
    allocation = CacheEngine.allocate_caches(num_blocks=3,
                                             model_config=model_config,
                                             cache_config=cache_config,
                                             world_size=1,
                                             device='cpu')
    mem_pool, caches = allocation

    assert isinstance(allocation, CacheAllocation)
    assert allocation.nbytes == mem_pool.numel() * mem_pool.element_size()
    assert tuple(mem_pool.shape) == (4, 3, 4096)
    assert mem_pool.dtype == torch.uint8
    assert torch.count_nonzero(mem_pool) == 0
    assert [tuple(cache.shape) for cache in caches] == [
        (4, 3, 64, 2, 8),
        (4, 3, 64, 2, 8),
    ]
    assert [cache.dtype for cache in caches] == [torch.bfloat16, torch.bfloat16]
    assert [cache.stride(1) for cache in caches] == [2048, 2048]
    assert [cache.storage_offset() for cache in caches] == [0, 1024]
    assert CacheEngine.get_cache_block_size(cache_config, model_config) == mem_pool.numel() // 3


@pytest.mark.parametrize(
    ('quant_policy', 'resource_shapes', 'resource_dtypes', 'pool_bytes'),
    [
        (
            QuantPolicy.INT4,
            [(64, 2, 4), (64, 2, 4), (64, 2, 2), (64, 2, 2)],
            [torch.uint8, torch.uint8, torch.bfloat16, torch.bfloat16],
            2048,
        ),
        (
            QuantPolicy.INT8,
            [(64, 2, 8), (64, 2, 8), (64, 2, 2), (64, 2, 2)],
            [torch.uint8, torch.uint8, torch.bfloat16, torch.bfloat16],
            3072,
        ),
        (
            QuantPolicy.TURBO_QUANT,
            [(64, 2, 4), (64, 2, 2), (64, 2, 2), (64, 2, 1)],
            [torch.uint8, torch.uint8, torch.bfloat16, torch.bfloat16],
            1536,
        ),
    ],
)
def test_quantized_cache_layout_preserves_resource_order(quant_policy, resource_shapes, resource_dtypes, pool_bytes):
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0,
                               quant_policy=quant_policy)
    model_config = _make_model_config(dtype=torch.bfloat16)

    mem_pool, caches = CacheEngine.allocate_caches(num_blocks=2,
                                                   model_config=model_config,
                                                   cache_config=cache_config,
                                                   world_size=1,
                                                   device='cpu')

    assert tuple(mem_pool.shape) == (4, 2, pool_bytes)
    assert [tuple(cache.shape[2:]) for cache in caches] == resource_shapes
    assert [cache.dtype for cache in caches] == resource_dtypes
    assert CacheEngine.get_cache_block_size(cache_config, model_config) == 4 * pool_bytes


def test_split_kernel_blocks_scale_physical_allocation_and_sizing():
    cache_config = CacheConfig(max_batches=1,
                               block_size=128,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)
    model_config = _make_model_config(dtype=torch.bfloat16)

    mem_pool, caches = CacheEngine.allocate_caches(num_blocks=3,
                                                   model_config=model_config,
                                                   cache_config=cache_config,
                                                   world_size=1,
                                                   device='cpu')

    assert tuple(mem_pool.shape) == (4, 6, 4096)
    assert all(cache.shape[1] == 6 for cache in caches)
    assert CacheEngine.get_cache_block_size(cache_config, model_config) == mem_pool.numel() // 3


def test_legacy_custom_cache_layout_preserves_alignment_and_tuple_order():
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)
    model_config = _make_model_config(
        use_standard_kv_cache=False,
        cache_shapes=[((3, ), torch.float32), ((5, ), torch.float16)],
    )

    mem_pool, caches = CacheEngine.allocate_caches(num_blocks=2,
                                                   model_config=model_config,
                                                   cache_config=cache_config,
                                                   world_size=1,
                                                   device='cpu')

    assert tuple(mem_pool.shape) == (4, 2, 1536)
    assert [tuple(cache.shape) for cache in caches] == [
        (4, 2, 64, 3),
        (4, 2, 64, 5),
    ]
    assert [cache.dtype for cache in caches] == [torch.float32, torch.float16]
    assert [cache.storage_offset() for cache in caches] == [0, 384]
    assert CacheEngine.get_cache_block_size(cache_config, model_config) == 4 * 1536


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


def test_named_block_cache_specs_do_not_require_total_layer_count():
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)
    model_config = _make_model_config(
        use_standard_kv_cache=False,
        block_cache_specs=[
            BlockCacheSpec('r4', [1, 9], (40, ), torch.float32),
            BlockCacheSpec('r128', [7], (96, ), torch.float32),
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
            9: 1,
        },
        'r128': {
            7: 0,
        },
    }


def test_cache_engine_retains_cpu_allocation_owner():
    cache_engine = object.__new__(CacheEngine)
    cache_engine.cache_config = CacheConfig(max_batches=1,
                                            block_size=64,
                                            kernel_block_size=64,
                                            num_cpu_blocks=2,
                                            num_gpu_blocks=0)
    cache_engine.model_config = _make_model_config(dtype=torch.bfloat16)
    cache_engine.world_size = 1

    cache_engine.allocate_cpu_cache()

    assert isinstance(cache_engine.cpu_allocation, CacheAllocation)
    assert cache_engine.full_cpu_cache is cache_engine.cpu_allocation.pools[0].tensor


def test_cache_engine_reuses_retained_plan_for_device_and_cpu_allocations(monkeypatch):
    resources = (CacheResource('only', CacheDesc(shape=[3], dtype=torch.float32)), )
    allocations = []

    class RecordingLayout:

        def allocate(self, num_blocks, device):
            allocations.append((num_blocks, device))
            cache = torch.zeros((4, num_blocks, 3), dtype=torch.float32)
            return CacheAllocation(pools=(CachePool(cache, entry_axis=1), ), caches=(cache, ))

    cache_engine = object.__new__(CacheEngine)
    cache_engine.cache_config = CacheConfig(max_batches=1,
                                            block_size=128,
                                            kernel_block_size=64,
                                            num_cpu_blocks=3,
                                            num_gpu_blocks=3)
    cache_engine.model_config = _make_model_config(dtype=torch.bfloat16)
    cache_engine.world_size = 1
    cache_engine.block_cache_plan = BlockCachePlan(resources=resources,
                                                   layout=RecordingLayout(),
                                                   kernel_blocks_per_logical_block=2)

    def unexpected_compatibility_allocation(**kwargs):
        raise AssertionError('retained plans must not rebuild cache resources or layouts')

    monkeypatch.setattr(cache_engine, 'allocate_caches', unexpected_compatibility_allocation)

    gpu_cache = cache_engine.allocate_gpu_cache()
    cpu_cache = cache_engine.allocate_cpu_cache()

    assert allocations == [(6, 'cuda'), (6, 'cpu')]
    assert cache_engine._cache_names == ['only']
    assert cache_engine._block_cache_layer_maps == {}
    assert len(gpu_cache) == len(cpu_cache) == 4
    assert all(len(layer_cache) == 1 for layer_cache in (*gpu_cache, *cpu_cache))
    assert torch.equal(gpu_cache[0][0], cache_engine._gpu_cache_list[0][0])
    assert torch.equal(cpu_cache[0][0], cache_engine._cpu_cache_list[0][0])


def test_cache_engine_accepts_legacy_allocation_tuple(monkeypatch):
    mem_pool = torch.empty((2, 1, 8), dtype=torch.uint8)
    caches = [torch.empty((2, 1, 1)), torch.empty((2, 1, 1))]

    @classmethod
    def legacy_allocate(cls, **kwargs):
        return mem_pool, caches

    monkeypatch.setattr(CacheEngine, 'allocate_caches', legacy_allocate)
    cache_engine = object.__new__(CacheEngine)
    cache_engine.cache_config = CacheConfig(max_batches=1,
                                            block_size=64,
                                            kernel_block_size=64,
                                            num_cpu_blocks=1,
                                            num_gpu_blocks=0)
    cache_engine.model_config = _make_model_config(dtype=torch.bfloat16)
    cache_engine.world_size = 1
    resources = (CacheResource('native', CacheDesc(shape=[1], dtype=torch.float32)), )

    def unexpected_native_allocation(**kwargs):
        raise AssertionError('patched allocators must remain active')

    native_layout = SimpleNamespace(allocate=unexpected_native_allocation)
    cache_engine.block_cache_plan = BlockCachePlan(resources=resources,
                                                   layout=native_layout,
                                                   kernel_blocks_per_logical_block=1)

    cache_engine.allocate_cpu_cache()

    assert cache_engine.cpu_allocation is None
    assert cache_engine.full_cpu_cache is mem_pool
    assert CacheEngine.get_cache_block_size(cache_engine.cache_config, cache_engine.model_config) == mem_pool.numel()


def test_dlinfer_backend_uses_native_block_and_state_allocations(monkeypatch):
    monkeypatch.setattr(cache_engine_module, 'get_backend', lambda: DlinferOpsBackend)
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)
    model_config = _make_model_config(dtype=torch.bfloat16)
    state_shapes = [((2, 3), torch.float32), ((2, ), torch.float16)]

    block_allocation = CacheEngine.allocate_caches(num_blocks=2,
                                                   model_config=model_config,
                                                   cache_config=cache_config,
                                                   world_size=1,
                                                   device='cpu')
    state_allocation = StateCacheEngine.allocate_caches(
        num_caches=3,
        state_shapes=state_shapes,
        device='cpu',
    )

    assert len(block_allocation.pools) == len(block_allocation.caches) == 2
    assert [pool.entry_axis for pool in block_allocation.pools] == [1, 1]
    assert [tuple(cache.shape) for cache in block_allocation.caches] == [
        (4, 2, 64, 2, 8),
        (4, 2, 64, 2, 8),
    ]
    assert len(state_allocation.pools) == len(state_allocation.caches) == 2
    assert [pool.entry_axis for pool in state_allocation.pools] == [0, 0]
    assert all(cache.is_contiguous() for cache in (*block_allocation.caches, *state_allocation.caches))
    assert CacheEngine.get_cache_block_size(cache_config, model_config) == block_allocation.nbytes // 2
    assert StateCacheEngine.get_cache_state_size(state_shapes) == state_allocation.nbytes // 3


def test_cache_engine_swap_uses_each_pool_entry_axis(monkeypatch):
    cpu_first = torch.arange(6).view(3, 2)
    cpu_second = torch.arange(12).view(2, 3, 2)
    gpu_first = torch.zeros((4, 2), dtype=cpu_first.dtype)
    gpu_second = torch.zeros((2, 4, 2), dtype=cpu_second.dtype)

    cache_engine = object.__new__(CacheEngine)
    cache_engine.cpu_allocation = CacheAllocation(
        pools=(CachePool(cpu_first, entry_axis=0), CachePool(cpu_second, entry_axis=1)),
        caches=(),
    )
    cache_engine.gpu_allocation = CacheAllocation(
        pools=(CachePool(gpu_first, entry_axis=0), CachePool(gpu_second, entry_axis=1)),
        caches=(),
    )
    cache_engine._cpu_cache_list = []
    cache_engine._gpu_cache_list = []
    cache_engine._build_swap_pairs()
    cache_engine.cache_stream = object()
    recorded_streams = []
    cache_engine.events = SimpleNamespace(record=lambda stream: recorded_streams.append(stream))
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: nullcontext())

    cache_engine.swap_in({0: 2, 2: 1})

    assert torch.equal(gpu_first[2], cpu_first[0])
    assert torch.equal(gpu_first[1], cpu_first[2])
    assert torch.equal(gpu_second[:, 2], cpu_second[:, 0])
    assert torch.equal(gpu_second[:, 1], cpu_second[:, 2])

    gpu_first[3].fill_(21)
    gpu_second[:, 3].fill_(22)
    cache_engine.swap_out({3: 1})

    assert torch.equal(cpu_first[1], gpu_first[3])
    assert torch.equal(cpu_second[:, 1], gpu_second[:, 3])
    assert recorded_streams == [cache_engine.cache_stream, cache_engine.cache_stream]


def test_cache_engine_legacy_swap_uses_typed_block_views(monkeypatch):
    cpu_cache = torch.arange(12).view(2, 3, 2)
    gpu_cache = torch.zeros((2, 4, 2), dtype=cpu_cache.dtype)
    cache_engine = object.__new__(CacheEngine)
    cache_engine.cpu_allocation = None
    cache_engine.gpu_allocation = None
    cache_engine._cpu_cache_list = [cpu_cache]
    cache_engine._gpu_cache_list = [gpu_cache]
    cache_engine._build_swap_pairs()
    cache_engine.cache_stream = object()
    cache_engine.events = SimpleNamespace(record=lambda stream: None)
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: nullcontext())

    cache_engine.swap_in({1: 3})

    assert torch.equal(gpu_cache[:, 3], cpu_cache[:, 1])


def test_cache_engine_rejects_incompatible_swap_entry_axes():
    cache_engine = object.__new__(CacheEngine)
    cache_engine.cpu_allocation = CacheAllocation(
        pools=(CachePool(torch.empty((3, 2)), entry_axis=0), ),
        caches=(),
    )
    cache_engine.gpu_allocation = CacheAllocation(
        pools=(CachePool(torch.empty((2, 4)), entry_axis=1), ),
        caches=(),
    )
    cache_engine._cpu_cache_list = []
    cache_engine._gpu_cache_list = []

    with pytest.raises(RuntimeError, match='entry axes differ'):
        cache_engine._build_swap_pairs()


def test_layered_state_cache_specs_do_not_require_total_layer_count():
    state_specs = [StateCacheSpec('subset', (96, ), torch.float32, layer_ids=[1, 9])]
    state_shapes = [(spec.shape, spec.dtype) for spec in state_specs]

    allocation = StateCacheEngine.allocate_caches(num_caches=2,
                                                  state_shapes=state_shapes,
                                                  state_specs=state_specs,
                                                  device='cpu')
    mem_pool, caches = allocation

    assert isinstance(allocation, CacheAllocation)
    assert tuple(mem_pool.shape) == (2, 768)
    assert tuple(caches[0].shape) == (2, 2, 96)
    assert StateCacheEngine.get_cache_state_size(state_shapes, state_specs=state_specs) == 768
    assert StateCacheEngine._get_state_cache_layer_maps(state_specs) == {'subset': {1: 0, 9: 1}}


def test_state_cache_engine_accepts_legacy_allocation_tuple(monkeypatch):
    mem_pool = torch.empty((2, 8), dtype=torch.uint8)
    caches = [torch.empty((2, 2), dtype=torch.float32)]

    @staticmethod
    def legacy_allocate(num_caches, state_shapes, device):
        return mem_pool, caches

    monkeypatch.setattr(StateCacheEngine, 'allocate_caches', legacy_allocate)
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0,
                               num_state_caches=2,
                               states_shapes=[((2, ), torch.float32)])

    cache_engine = StateCacheEngine(cache_config)

    assert cache_engine.allocation is None
    assert cache_engine.mem_pool is mem_pool
    assert cache_engine.state_caches is caches
    assert cache_engine._state_entries[0][0] is caches[0]
    assert cache_engine._state_entries[0][1] == 0

    caches[0].fill_(1)
    cache_engine.init_caches(torch.tensor([1]), torch.tensor([True]))
    assert torch.count_nonzero(caches[0][1]) == 0
    caches[0][0].fill_(3)
    cache_engine.copy_caches(0, 1)
    assert torch.equal(caches[0][1], caches[0][0])

    assert StateCacheEngine.get_cache_state_size(cache_config.states_shapes) == mem_pool.numel()


def _make_multi_pool_state_allocation(num_caches: int, device: torch.device | str = 'cpu'):
    first = torch.zeros((num_caches, 2), dtype=torch.float32, device=device)
    second = torch.zeros((3, num_caches, 2), dtype=torch.float16, device=device)
    return CacheAllocation(
        pools=(CachePool(first, entry_axis=0), CachePool(second, entry_axis=1)),
        caches=(first, second),
    )


def test_state_cache_engine_accepts_multi_pool_layout(monkeypatch):
    layout = SimpleNamespace(
        allocate=lambda num_caches, device: _make_multi_pool_state_allocation(num_caches, device),
    )
    cache_backend = SimpleNamespace(build_state_layout=lambda resources: layout)
    ops_backend = SimpleNamespace(get_cache_backend=lambda: cache_backend)
    monkeypatch.setattr(cache_engine_module, 'get_backend', lambda: ops_backend)

    allocation = StateCacheEngine.allocate_caches(num_caches=2, state_shapes=[], device='cpu')

    assert [pool.entry_axis for pool in allocation.pools] == [0, 1]
    assert allocation.nbytes == 2 * (2 * 4 + 3 * 2 * 2)
    assert StateCacheEngine.get_cache_state_size([]) == 2 * 4 + 3 * 2 * 2


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

    negative_layer = [StateCacheSpec('bad', (1, ), torch.float32, layer_ids=[-1])]
    with pytest.raises(ValueError, match='non-negative'):
        StateCacheEngine.allocate_caches(num_caches=1,
                                         state_shapes=[((1, ), torch.float32)],
                                         state_specs=negative_layer,
                                         device='meta')

    empty_state_layers = [StateCacheSpec('empty', (1, ), torch.float32, layer_ids=[])]
    with pytest.raises(ValueError, match='must not be empty'):
        StateCacheEngine.allocate_caches(num_caches=1,
                                         state_shapes=[((1, ), torch.float32)],
                                         state_specs=empty_state_layers,
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
    cache_engine.allocation = StateCacheEngine.allocate_caches(
        num_caches=num_caches,
        state_shapes=cache_engine.cache_config.states_shapes,
        device='cpu',
    )
    cache_engine.mem_pool, cache_engine._state_caches = cache_engine.allocation
    cache_engine._state_entries = StateCacheEngine._build_state_entries(cache_engine.allocation,
                                                                         cache_engine._state_caches)
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
    cache_engine.mem_pool, cache_engine._state_caches = cache_engine.allocation
    cache_engine._state_entries = StateCacheEngine._build_state_entries(cache_engine.allocation,
                                                                         cache_engine._state_caches)
    return cache_engine


def test_state_cache_engine_init_caches_uses_each_pool_entry_axis():
    cache_engine = _make_multi_pool_state_cache_engine()
    first, second = cache_engine.state_caches
    first.fill_(1)
    second.fill_(1)

    cache_engine.init_caches(torch.tensor([1, 3]), torch.tensor([True, False]))

    assert torch.count_nonzero(first[1]) == 0
    assert torch.count_nonzero(second[:, 1]) == 0
    assert torch.all(first[3] == 1)
    assert torch.all(second[:, 3] == 1)


def test_state_cache_engine_copy_caches_uses_each_pool_entry_axis():
    cache_engine = _make_multi_pool_state_cache_engine()
    first, second = cache_engine.state_caches
    first[0].fill_(3)
    first[1].fill_(5)
    second[:, 0].fill_(7)
    second[:, 1].fill_(9)

    cache_engine.copy_caches((1, 0), (3, 2))

    assert torch.equal(first[2], first[0])
    assert torch.equal(first[3], first[1])
    assert torch.equal(second[:, 2], second[:, 0])
    assert torch.equal(second[:, 3], second[:, 1])


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

# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch.config import CacheConfig, ModelConfig
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol
from lmdeploy.pytorch.disagg.messages import MigrationExecutionBatch
from lmdeploy.pytorch.engine.cache_engine import CacheEngine
from lmdeploy.pytorch.engine.cache_engine.layout import CacheAllocation, CachePool
from lmdeploy.pytorch.engine.cache_engine.plan import BlockCachePlan, build_block_cache_plan
from lmdeploy.pytorch.engine.cache_engine.schema import (
    BlockCacheGeometry,
    BlockCacheRequest,
    BlockCacheRequestContext,
    CacheDesc,
    CacheTensorSpec,
    LayerRowMap,
    build_k_cache_desc,
    build_v_cache_desc,
)
from lmdeploy.pytorch.engine.cache_engine.view import NamedCacheView


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


def _get_native_logical_block_nbytes(cache_config: CacheConfig, model_config: ModelConfig, world_size: int = 1):
    plan = build_block_cache_plan(model_config, cache_config, world_size)
    return plan.logical_block_nbytes


def _allocate_native_block_caches(num_blocks: int, model_config: ModelConfig, cache_config: CacheConfig,
                                  world_size: int = 1, device: str = 'cpu') -> CacheAllocation:
    plan = build_block_cache_plan(model_config, cache_config, world_size)
    return plan.allocate(num_logical_blocks=num_blocks, device=device)


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

    k_desc = build_k_cache_desc(model_config, cache_config)
    v_desc = build_v_cache_desc(model_config, cache_config)

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
                               mla_kv_cache_dtype='fp8_ds_mla',
                               mla_index_topk=2048)
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)

    plan = build_block_cache_plan(model_config, cache_config, world_size=1)
    k_desc, v_desc = (spec.desc for spec in plan.tensor_specs)

    assert model_config.mla_kv_cache_dtype == 'fp8_ds_mla'
    assert cache_config.quant_policy == QuantPolicy.NONE
    assert k_desc.shape == [64, 1, 656]
    assert k_desc.dtype == torch.float8_e4m3fn
    assert v_desc.shape == [64, 1, 0]
    assert v_desc.dtype == torch.float8_e4m3fn

def test_build_cache_plan_requires_block_size_divisible_by_kernel_block_size():
    cache_config = CacheConfig(max_batches=1,
                               block_size=96,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)

    with pytest.raises(ValueError, match='block_size 96 must be divisible by kernel_block_size 64'):
        build_block_cache_plan(_make_model_config(), cache_config, world_size=1)


def test_build_cache_plan_collects_built_operator_requests():
    model_config = _make_model_config(use_standard_kv_cache=False)
    cache_config = CacheConfig(max_batches=1,
                               block_size=128,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)
    request_contexts = []

    def request_collector(context: BlockCacheRequestContext):
        request_contexts.append(context)
        request = BlockCacheRequest('operator_cache', (64, 5), torch.float16)
        return [request, request]

    plan = build_block_cache_plan(model_config,
                                  cache_config,
                                  world_size=1,
                                  request_collector=request_collector)

    assert request_contexts == [
        BlockCacheRequestContext(
            geometry=BlockCacheGeometry(logical_block_size=128, kernel_block_size=64))
    ]
    assert tuple(spec.name for spec in plan.tensor_specs) == ('operator_cache', )
    assert plan.tensor_specs[0].consumer_rows == (0, 1)
    assert plan.kernel_blocks_per_logical_block == 2

def test_build_cache_plan_combines_standard_kv_and_operator_requests():
    model_config = _make_model_config(use_standard_kv_cache=True)
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)
    request = BlockCacheRequest(
        'operator_cache',
        (64, 5),
        torch.float16,
        per_row_contiguous=True,
    )

    plan = build_block_cache_plan(
        model_config,
        cache_config,
        world_size=1,
        request_collector=lambda context: [request, request],
    )
    allocation = plan.allocate(num_logical_blocks=2, device='cpu')

    assert tuple(spec.name for spec in plan.tensor_specs) == ('k_cache', 'v_cache', 'operator_cache')
    assert plan.model_cache_indices == (0, 1)
    assert len(allocation.pools) == 2
    assert allocation.tensor_views[2].is_contiguous()
    assert allocation.tensor_views[2][0].is_contiguous()


def test_operator_cache_requests_cannot_shadow_standard_cache_names():
    model_config = _make_model_config(use_standard_kv_cache=True)
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)
    request = BlockCacheRequest('k_cache', (64, 5), torch.float16)

    with pytest.raises(ValueError, match='cannot mix plain and consumer tensor specs'):
        build_block_cache_plan(model_config,
                                     cache_config,
                                     world_size=1,
                                     request_collector=lambda context: [request])


def test_mixed_cache_plan_keeps_named_tensor_out_of_model_layer_tuples():
    model_config = _make_model_config(use_standard_kv_cache=True)
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=2,
                               num_gpu_blocks=2)
    request = BlockCacheRequest(
        'operator_cache',
        (64, 5),
        torch.float16,
        per_row_contiguous=True,
    )
    plan = build_block_cache_plan(
        model_config,
        cache_config,
        world_size=1,
        request_collector=lambda context: [request, request],
    )

    class CpuLayout:

        def allocate(self, num_blocks, device):
            return plan.layout.allocate(num_blocks, device='cpu')

    cache_engine = object.__new__(CacheEngine)
    cache_engine.cache_config = cache_config
    cache_engine.block_cache_plan = BlockCachePlan(tensor_specs=plan.tensor_specs,
                                                   layout=CpuLayout(),
                                                   kernel_blocks_per_logical_block=1)

    layer_caches = cache_engine.allocate_gpu_cache()
    cache_engine.allocate_cpu_cache()
    cache_engine._build_swap_pairs()

    assert len(layer_caches) == model_config.num_layers
    assert all(len(layer_cache) == 2 for layer_cache in layer_caches)
    assert torch.equal(layer_caches[0][0], cache_engine.gpu_allocation.tensor_views[0][0])
    assert torch.equal(layer_caches[0][1], cache_engine.gpu_allocation.tensor_views[1][0])
    assert len(cache_engine.gpu_allocation.pools) == 2
    assert len(cache_engine._swap_in_pairs) == 2
    assert all(entry_axis == 1 for _, _, entry_axis in cache_engine._swap_in_pairs)
    assert isinstance(cache_engine.block_caches, NamedCacheView)
    assert cache_engine.block_caches['operator_cache'].is_contiguous()
    assert torch.equal(cache_engine.block_caches['operator_cache'][1], cache_engine.gpu_allocation.tensor_views[2][1])
    assert torch.equal(cache_engine.block_caches.row('operator_cache', 1),
                       cache_engine.gpu_allocation.tensor_views[2][1])


def test_heterogeneous_operator_cache_rows_resolve_physical_tensors():
    model_config = _make_model_config(use_standard_kv_cache=False)
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=2)
    narrow = BlockCacheRequest('operator_cache', (64, 3), torch.float16)
    wide = BlockCacheRequest('operator_cache', (64, 5), torch.float16, per_row_contiguous=True)
    plan = build_block_cache_plan(
        model_config,
        cache_config,
        world_size=1,
        request_collector=lambda context: [narrow, wide, narrow],
    )

    class CpuLayout:

        def allocate(self, num_blocks, device):
            return plan.layout.allocate(num_blocks, device='cpu')

    cache_engine = object.__new__(CacheEngine)
    cache_engine.cache_config = cache_config
    cache_engine.block_cache_plan = BlockCachePlan(tensor_specs=plan.tensor_specs,
                                                   layout=CpuLayout(),
                                                   kernel_blocks_per_logical_block=1)

    assert cache_engine.allocate_gpu_cache() == []
    block_caches = cache_engine.block_caches
    cache_tensors = cache_engine.gpu_allocation.tensor_views

    assert [spec.consumer_rows for spec in plan.tensor_specs] == [(0, 2), (1, )]
    assert len(cache_engine.gpu_allocation.pools) == 2
    assert cache_tensors[1].is_contiguous()
    assert torch.equal(block_caches.row('operator_cache', 0), cache_tensors[0][0])
    assert torch.equal(block_caches.row('operator_cache', 1), cache_tensors[1][0])
    assert torch.equal(block_caches.row('operator_cache', 2), cache_tensors[0][1])
    with pytest.raises(RuntimeError, match='multiple physical tensors'):
        block_caches['operator_cache']
    with pytest.raises(RuntimeError, match='Consumer row 3 does not own cache'):
        block_caches.row('operator_cache', 3)


def test_standard_cache_layout_preserves_pool_bytes_strides_and_tensor_order():
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)
    model_config = _make_model_config(dtype=torch.bfloat16)
    allocation = _allocate_native_block_caches(num_blocks=3,
                                               model_config=model_config,
                                               cache_config=cache_config)
    mem_pool = allocation.pools[0].tensor
    caches = allocation.tensor_views

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
    assert _get_native_logical_block_nbytes(cache_config, model_config) == mem_pool.numel() // 3


@pytest.mark.parametrize(
    ('quant_policy', 'expected_shapes', 'expected_dtypes', 'pool_bytes'),
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
        (
            QuantPolicy.FP8,
            [(64, 2, 8), (64, 2, 8)],
            [torch.float8_e4m3fn, torch.float8_e4m3fn],
            2048,
        ),
        (
            QuantPolicy.FP8_E5M2,
            [(64, 2, 8), (64, 2, 8)],
            [torch.float8_e5m2, torch.float8_e5m2],
            2048,
        ),
    ],
)
def test_quantized_cache_layout_preserves_tensor_order(quant_policy, expected_shapes, expected_dtypes, pool_bytes):
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0,
                               quant_policy=quant_policy)
    model_config = _make_model_config(dtype=torch.bfloat16)

    allocation = _allocate_native_block_caches(num_blocks=2,
                                               model_config=model_config,
                                               cache_config=cache_config)
    mem_pool = allocation.pools[0].tensor
    caches = allocation.tensor_views

    assert tuple(mem_pool.shape) == (4, 2, pool_bytes)
    assert [tuple(cache.shape[2:]) for cache in caches] == expected_shapes
    assert [cache.dtype for cache in caches] == expected_dtypes
    assert _get_native_logical_block_nbytes(cache_config, model_config) == 4 * pool_bytes


def test_split_kernel_blocks_scale_physical_allocation_and_sizing():
    cache_config = CacheConfig(max_batches=1,
                               block_size=128,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)
    model_config = _make_model_config(dtype=torch.bfloat16)

    allocation = _allocate_native_block_caches(num_blocks=3,
                                               model_config=model_config,
                                               cache_config=cache_config)
    mem_pool = allocation.pools[0].tensor
    caches = allocation.tensor_views

    assert tuple(mem_pool.shape) == (4, 6, 4096)
    assert all(cache.shape[1] == 6 for cache in caches)
    assert _get_native_logical_block_nbytes(cache_config, model_config) == mem_pool.numel() // 3

def test_pd_migration_rejects_split_kernel_blocks():
    cache_engine = object.__new__(CacheEngine)
    cache_engine.cache_config = CacheConfig(max_batches=1,
                                            block_size=96,
                                            kernel_block_size=64,
                                            num_cpu_blocks=0,
                                            num_gpu_blocks=0)
    cache_engine._pd_cache_pool_infos = None
    migration_inputs = MigrationExecutionBatch(protocol=MigrationProtocol.RDMA, requests=[])

    with pytest.raises(RuntimeError, match='PD migration does not support block_size != kernel_block_size'):
        asyncio.run(cache_engine.migrate(migration_inputs))


def test_cache_engine_reuses_retained_plan_for_device_and_cpu_allocations():
    tensor_specs = (CacheTensorSpec('only', CacheDesc(shape=[3], dtype=torch.float32)), )
    allocations = []

    class RecordingLayout:

        def allocate(self, num_blocks, device):
            allocations.append((num_blocks, device))
            cache = torch.zeros((4, num_blocks, 3), dtype=torch.float32)
            return CacheAllocation(pools=(CachePool(cache, entry_axis=1), ), tensor_views=(cache, ))

    cache_engine = object.__new__(CacheEngine)
    cache_engine.cache_config = CacheConfig(max_batches=1,
                                            block_size=128,
                                            kernel_block_size=64,
                                            num_cpu_blocks=3,
                                            num_gpu_blocks=3)
    cache_engine.block_cache_plan = BlockCachePlan(tensor_specs=tensor_specs,
                                                   layout=RecordingLayout(),
                                                   kernel_blocks_per_logical_block=2)

    gpu_cache = cache_engine.allocate_gpu_cache()
    cpu_cache = cache_engine.allocate_cpu_cache()

    assert allocations == [(6, 'cuda'), (6, 'cpu')]
    assert type(cache_engine.block_caches) is dict
    assert set(cache_engine.block_caches) == {'only'}
    assert len(gpu_cache) == len(cpu_cache) == 4
    assert all(len(layer_cache) == 1 for layer_cache in (*gpu_cache, *cpu_cache))
    assert torch.equal(gpu_cache[0][0], cache_engine.gpu_allocation.tensor_views[0][0])
    assert torch.equal(cpu_cache[0][0], cache_engine.cpu_allocation.tensor_views[0][0])

def test_cache_engine_swap_expands_logical_blocks_for_each_pool_entry_axis(monkeypatch):
    cpu_first = torch.arange(12).view(6, 2)
    cpu_second = torch.arange(24).view(2, 6, 2)
    gpu_first = torch.zeros((8, 2), dtype=cpu_first.dtype)
    gpu_second = torch.zeros((2, 8, 2), dtype=cpu_second.dtype)

    cache_engine = object.__new__(CacheEngine)
    cache_engine.block_cache_plan = SimpleNamespace(kernel_blocks_per_logical_block=2)
    cache_engine.cpu_allocation = CacheAllocation(
        pools=(CachePool(cpu_first, entry_axis=0), CachePool(cpu_second, entry_axis=1)),
        tensor_views=(),
    )
    cache_engine.gpu_allocation = CacheAllocation(
        pools=(CachePool(gpu_first, entry_axis=0), CachePool(gpu_second, entry_axis=1)),
        tensor_views=(),
    )
    cache_engine._build_swap_pairs()
    cache_engine.cache_stream = object()
    recorded_streams = []
    cache_engine.swap_event = SimpleNamespace(record=lambda stream: recorded_streams.append(stream))
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: nullcontext())

    cache_engine.swap_in({0: 2, 2: 1})

    assert torch.equal(gpu_first[4:6], cpu_first[0:2])
    assert torch.equal(gpu_first[2:4], cpu_first[4:6])
    assert torch.equal(gpu_second[:, 4:6], cpu_second[:, 0:2])
    assert torch.equal(gpu_second[:, 2:4], cpu_second[:, 4:6])

    gpu_first[6:8].fill_(21)
    gpu_second[:, 6:8].fill_(22)
    cache_engine.swap_out({3: 1})

    assert torch.equal(cpu_first[2:4], gpu_first[6:8])
    assert torch.equal(cpu_second[:, 2:4], gpu_second[:, 6:8])
    assert recorded_streams == [cache_engine.cache_stream, cache_engine.cache_stream]


def test_cache_engine_rejects_incompatible_swap_entry_axes():
    cache_engine = object.__new__(CacheEngine)
    cache_engine.cpu_allocation = CacheAllocation(
        pools=(CachePool(torch.empty((3, 2)), entry_axis=0), ),
        tensor_views=(),
    )
    cache_engine.gpu_allocation = CacheAllocation(
        pools=(CachePool(torch.empty((2, 4)), entry_axis=1), ),
        tensor_views=(),
    )

    with pytest.raises(RuntimeError, match='entry axes differ'):
        cache_engine._build_swap_pairs()


def test_deepseek_v4_caches_resolve_state_and_carry_block_view():
    from lmdeploy.pytorch.models.deepseek_v4 import V4Caches

    state_cache = torch.arange(24).view(2, 3, 4)
    block_cache = torch.arange(40).view(2, 5, 4)
    block_cache_view = NamedCacheView(
        (CacheTensorSpec('block', CacheDesc([5, 4], block_cache.dtype), consumer_rows=(0, 1)), ),
        (block_cache, ),
    )
    state_cache_view = NamedCacheView(
        (CacheTensorSpec('state',
                         CacheDesc([2, 3, 4], state_cache.dtype),
                         layer_rows=LayerRowMap.build('state', [1, 3])), ),
        (state_cache, ),
    )
    caches = V4Caches(
        named_state_caches=state_cache_view,
        block_caches=block_cache_view,
    )

    assert torch.equal(caches.state_cache('state', 3), state_cache[1])
    assert caches.block_caches is block_cache_view
    with pytest.raises(RuntimeError, match='does not own cache'):
        caches.state_cache('state', 2)

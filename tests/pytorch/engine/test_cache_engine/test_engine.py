# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import lmdeploy.pytorch.engine.cache_engine.plan as cache_plan_module
import lmdeploy.pytorch.engine.cache_engine.schema as cache_schema_module
import lmdeploy.pytorch.engine.cache_engine.state as state_cache_module
from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch.backends.dlinfer.op_backend import DlinferOpsBackend
from lmdeploy.pytorch.config import BlockCacheSpec, CacheConfig, ModelConfig
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol
from lmdeploy.pytorch.disagg.messages import MigrationExecutionBatch
from lmdeploy.pytorch.engine.cache_engine import CacheEngine, StateCacheEngine
from lmdeploy.pytorch.engine.cache_engine.engine import _resolve_legacy_kv_cache_dtype
from lmdeploy.pytorch.engine.cache_engine.layout import CacheAllocation, CachePool
from lmdeploy.pytorch.engine.cache_engine.plan import BlockCachePlan
from lmdeploy.pytorch.engine.cache_engine.schema import (
    BlockCacheGeometry,
    BlockCacheRequest,
    BlockCacheRequestContext,
    CacheDesc,
    CacheTensorSpec,
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


@pytest.mark.parametrize(
    ('quant_policy', 'device_type', 'expected_dtype'),
    [
        (QuantPolicy.NONE, 'cuda', torch.float16),
        (QuantPolicy.INT8, 'cuda', torch.uint8),
        (QuantPolicy.INT8, 'npu', torch.int8),
        (QuantPolicy.FP8, 'cuda', torch.float8_e4m3fn),
    ],
)
def test_resolve_legacy_kv_cache_dtype(quant_policy, device_type, expected_dtype):
    model_config = SimpleNamespace(dtype=torch.float16,
                                   use_mla_fp8_cache=False,
                                   mla_kv_cache_dtype=None,
                                   mla_index_topk=None)
    cache_config = SimpleNamespace(quant_policy=quant_policy, device_type=device_type)

    assert _resolve_legacy_kv_cache_dtype(model_config, cache_config) is expected_dtype


def test_sparse_mla_legacy_dtype_bypasses_generic_quantization():
    model_config = SimpleNamespace(dtype=torch.bfloat16,
                                   use_mla_fp8_cache=True,
                                   mla_kv_cache_dtype='fp8_ds_mla',
                                   mla_index_topk=2048)
    cache_config = SimpleNamespace(quant_policy=QuantPolicy.FP8, device_type='npu')

    assert _resolve_legacy_kv_cache_dtype(model_config, cache_config) is torch.float8_e4m3fn


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
    CacheEngine.get_logical_block_nbytes(cache_config, model_config)

    assert model_config.mla_kv_cache_dtype == 'fp8_ds_mla'

    k_desc = CacheEngine.get_k_cache_desc(model_config, cache_config)
    v_desc = CacheEngine.get_v_cache_desc(model_config, cache_config)

    assert k_desc.shape == [64, 1, 656]
    assert k_desc.dtype == torch.float8_e4m3fn
    assert v_desc.shape == [64, 1, 0]
    assert v_desc.dtype == torch.float8_e4m3fn


def test_retained_plan_uses_finalized_sparse_mla_dtype():
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

    plan = CacheEngine.build_cache_plan(model_config, cache_config, world_size=1)

    assert model_config.mla_kv_cache_dtype == 'fp8_ds_mla'
    assert plan.tensor_specs[0].desc.dtype == torch.float8_e4m3fn


def test_cache_plan_finalizes_sparse_mla_policy_before_request_collection():
    model_config = _make_model_config(dtype=torch.bfloat16,
                                      mla_index_topk=2048,
                                      mla_kv_cache_dtype='bfloat16')
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0,
                               quant_policy=QuantPolicy.FP8)
    collected_dtypes = []

    def collect_requests(context):
        collected_dtypes.append(model_config.mla_kv_cache_dtype)
        return None

    CacheEngine.build_cache_plan(model_config,
                                 cache_config,
                                 world_size=1,
                                 request_collector=collect_requests)

    assert collected_dtypes == ['fp8_ds_mla']


@pytest.mark.parametrize('quant_policy',
                         [QuantPolicy.INT4, QuantPolicy.INT8, QuantPolicy.FP8_E5M2, QuantPolicy.TURBO_QUANT])
def test_sparse_mla_rejects_other_cache_policies(quant_policy):
    model_config = SimpleNamespace(mla_index_topk=2048)
    cache_config = SimpleNamespace(quant_policy=quant_policy)

    with pytest.raises(ValueError, match='Sparse MLA does not support quant_policy'):
        CacheEngine.get_logical_block_nbytes(cache_config, model_config)


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


def test_build_cache_plan_collects_built_operator_requests():
    model_config = _make_model_config(use_standard_kv_cache=False,
                                      block_cache_specs=[
                                          BlockCacheSpec('fallback', [0], (64, 3), torch.float16),
                                      ])
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

    plan = CacheEngine.build_cache_plan(model_config,
                                        cache_config,
                                        world_size=1,
                                        request_collector=request_collector)

    assert request_contexts == [
        BlockCacheRequestContext(
            geometry=BlockCacheGeometry(logical_block_size=128, kernel_block_size=64))
    ]
    assert plan.cache_names == ('operator_cache', )
    assert plan.tensor_specs[0].consumer_rows == (0, 1)
    assert plan.kernel_blocks_per_logical_block == 2


def test_empty_built_operator_requests_are_authoritative_for_custom_caches():
    model_config = _make_model_config(use_standard_kv_cache=False,
                                      block_cache_specs=[
                                          BlockCacheSpec('fallback', [0], (64, 3), torch.float16),
                                      ])
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)

    plan = CacheEngine.build_cache_plan(model_config,
                                        cache_config,
                                        world_size=1,
                                        request_collector=lambda context: ())

    assert plan.cache_names == ()


def test_absent_built_operator_requester_uses_config_fallback():
    model_config = _make_model_config(use_standard_kv_cache=False,
                                      block_cache_specs=[
                                          BlockCacheSpec('fallback', [0], (64, 3), torch.float16),
                                      ])
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)

    plan = CacheEngine.build_cache_plan(model_config,
                                        cache_config,
                                        world_size=1,
                                        request_collector=lambda context: None)

    assert plan.cache_names == ('fallback', )


def test_built_operator_request_collection_rejects_patched_allocator(monkeypatch):
    model_config = _make_model_config(use_standard_kv_cache=False)
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)

    @classmethod
    def legacy_allocate(cls, **kwargs):
        raise AssertionError('the patched allocator must not be called')

    monkeypatch.setattr(CacheEngine, 'allocate_caches', legacy_allocate)

    with pytest.raises(RuntimeError, match='request collection requires the native'):
        CacheEngine.build_cache_plan(model_config,
                                     cache_config,
                                     world_size=1,
                                     request_collector=lambda context: ())


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

    plan = CacheEngine.build_cache_plan(
        model_config,
        cache_config,
        world_size=1,
        request_collector=lambda context: [request, request],
    )
    allocation = plan.allocate(num_logical_blocks=2, device='cpu')

    assert plan.cache_names == ('k_cache', 'v_cache', 'operator_cache')
    assert plan.legacy_cache_indices == (0, 1)
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
        CacheEngine.build_cache_plan(model_config,
                                     cache_config,
                                     world_size=1,
                                     request_collector=lambda context: [request])


def test_mixed_cache_plan_keeps_named_tensor_out_of_legacy_layer_tuples():
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
    plan = CacheEngine.build_cache_plan(
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
    cache_engine.model_config = model_config
    cache_engine.world_size = 1
    cache_engine.block_cache_plan = BlockCachePlan(tensor_specs=plan.tensor_specs,
                                                   layout=CpuLayout(),
                                                   kernel_blocks_per_logical_block=1)

    layer_caches = cache_engine.allocate_gpu_cache()
    cache_engine.allocate_cpu_cache()
    cache_engine._build_swap_pairs()

    assert len(layer_caches) == model_config.num_layers
    assert all(len(layer_cache) == 2 for layer_cache in layer_caches)
    assert torch.equal(layer_caches[0][0], cache_engine._gpu_cache_list[0][0])
    assert torch.equal(layer_caches[0][1], cache_engine._gpu_cache_list[1][0])
    assert len(cache_engine.gpu_allocation.pools) == 2
    assert len(cache_engine._swap_in_pairs) == 2
    assert all(entry_axis == 1 for _, _, entry_axis in cache_engine._swap_in_pairs)
    assert isinstance(cache_engine.block_caches, NamedCacheView)
    assert cache_engine.block_caches['operator_cache'].is_contiguous()
    assert torch.equal(cache_engine.block_caches['operator_cache'][1], cache_engine._gpu_cache_list[2][1])
    assert torch.equal(cache_engine.block_caches.row('operator_cache', 1), cache_engine._gpu_cache_list[2][1])


def test_heterogeneous_operator_cache_rows_resolve_physical_tensors():
    model_config = _make_model_config(use_standard_kv_cache=False)
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=2)
    narrow = BlockCacheRequest('operator_cache', (64, 3), torch.float16)
    wide = BlockCacheRequest('operator_cache', (64, 5), torch.float16, per_row_contiguous=True)
    plan = CacheEngine.build_cache_plan(
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
    cache_engine.model_config = model_config
    cache_engine.world_size = 1
    cache_engine.block_cache_plan = BlockCachePlan(tensor_specs=plan.tensor_specs,
                                                   layout=CpuLayout(),
                                                   kernel_blocks_per_logical_block=1)

    assert cache_engine.allocate_gpu_cache() == []
    block_caches = cache_engine.block_caches

    assert [spec.consumer_rows for spec in plan.tensor_specs] == [(0, 2), (1, )]
    assert len(cache_engine.gpu_allocation.pools) == 2
    assert cache_engine._gpu_cache_list[1].is_contiguous()
    assert torch.equal(block_caches.row('operator_cache', 0), cache_engine._gpu_cache_list[0][0])
    assert torch.equal(block_caches.row('operator_cache', 1), cache_engine._gpu_cache_list[1][0])
    assert torch.equal(block_caches.row('operator_cache', 2), cache_engine._gpu_cache_list[0][1])
    with pytest.raises(RuntimeError, match='multiple physical tensors'):
        block_caches['operator_cache']
    with pytest.raises(RuntimeError, match='Consumer row 3 does not own cache'):
        block_caches.row('operator_cache', 3)


def test_heterogeneous_config_cache_layers_resolve_physical_tensors():
    model_config = _make_model_config(
        use_standard_kv_cache=False,
        block_cache_specs=[
            BlockCacheSpec('layer_cache', [0, 2], (64, 3), torch.float16),
            BlockCacheSpec('layer_cache', [1], (64, 5), torch.float16),
        ],
    )
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=2)
    plan = CacheEngine.build_cache_plan(model_config, cache_config, world_size=1)
    allocation = plan.allocate(num_logical_blocks=2, device='cpu')
    block_caches = NamedCacheView.from_specs(plan.tensor_specs, allocation.tensor_views)

    assert torch.equal(block_caches.layer('layer_cache', 0), allocation.tensor_views[0][0])
    assert torch.equal(block_caches.layer('layer_cache', 1), allocation.tensor_views[1][0])
    assert torch.equal(block_caches.layer('layer_cache', 2), allocation.tensor_views[0][1])
    with pytest.raises(RuntimeError, match='multiple physical tensors'):
        block_caches['layer_cache']


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
    assert CacheEngine.get_logical_block_nbytes(cache_config, model_config) == mem_pool.numel() // 3


@pytest.mark.parametrize(
    ('quant_policy', 'cache_shapes', 'cache_dtypes', 'pool_bytes'),
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
def test_quantized_cache_layout_preserves_tensor_order(quant_policy, cache_shapes, cache_dtypes, pool_bytes):
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
    assert [tuple(cache.shape[2:]) for cache in caches] == cache_shapes
    assert [cache.dtype for cache in caches] == cache_dtypes
    assert CacheEngine.get_logical_block_nbytes(cache_config, model_config) == 4 * pool_bytes


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
    assert CacheEngine.get_logical_block_nbytes(cache_config, model_config) == mem_pool.numel() // 3


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
    assert CacheEngine.get_logical_block_nbytes(cache_config, model_config) == 4 * 1536


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
    assert CacheEngine.get_logical_block_nbytes(cache_config, model_config) == 1024
    assert CacheEngine._get_block_rows_by_layer(model_config) == {
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
    cache_engine.model_config = _make_model_config(dtype=torch.bfloat16)
    cache_engine.world_size = 1
    cache_engine.block_cache_plan = BlockCachePlan(tensor_specs=tensor_specs,
                                                   layout=RecordingLayout(),
                                                   kernel_blocks_per_logical_block=2)

    def unexpected_compatibility_allocation(**kwargs):
        raise AssertionError('retained plans must not rebuild cache tensor specs or layouts')

    monkeypatch.setattr(cache_engine, 'allocate_caches', unexpected_compatibility_allocation)

    gpu_cache = cache_engine.allocate_gpu_cache()
    cpu_cache = cache_engine.allocate_cpu_cache()

    assert allocations == [(6, 'cuda'), (6, 'cpu')]
    assert cache_engine._legacy_gpu_cache_pool is None
    assert cache_engine.full_gpu_cache is cache_engine.gpu_allocation.pools[0].tensor
    assert cache_engine.full_cpu_cache is cache_engine.cpu_allocation.pools[0].tensor
    assert cache_engine._block_cache_names == ['only']
    assert cache_engine._block_rows_by_layer == {}
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
    tensor_specs = (CacheTensorSpec('native', CacheDesc(shape=[1], dtype=torch.float32)), )

    def unexpected_native_allocation(**kwargs):
        raise AssertionError('patched allocators must remain active')

    native_layout = SimpleNamespace(allocate=unexpected_native_allocation)
    cache_engine.block_cache_plan = BlockCachePlan(tensor_specs=tensor_specs,
                                                   layout=native_layout,
                                                   kernel_blocks_per_logical_block=1)

    cache_engine.allocate_gpu_cache()
    cache_engine.allocate_cpu_cache()

    assert cache_engine.gpu_allocation is None
    assert cache_engine.cpu_allocation is None
    assert cache_engine._legacy_gpu_cache_pool is mem_pool
    assert cache_engine.full_gpu_cache is mem_pool
    assert cache_engine.full_cpu_cache is mem_pool
    assert CacheEngine.get_logical_block_nbytes(cache_engine.cache_config,
                                                cache_engine.model_config) == mem_pool.numel()


def test_dlinfer_backend_uses_native_block_and_state_allocations(monkeypatch):
    monkeypatch.setattr(cache_plan_module, 'get_backend', lambda: DlinferOpsBackend)
    monkeypatch.setattr(cache_schema_module, 'get_backend', lambda: DlinferOpsBackend)
    monkeypatch.setattr(state_cache_module, 'get_backend', lambda: DlinferOpsBackend)
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

    assert len(block_allocation.pools) == len(block_allocation.tensor_views) == 2
    assert [pool.entry_axis for pool in block_allocation.pools] == [1, 1]
    assert [tuple(cache.shape) for cache in block_allocation.tensor_views] == [
        (4, 2, 64, 2, 8),
        (4, 2, 64, 2, 8),
    ]
    assert len(state_allocation.pools) == len(state_allocation.tensor_views) == 2
    assert [pool.entry_axis for pool in state_allocation.pools] == [0, 0]
    assert all(cache.is_contiguous()
               for cache in (*block_allocation.tensor_views, *state_allocation.tensor_views))
    assert CacheEngine.get_logical_block_nbytes(cache_config, model_config) == block_allocation.nbytes // 2
    assert StateCacheEngine.get_state_slot_nbytes(state_shapes) == state_allocation.nbytes // 3


def test_cache_engine_swap_uses_each_pool_entry_axis(monkeypatch):
    cpu_first = torch.arange(6).view(3, 2)
    cpu_second = torch.arange(12).view(2, 3, 2)
    gpu_first = torch.zeros((4, 2), dtype=cpu_first.dtype)
    gpu_second = torch.zeros((2, 4, 2), dtype=cpu_second.dtype)

    cache_engine = object.__new__(CacheEngine)
    cache_engine.cpu_allocation = CacheAllocation(
        pools=(CachePool(cpu_first, entry_axis=0), CachePool(cpu_second, entry_axis=1)),
        tensor_views=(),
    )
    cache_engine.gpu_allocation = CacheAllocation(
        pools=(CachePool(gpu_first, entry_axis=0), CachePool(gpu_second, entry_axis=1)),
        tensor_views=(),
    )
    cache_engine._cpu_cache_list = []
    cache_engine._gpu_cache_list = []
    cache_engine._build_swap_pairs()
    cache_engine.cache_stream = object()
    recorded_streams = []
    cache_engine.swap_event = SimpleNamespace(record=lambda stream: recorded_streams.append(stream))
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
    cache_engine.swap_event = SimpleNamespace(record=lambda stream: None)
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: nullcontext())

    cache_engine.swap_in({1: 3})

    assert torch.equal(gpu_cache[:, 3], cpu_cache[:, 1])


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
    cache_engine._cpu_cache_list = []
    cache_engine._gpu_cache_list = []

    with pytest.raises(RuntimeError, match='entry axes differ'):
        cache_engine._build_swap_pairs()


def test_layer_scoped_block_cache_specs_reject_invalid_layer_ids():
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

    overlapping_specs = _make_model_config(
        use_standard_kv_cache=False,
        block_cache_specs=[
            BlockCacheSpec('overlap', [1], (1, ), torch.float32),
            BlockCacheSpec('overlap', [1, 2], (2, ), torch.float32),
        ],
    )
    with pytest.raises(ValueError, match='row 1 belongs to multiple tensor specs'):
        CacheEngine.build_cache_plan(overlapping_specs, cache_config, world_size=1)

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


def test_named_block_cache_property_returns_dict_without_layer_rows():
    block_cache_engine = object.__new__(CacheEngine)
    block_cache_engine._block_cache_names = ['k_cache']
    block_cache_engine._gpu_cache_list = [torch.empty(1)]
    block_cache_engine._block_rows_by_layer = {}

    assert type(block_cache_engine.block_caches) is dict

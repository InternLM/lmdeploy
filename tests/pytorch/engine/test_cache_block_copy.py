# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

import lmdeploy.pytorch.engine.cache_engine as cache_engine_module
from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch.backends.base import OpType
from lmdeploy.pytorch.backends.default.cache_block_copy import (
    DefaultCacheBlockCopyBuilder,
    DefaultCacheBlockCopyImpl,
)
from lmdeploy.pytorch.backends.default.op_backend import DefaultOpsBackend
from lmdeploy.pytorch.config import CacheConfig, ModelConfig
from lmdeploy.pytorch.engine.cache_engine import CacheEngine


def test_default_backend_registers_cache_block_copy_builder():
    assert DefaultOpsBackend.get_layer_impl_builder(OpType.CacheBlockCopy) is DefaultCacheBlockCopyBuilder


def test_cache_engine_build_cache_block_copy_routes_stable_list_pools(monkeypatch):
    cache_config = CacheConfig(max_batches=1,
                               block_size=6,
                               kernel_block_size=2,
                               num_cpu_blocks=0,
                               num_gpu_blocks=4,
                               enable_prefix_caching=True,
                               states_shapes=[((2, 3), torch.float32)])
    physical_pages = cache_config.num_gpu_blocks * (cache_config.block_size // cache_config.kernel_block_size)
    packed_pools = [
        torch.empty((2, physical_pages, 8), dtype=torch.uint8),
        torch.empty((2, 3, physical_pages, 16), dtype=torch.uint8),
    ]
    build_calls = []
    requested_ops = []
    copy_impl = object()

    class _RecordingBuilder:

        @staticmethod
        def build(**kwargs):
            build_calls.append(kwargs)
            return copy_impl

    class _RecordingBackend:

        @staticmethod
        def get_layer_impl_builder(op_type):
            requested_ops.append(op_type)
            return _RecordingBuilder

    monkeypatch.setattr(cache_engine_module, 'get_backend', lambda: _RecordingBackend())
    cache_engine = object.__new__(CacheEngine)
    cache_engine.cache_config = cache_config
    cache_engine.full_gpu_cache = packed_pools

    cache_engine._build_cache_block_copy()

    assert requested_ops == [OpType.CacheBlockCopy]
    assert len(build_calls) == 1
    build_args = build_calls[0]
    assert [id(pool) for pool in build_args['packed_caches']] == [id(pool) for pool in packed_pools]
    assert build_args['num_logical_blocks'] == cache_config.num_gpu_blocks
    assert build_args['pages_per_block'] == 3
    assert cache_engine._cache_block_copy_device == packed_pools[0].device
    assert cache_engine._cache_block_copy_impl is copy_impl


@pytest.mark.parametrize(
    ('enable_prefix_caching', 'states_shapes'),
    [
        pytest.param(False, [((2, 3), torch.float32)], id='prefix-cache-disabled'),
        pytest.param(True, [], id='non-ssm'),
    ],
)
def test_cache_engine_build_cache_block_copy_skips_disabled(monkeypatch, enable_prefix_caching,
                                                            states_shapes):
    cache_config = CacheConfig(max_batches=1,
                               block_size=4,
                               kernel_block_size=2,
                               num_cpu_blocks=0,
                               num_gpu_blocks=1,
                               enable_prefix_caching=enable_prefix_caching,
                               states_shapes=states_shapes)
    cache_engine = object.__new__(CacheEngine)
    cache_engine.cache_config = cache_config
    cache_engine.full_gpu_cache = object()
    monkeypatch.setattr(cache_engine_module, 'get_backend',
                        lambda: (_ for _ in ()).throw(AssertionError('disabled copy must not request a backend')))

    cache_engine._build_cache_block_copy()

    assert cache_engine._cache_block_copy_device is None
    assert cache_engine._cache_block_copy_impl is None


def _make_cache_engine(quant_policy: QuantPolicy = QuantPolicy.NONE,
                       cache_shapes: list[tuple[list[int], torch.dtype]] | None = None,
                       num_blocks: int = 5):
    cache_config = CacheConfig(max_batches=1,
                               block_size=4,
                               kernel_block_size=2,
                               num_cpu_blocks=0,
                               num_gpu_blocks=num_blocks,
                               quant_policy=quant_policy)
    model_config = ModelConfig(hidden_size=16,
                               num_layers=2,
                               num_attention_heads=2,
                               num_key_value_heads=1,
                               bos_token_id=1,
                               eos_token_id=[2],
                               head_dim=8,
                               cache_shapes=cache_shapes or [])
    cache_engine = object.__new__(CacheEngine)
    cache_engine.cache_config = cache_config
    cache_engine.model_config = model_config
    cache_engine.full_gpu_cache, caches = CacheEngine.allocate_caches(num_blocks=num_blocks,
                                                                      model_config=model_config,
                                                                      cache_config=cache_config,
                                                                      world_size=1,
                                                                      device='cpu')
    cache_engine._cache_block_copy_device = cache_engine.full_gpu_cache.device
    cache_engine._cache_block_copy_impl = DefaultCacheBlockCopyImpl(
        packed_caches=[cache_engine.full_gpu_cache],
        num_logical_blocks=num_blocks,
        pages_per_block=2,
        blocks_per_chunk=num_blocks,
    )
    return cache_engine, caches


class _RecordingCacheBlockCopy:

    def __init__(self):
        self.calls = []

    def forward(self, src_block_offsets, dst_block_offsets):
        self.calls.append((src_block_offsets.clone(), dst_block_offsets.clone()))


def _make_copy_plan(src_block_offsets, dst_block_offsets):
    return torch.tensor((src_block_offsets, dst_block_offsets), dtype=torch.long)


def test_default_cache_block_copy_chunks_and_reuses_workspaces():
    packed_cache = torch.full((2, 3, 12, 8), 99, dtype=torch.uint8)
    copy_impl = DefaultCacheBlockCopyImpl(packed_caches=[packed_cache],
                                          num_logical_blocks=6,
                                          pages_per_block=2,
                                          blocks_per_chunk=2)
    logical_cache = packed_cache.unflatten(-2, (6, 2))
    src_blocks = torch.tensor([0, 1, 0])
    dst_blocks = torch.tensor([3, 4, 5])
    for block_id in set(src_blocks.tolist()):
        logical_cache.select(-3, block_id).fill_(block_id + 1)
    expected = logical_cache.index_select(-3, src_blocks).clone()

    copy_impl.forward(src_blocks, dst_blocks)

    assert torch.equal(logical_cache.index_select(-3, dst_blocks), expected)
    workspace_ptrs = tuple(workspace.data_ptr() for workspace in copy_impl._workspaces)
    copy_impl.forward(src_blocks, dst_blocks)
    assert tuple(workspace.data_ptr() for workspace in copy_impl._workspaces) == workspace_ptrs


def test_default_cache_block_copy_builder_uses_total_byte_budget(monkeypatch):
    packed_pools = [
        torch.empty((2, 12, 8), dtype=torch.uint8),
        torch.empty((2, 3, 12, 16), dtype=torch.uint8),
    ]
    bytes_per_block = sum(pool.numel() * pool.element_size() // 6 for pool in packed_pools)
    monkeypatch.setattr(DefaultCacheBlockCopyBuilder, '_TARGET_WORKSPACE_BYTES', bytes_per_block * 2)

    copy_impl = DefaultCacheBlockCopyBuilder.build(packed_caches=packed_pools,
                                                   num_logical_blocks=6,
                                                   pages_per_block=2)

    assert copy_impl.blocks_per_chunk == 2
    assert copy_impl._workspaces is None
    copy_impl.forward(torch.tensor([0]), torch.tensor([1]))
    assert sum(workspace.nbytes for workspace in copy_impl._workspaces) == bytes_per_block * 2


def test_default_cache_block_copy_does_not_allocate_workspace_for_empty_plan():
    packed_cache = torch.empty((2, 12, 8), dtype=torch.uint8)
    copy_impl = DefaultCacheBlockCopyBuilder.build(packed_caches=[packed_cache],
                                                   num_logical_blocks=6,
                                                   pages_per_block=2)

    copy_impl.forward(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))

    assert copy_impl._workspaces is None


def test_default_cache_block_copy_groups_kernel_pages_without_expanding_plan():
    pages_per_block = 3
    packed_cache = torch.full((2, 18, 10), 99, dtype=torch.uint8)
    copy_impl = DefaultCacheBlockCopyImpl(packed_caches=[packed_cache],
                                          num_logical_blocks=6,
                                          pages_per_block=pages_per_block,
                                          blocks_per_chunk=2)
    src_blocks = torch.tensor([0, 2, 0])
    dst_blocks = torch.tensor([3, 1, 4])
    logical_cache = packed_cache.unflatten(-2, (6, pages_per_block))
    for block_id in set(src_blocks.tolist()):
        logical_cache.select(-3, block_id).fill_(block_id + 1)
    expected = logical_cache.index_select(-3, src_blocks).clone()

    copy_impl.forward(src_blocks, dst_blocks)

    assert torch.equal(logical_cache.index_select(-3, dst_blocks), expected)


@pytest.mark.parametrize(
    ('quant_policy', 'cache_shapes', 'expected_num_cache_views'),
    [
        pytest.param(QuantPolicy.NONE, [], 2, id='standard'),
        pytest.param(QuantPolicy.INT8, [], 4, id='quantized'),
        pytest.param(QuantPolicy.NONE, [([3], torch.float32)], 3, id='custom'),
    ],
)
def test_cache_engine_copy_logical_blocks_copies_complete_packed_blocks(quant_policy, cache_shapes,
                                                                        expected_num_cache_views):
    cache_engine, caches = _make_cache_engine(quant_policy=quant_policy, cache_shapes=cache_shapes)
    packed_cache = cache_engine.full_gpu_cache
    pages_per_block = 2
    src_block_offsets = (0, 2, 0)
    dst_block_offsets = (3, 1, 4)
    assert len(caches) == expected_num_cache_views

    for block_id in set(src_block_offsets):
        page_start = block_id * pages_per_block
        for page_offset in range(pages_per_block):
            packed_cache[:, page_start + page_offset].fill_(block_id * 20 + page_offset + 1)
            for cache_id, cache in enumerate(caches):
                cache[:, page_start + page_offset].fill_(cache_id * 20 + block_id * 2 + page_offset + 1)

    packed_sources = [
        packed_cache[:, src * pages_per_block:(src + 1) * pages_per_block].clone()
        for src in src_block_offsets
    ]
    copy_plan = _make_copy_plan(src_block_offsets, dst_block_offsets)
    cache_engine.copy_logical_blocks(copy_plan)

    for pair_idx, dst in enumerate(dst_block_offsets):
        dst_start = dst * pages_per_block
        dst_end = (dst + 1) * pages_per_block
        assert torch.equal(packed_cache[:, dst_start:dst_end], packed_sources[pair_idx])


def test_cache_engine_copy_logical_blocks_dispatches_one_batched_plan():
    cache_engine, _ = _make_cache_engine(num_blocks=6)
    copy_impl = _RecordingCacheBlockCopy()
    cache_engine._cache_block_copy_impl = copy_impl
    copy_plan = _make_copy_plan((0, 2, 0), (3, 1, 4))

    cache_engine.copy_logical_blocks(copy_plan)

    assert len(copy_impl.calls) == 1
    assert torch.equal(copy_impl.calls[0][0], copy_plan[0])
    assert torch.equal(copy_impl.calls[0][1], copy_plan[1])


def test_cache_engine_copy_logical_blocks_copies_heterogeneous_raw_pools():
    cache_engine, _ = _make_cache_engine(num_blocks=6)
    physical_pages = 2 * cache_engine.num_gpu_blocks
    packed_pools = [
        torch.full((2, physical_pages, 8), 99, dtype=torch.uint8),
        torch.full((2, 3, physical_pages, 16), 99, dtype=torch.uint8),
    ]
    cache_engine.full_gpu_cache = packed_pools
    cache_engine._cache_block_copy_device = packed_pools[0].device
    cache_engine._cache_block_copy_impl = DefaultCacheBlockCopyImpl(packed_caches=packed_pools,
                                                                     num_logical_blocks=6,
                                                                     pages_per_block=2,
                                                                     blocks_per_chunk=6)
    src_block_offsets = (0, 2, 0)
    dst_block_offsets = (3, 1, 4)
    src_pages = torch.tensor([0, 1, 4, 5, 0, 1])
    dst_pages = torch.tensor([6, 7, 2, 3, 8, 9])
    for pool_id, packed_pool in enumerate(packed_pools):
        for page_id in set(src_pages.tolist()):
            packed_pool.select(-2, page_id).fill_(pool_id * 40 + page_id + 1)
    expected_sources = [packed_pool.index_select(-2, src_pages).clone() for packed_pool in packed_pools]

    copy_plan = _make_copy_plan(src_block_offsets, dst_block_offsets)
    cache_engine.copy_logical_blocks(copy_plan)

    for packed_pool, expected_source in zip(packed_pools, expected_sources):
        assert torch.equal(packed_pool.index_select(-2, dst_pages), expected_source)


@pytest.mark.parametrize(
    ('copy_plan', 'error_type', 'error'),
    [
        pytest.param([[0], [2]], TypeError, 'torch.Tensor', id='not-a-tensor'),
        pytest.param(torch.tensor([0, 2]), ValueError, r'\[2, num_pairs\]', id='wrong-shape'),
        pytest.param(torch.tensor([[0.0], [2.0]]), TypeError, 'torch.long', id='wrong-dtype'),
        pytest.param(torch.empty((2, 1), dtype=torch.long, device='meta'),
                     ValueError,
                     'packed cache device',
                     id='wrong-device'),
    ],
)
def test_cache_engine_copy_logical_blocks_rejects_invalid_device_plan(copy_plan, error_type, error):
    cache_engine, _ = _make_cache_engine()
    with pytest.raises(error_type, match=error):
        cache_engine.copy_logical_blocks(copy_plan)

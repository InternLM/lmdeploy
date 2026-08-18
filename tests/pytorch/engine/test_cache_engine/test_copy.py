# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

import lmdeploy.pytorch.engine.cache_engine.engine as cache_engine_module
from lmdeploy.pytorch.backends.cuda.cache import CudaCacheBackend
from lmdeploy.pytorch.backends.cuda.op_backend import CudaOpsBackend
from lmdeploy.pytorch.backends.default.cache import DefaultCacheBackend, TorchBlockCacheCopy
from lmdeploy.pytorch.engine.cache_engine import CacheEngine
from lmdeploy.pytorch.engine.cache_engine.layout import CacheAllocation, CachePool


def _make_allocation(num_logical_blocks: int = 6, pages_per_block: int = 2):
    physical_pages = num_logical_blocks * pages_per_block
    pools = (
        CachePool(torch.full((2, physical_pages, 5), 99, dtype=torch.uint8), entry_axis=1),
        CachePool(torch.full((physical_pages, 3, 7), 99, dtype=torch.float16), entry_axis=0),
        CachePool(torch.full((2, 3, physical_pages, 4), 99, dtype=torch.float32), entry_axis=2),
    )
    return CacheAllocation(pools=pools, tensor_views=tuple(pool.tensor for pool in pools))


def _fill_sources(allocation, src_blocks, num_logical_blocks, pages_per_block):
    expected = []
    for pool_id, pool in enumerate(allocation.pools):
        logical_pool = pool.tensor.unflatten(pool.entry_axis,
                                             (num_logical_blocks, pages_per_block))
        for block_id in set(src_blocks.tolist()):
            logical_pool.select(pool.entry_axis, block_id).fill_(pool_id * 20 + block_id + 1)
        expected.append(logical_pool.index_select(pool.entry_axis, src_blocks).clone())
    return expected


def test_default_block_copy_handles_every_pool_entry_axis_and_repeated_sources():
    num_logical_blocks = 6
    pages_per_block = 2
    allocation = _make_allocation(num_logical_blocks, pages_per_block)
    block_copy = DefaultCacheBackend.build_block_copy(allocation,
                                                      num_logical_blocks,
                                                      pages_per_block)
    src_blocks = torch.tensor([0, 2, 0])
    dst_blocks = torch.tensor([3, 1, 4])
    expected = _fill_sources(allocation, src_blocks, num_logical_blocks, pages_per_block)

    block_copy.copy(src_blocks, dst_blocks)

    for pool, expected_pool in zip(allocation.pools, expected):
        logical_pool = pool.tensor.unflatten(pool.entry_axis,
                                             (num_logical_blocks, pages_per_block))
        assert torch.equal(logical_pool.index_select(pool.entry_axis, dst_blocks), expected_pool)

    workspace_ptrs = tuple(workspace.data_ptr() for workspace in block_copy._workspaces)
    block_copy.copy(src_blocks, dst_blocks)
    assert tuple(workspace.data_ptr() for workspace in block_copy._workspaces) == workspace_ptrs


def test_default_block_copy_bounds_aggregate_workspace(monkeypatch):
    num_logical_blocks = 6
    pages_per_block = 2
    allocation = _make_allocation(num_logical_blocks, pages_per_block)
    bytes_per_block = sum(pool.nbytes // num_logical_blocks for pool in allocation.pools)
    monkeypatch.setattr(TorchBlockCacheCopy, '_TARGET_WORKSPACE_BYTES', bytes_per_block * 2)

    block_copy = TorchBlockCacheCopy.build(allocation, num_logical_blocks, pages_per_block)

    assert block_copy.blocks_per_chunk == 2
    assert block_copy._workspaces is None
    block_copy.copy(torch.tensor([0]), torch.tensor([1]))
    assert sum(workspace.nbytes for workspace in block_copy._workspaces) == bytes_per_block * 2


def test_block_copy_validates_allocation_geometry_and_device():
    empty = CacheAllocation(pools=(), tensor_views=())
    with pytest.raises(ValueError, match='at least one allocation pool'):
        TorchBlockCacheCopy.build(empty, num_logical_blocks=2, pages_per_block=1)

    wrong_pages = CacheAllocation(
        pools=(CachePool(torch.empty((2, 3, 4)), entry_axis=1), ),
        tensor_views=(),
    )
    with pytest.raises(ValueError, match='has 3 physical pages; expected 4'):
        TorchBlockCacheCopy.build(wrong_pages, num_logical_blocks=2, pages_per_block=2)

    mixed_devices = CacheAllocation(
        pools=(
            CachePool(torch.empty((2, 2)), entry_axis=0),
            CachePool(torch.empty((2, 2), device='meta'), entry_axis=0),
        ),
        tensor_views=(),
    )
    with pytest.raises(ValueError, match='must use one device'):
        TorchBlockCacheCopy.build(mixed_devices, num_logical_blocks=2, pages_per_block=1)


def test_default_block_copy_supports_an_empty_allocation_extent():
    pool = CachePool(torch.empty((2, 0, 5)), entry_axis=1)
    allocation = CacheAllocation(pools=(pool, ), tensor_views=())
    block_copy = TorchBlockCacheCopy.build(allocation, num_logical_blocks=0, pages_per_block=2)

    block_copy.copy(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))

    assert block_copy.blocks_per_chunk == 1
    assert block_copy._workspaces is None


class _RecordingBlockCopy:

    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.calls = []

    def copy(self, src_block_offsets, dst_block_offsets):
        self.calls.append((src_block_offsets.clone(), dst_block_offsets.clone()))


def test_cache_engine_builds_block_copy_from_retained_allocation(monkeypatch):
    allocation = _make_allocation(num_logical_blocks=4, pages_per_block=3)
    block_copy = _RecordingBlockCopy()
    build_calls = []

    class _RecordingCacheBackend:

        @staticmethod
        def build_block_copy(allocation, **kwargs):
            build_calls.append((allocation, kwargs))
            return block_copy

    ops_backend = SimpleNamespace(get_cache_backend=lambda: _RecordingCacheBackend)
    monkeypatch.setattr(cache_engine_module, 'get_backend', lambda: ops_backend)
    cache_engine = object.__new__(CacheEngine)
    cache_engine.cache_config = SimpleNamespace(num_gpu_blocks=4)
    cache_engine.block_cache_plan = SimpleNamespace(kernel_blocks_per_logical_block=3)
    cache_engine.gpu_allocation = allocation

    cache_engine._build_block_copy()

    assert build_calls == [(allocation, dict(num_logical_blocks=4, pages_per_block=3))]
    assert cache_engine._block_copy is block_copy


def test_cache_engine_copy_logical_blocks_dispatches_device_plan():
    block_copy = _RecordingBlockCopy()
    cache_engine = object.__new__(CacheEngine)
    cache_engine._block_copy = block_copy
    copy_plan = torch.tensor(((0, 2, 0), (3, 1, 4)), dtype=torch.long)

    cache_engine.copy_logical_blocks(copy_plan)

    assert len(block_copy.calls) == 1
    assert torch.equal(block_copy.calls[0][0], copy_plan[0])
    assert torch.equal(block_copy.calls[0][1], copy_plan[1])

    cache_engine.copy_logical_blocks(torch.empty((2, 0), dtype=torch.long))
    assert len(block_copy.calls) == 1


@pytest.mark.parametrize(
    ('copy_plan', 'error_type', 'message'),
    [
        pytest.param([[0], [1]], TypeError, 'torch.Tensor', id='not-tensor'),
        pytest.param(torch.tensor([0, 1]), ValueError, r'\[2, num_pairs\]', id='shape'),
        pytest.param(torch.tensor([[0.0], [1.0]]), TypeError, 'torch.long', id='dtype'),
        pytest.param(torch.empty((2, 1), dtype=torch.long, device='meta'),
                     ValueError,
                     'allocation device',
                     id='device'),
    ],
)
def test_cache_engine_copy_logical_blocks_rejects_invalid_plan(copy_plan, error_type, message):
    cache_engine = object.__new__(CacheEngine)
    cache_engine._block_copy = _RecordingBlockCopy()

    with pytest.raises(error_type, match=message):
        cache_engine.copy_logical_blocks(copy_plan)


def test_cuda_backend_selects_cuda_cache_primitives():
    assert CudaOpsBackend.get_cache_backend() is CudaCacheBackend

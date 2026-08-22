# Copyright (c) OpenMMLab. All rights reserved.

import pytest
import torch

pytest.importorskip('triton')

from lmdeploy.pytorch.backends.cache_block_copy import CacheBlockCopyBuildSpec
from lmdeploy.pytorch.backends.cuda.op_backend import CudaOpsBackend
from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.engine.cache_engine import CacheEngine
from lmdeploy.pytorch.engine.cache_inputs import CacheCheckpointInputs

if torch.cuda.is_available():
    from lmdeploy.pytorch.backends.cuda.cache_block_copy import CudaCacheBlockCopyImpl
    from lmdeploy.pytorch.kernels.cuda.copy_packed_cache import copy_packed_cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA backend import')
def test_cuda_backend_builds_cache_block_copy():
    from lmdeploy.pytorch.backends.cuda.cache_block_copy import CudaCacheBlockCopyImpl

    copy_impl = CudaOpsBackend.build_op(
        CacheBlockCopyBuildSpec(packed_caches=(), num_logical_blocks=1, pages_per_block=1))

    assert isinstance(copy_impl, CudaCacheBlockCopyImpl)


def _logical_view(packed_cache: torch.Tensor, pages_per_block: int):
    num_logical_blocks = packed_cache.size(-2) // pages_per_block
    return packed_cache.unflatten(-2, (num_logical_blocks, pages_per_block))


def _seed_logical_blocks(packed_cache: torch.Tensor, block_offsets: tuple[int, ...], pages_per_block: int,
                         value_offset: int = 0):
    for block_offset in set(block_offsets):
        for page_offset in range(pages_per_block):
            page_id = block_offset * pages_per_block + page_offset
            packed_cache.select(-2, page_id).fill_(value_offset + block_offset * 20 + page_offset + 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize(('shape', 'pages_per_block'), [((3, 12, 1280), 2), ((2, 3, 18, 8448), 3)])
def test_copy_packed_cache_copies_logical_blocks(shape, pages_per_block):
    src_blocks_tuple = (0, 2, 0)
    dst_blocks_tuple = (3, 1, 4)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        packed_cache = torch.full(shape, 99, dtype=torch.uint8, device='cuda')
        src_blocks = torch.tensor(src_blocks_tuple, dtype=torch.long, device='cuda')
        dst_blocks = torch.tensor(dst_blocks_tuple, dtype=torch.long, device='cuda')
        _seed_logical_blocks(packed_cache, src_blocks_tuple, pages_per_block)
        logical_cache = _logical_view(packed_cache, pages_per_block)
        expected = logical_cache.index_select(-3, src_blocks).clone()
        expected_sources = expected.clone()
        untouched = logical_cache.select(-3, 5).clone()

        copy_packed_cache(packed_cache, src_blocks, dst_blocks, pages_per_block)

    stream.synchronize()
    assert torch.equal(logical_cache.index_select(-3, dst_blocks), expected)
    assert torch.equal(logical_cache.index_select(-3, src_blocks), expected_sources)
    assert torch.equal(logical_cache.select(-3, 5), untouched)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_cuda_cache_block_copy_uses_one_device_plan_for_multiple_pools():
    device = torch.device('cuda')
    pages_per_block = 2
    packed_pools = [
        torch.full((3, 12, 1280), 99, dtype=torch.uint8, device=device),
        torch.full((2, 12, 8448), 99, dtype=torch.uint8, device=device),
    ]
    copy_impl = CudaCacheBlockCopyImpl(pages_per_block=pages_per_block,
                                       packed_caches=packed_pools)
    src_blocks_tuple = (0, 2, 0)
    dst_blocks_tuple = (3, 1, 4)
    src_blocks = torch.tensor(src_blocks_tuple, device=device)
    dst_blocks = torch.tensor(dst_blocks_tuple, device=device)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        expected = []
        for pool_id, packed_cache in enumerate(packed_pools):
            _seed_logical_blocks(packed_cache, src_blocks_tuple, pages_per_block, value_offset=pool_id * 100)
            logical_cache = _logical_view(packed_cache, pages_per_block)
            expected.append(logical_cache.index_select(-3, src_blocks).clone())

        copy_impl.forward(src_blocks, dst_blocks)

    stream.synchronize()
    for packed_cache, expected_blocks in zip(packed_pools, expected):
        logical_cache = _logical_view(packed_cache, pages_per_block)
        assert torch.equal(logical_cache.index_select(-3, dst_blocks), expected_blocks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_cache_engine_dispatches_cache_checkpoint_device_plan():
    device = torch.device('cuda')
    pages_per_block = 2
    packed_cache = torch.zeros((2, 12, 256), dtype=torch.uint8, device=device)
    _seed_logical_blocks(packed_cache, (0, 2), pages_per_block)
    cache_engine = object.__new__(CacheEngine)
    cache_engine.cache_config = CacheConfig(max_batches=3,
                                            block_size=4,
                                            kernel_block_size=2,
                                            num_cpu_blocks=0,
                                            num_gpu_blocks=6,
                                            enable_prefix_caching=True,
                                            states_shapes=[((1, ), torch.float32)])
    cache_engine.full_gpu_cache = packed_cache
    cache_engine._cache_block_copy_device = packed_cache.device
    cache_engine._cache_block_copy_impl = CudaCacheBlockCopyImpl(pages_per_block=pages_per_block,
                                                                 packed_caches=[packed_cache])
    host_plan = torch.tensor(((0, 2, 0), (3, 1, 4)), dtype=torch.long).pin_memory()
    host_cache_inputs = CacheCheckpointInputs(kv_restore_plan=host_plan)
    h2d_stream = torch.cuda.Stream()
    forward_stream = torch.cuda.Stream()
    h2d_event = torch.cuda.Event()
    logical_cache = _logical_view(packed_cache, pages_per_block)
    torch.cuda.current_stream().synchronize()

    with torch.cuda.stream(h2d_stream):
        device_cache_inputs = host_cache_inputs.to_device(device, non_blocking=True)
        h2d_event.record()
    forward_stream.wait_event(h2d_event)
    with torch.cuda.stream(forward_stream):
        copy_plan = device_cache_inputs.kv_restore_plan
        expected = logical_cache.index_select(-3, copy_plan[0]).clone()
        cache_engine.copy_logical_blocks(copy_plan)

    forward_stream.synchronize()
    assert torch.equal(logical_cache.index_select(-3, copy_plan[1]), expected)

import itertools
import math

import pytest
import torch

tm = pytest.importorskip('_turbomind')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')


def _copy(src, dst):
    tm.generic_copy_on_stream(
        tm.from_dlpack_with_strides(src),
        tm.from_dlpack_with_strides(dst),
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()


def _check_copy(shape, src_strides, dst_strides, dtype=torch.int32, offset=16):
    # Compare the entire backing storage, including padding and guards.
    def storage_size(strides):
        return 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))

    src_storage = (torch.arange(storage_size(src_strides) + 2 * offset,
                                device='cuda', dtype=torch.int64) % 97).to(dtype)
    dst_storage = torch.full((storage_size(dst_strides) + 2 * offset,), 113,
                             device='cuda', dtype=dtype)
    expected = dst_storage.clone()
    src_before = src_storage.clone()
    src = src_storage.as_strided(shape, src_strides, offset)
    dst = dst_storage.as_strided(shape, dst_strides, offset)
    expected.as_strided(shape, dst_strides, offset).copy_(src)
    _copy(src, dst)
    torch.testing.assert_close(dst_storage, expected, rtol=0, atol=0)
    torch.testing.assert_close(src_storage, src_before, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.uint8, torch.int16, torch.int32, torch.int64])
@pytest.mark.parametrize('layers,requests', [(24, 1), (25, 1), (24, 4), (24, 7)])
def test_broadcast(dtype, layers, requests):
    _check_copy((layers, requests), (0, 1), (requests, 1), dtype)


def test_broadcast_to_permuted_destination():
    # Source coalesces its first two axes; destination coalesces its last two.
    # Equal resulting ranks do not imply equal resulting coordinate shapes.
    _check_copy((2, 3, 4), (0, 0, 1), (1, 8, 2))


@pytest.mark.parametrize('order', list(itertools.permutations(range(3))))
def test_padded_permutations(order):
    shape = (5, 3, 37)
    strides = [0] * 3
    pitch = 1
    for axis in order:
        strides[axis] = pitch
        pitch = pitch * shape[axis] + 1
    _check_copy(shape, (137, 43, 1), tuple(strides), offset=17)


@pytest.mark.parametrize('shape,src_strides,dst_strides', [
    ((2, 1, 3, 4), (0, 100, 0, 1), (1, 100, 8, 2)),
    ((1, 1, 1), (0, 0, 0), (9, 3, 1)),
    ((2, 3, 4, 5, 6), (360, 120, 30, 6, 1), (360, 120, 30, 6, 1)),
    ((5, 3, 100), (1, 7, 29), (1, 9, 41)),
])
def test_strided_and_coalesced_shapes(shape, src_strides, dst_strides):
    _check_copy(shape, src_strides, dst_strides)


@pytest.mark.parametrize('dtype', [torch.uint8, torch.int16, torch.int32, torch.int64])
@pytest.mark.parametrize('offset,padding', [(16, 0), (17, 0), (16, 1)])
def test_transpose_alignment(dtype, offset, padding):
    _check_copy((64, 64), (64 + padding, 1), (1, 64 + padding), dtype, offset)


@pytest.mark.parametrize('shape,src_strides,dst_strides', [
    ((3, 128, 192), (24576, 192, 1), (24576, 1, 128)),
    ((2, 3, 64, 64), (0, 4096, 64, 1), (12288, 4096, 1, 64)),
    ((64, 2, 64), (128, 64, 1), (1, 4096, 64)),
    ((2, 64, 64), (4161, 64, 1), (4161, 1, 64)),
    ((2, 3, 4, 64, 64), (49152, 16384, 4096, 64, 1),
     (49152, 16384, 4096, 1, 64)),
])
def test_batched_transpose(shape, src_strides, dst_strides):
    _check_copy(shape, src_strides, dst_strides)


@pytest.mark.parametrize('shape', [(0,), (2, 0, 8)])
def test_empty_copy(shape):
    src = torch.empty(shape, device='cuda')
    dst = torch.empty_like(src)
    _copy(src, dst)


def _check_large_copy(shape, dst_strides=None):
    count = math.prod(shape)
    free, _ = torch.cuda.mem_get_info()
    reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    # Two byte buffers plus space for the comparison and allocator overhead.
    if free + reusable < 3 * count + 512 * 1024**2:
        pytest.skip('Insufficient free GPU memory for the large-copy regression')

    guard = 64
    src_storage = torch.full((count + 2 * guard,), 251, device='cuda', dtype=torch.uint8)
    dst_storage = torch.full_like(src_storage, 251)
    src = src_storage[guard:-guard].view(shape)
    dst = dst_storage[guard:-guard].view(shape)
    if dst_strides is not None:
        dst = dst.as_strided(shape, dst_strides)
    dst.fill_(253)

    # Generate varying data without a count-sized int64 arange or reference copy.
    rows = src.view(-1, shape[-1])
    row_values = torch.arange(rows.shape[0], device='cuda', dtype=torch.int32).remainder_(97).to(torch.uint8)
    col_values = torch.arange(rows.shape[1], device='cuda', dtype=torch.int32).mul_(3).remainder_(97).to(torch.uint8)
    rows.copy_(row_values[:, None])
    rows.add_(col_values[None, :])
    _copy(src, dst)

    assert torch.equal(dst, src)
    for storage in (src_storage, dst_storage):
        assert bool((storage[:guard] == 251).all())
        assert bool((storage[-guard:] == 251).all())


@pytest.mark.parametrize('rows', [32767, 32768, 32769])
def test_coalesced_shape_int32_boundary(rows):
    _check_large_copy((rows, 65536))


@pytest.mark.parametrize('rows,columns', [(65535, 64), (32768, 128)])
def test_coalesced_transpose_grid_y_boundary(rows, columns):
    _check_large_copy((64, rows, columns), (1, columns * 64, 64))


@pytest.mark.parametrize('batch', [65535, 65536])
def test_transpose_grid_z_boundary(batch):
    _check_large_copy((batch, 64, 64), (4096, 1, 64))


def test_transpose_large_permuted_batch():
    _check_large_copy((257, 257, 64, 64), (4096, 257 * 4096, 1, 64))

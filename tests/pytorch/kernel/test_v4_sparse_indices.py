# Copyright (c) OpenMMLab. All rights reserved.

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='CUDA is required for V4 sparse-index kernels')


def _pad_ref(indices: torch.Tensor, block: int):
    topk = indices.size(-1)
    padded_topk = ((topk + block - 1) // block) * block
    if padded_topk == topk:
        return indices
    return torch.nn.functional.pad(indices, (0, padded_topk - topk), value=-1)


@torch.inference_mode()
def test_pad_sparse_indices_matches_torch_pad():
    from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import pad_sparse_indices

    indices = torch.tensor(
        [[[1, 3, -1, 7, 9]], [[-1, 2, 4, 6, 8]]],
        dtype=torch.int32,
        device='cuda')

    out = pad_sparse_indices(indices, block=4)
    ref = _pad_ref(indices, block=4).to(torch.int32)

    assert out.shape == (2, 1, 8)
    torch.testing.assert_close(out.cpu(), ref.cpu())


@torch.inference_mode()
def test_build_decode_window_sparse_indices_matches_reference():
    from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import build_decode_window_sparse_indices

    kv_seqlens = torch.tensor([3, 9, 17], dtype=torch.int32, device='cuda')
    start_pos = torch.tensor([2, 8, 16], dtype=torch.int64, device='cuda')
    is_padded = torch.tensor([False, True, False], dtype=torch.bool, device='cuda')
    window_size = 6
    block = 4

    indices, topk_length, window_pos, disabled_indices, disabled_topk_length = build_decode_window_sparse_indices(
        kv_seqlens, start_pos, is_padded, window_size, block=block)

    cols = torch.arange(window_size, dtype=torch.int32, device='cuda').unsqueeze(0)
    window_lens = kv_seqlens.clamp(max=window_size)
    first_abs = kv_seqlens - window_lens
    valid = cols < window_lens.unsqueeze(1)
    ref_indices = torch.remainder(first_abs.unsqueeze(1) + cols, window_size)
    ref_indices = torch.where(
        valid,
        ref_indices + torch.arange(3, dtype=torch.int32, device='cuda').unsqueeze(1) * window_size,
        torch.full((), -1, dtype=torch.int32, device='cuda'))
    ref_indices = _pad_ref(ref_indices.unsqueeze(1), block=block).to(torch.int32)
    ref_topk_length = torch.where(is_padded, torch.ones_like(window_lens), window_lens)
    ref_window_pos = torch.remainder(start_pos, window_size).to(torch.int32)
    ref_disabled_indices = torch.full((3, 1, block), -1, dtype=torch.int32, device='cuda')
    ref_disabled_topk_length = torch.where(
        is_padded,
        torch.ones_like(window_lens),
        torch.zeros_like(window_lens))

    torch.testing.assert_close(indices.cpu(), ref_indices.cpu())
    torch.testing.assert_close(topk_length.cpu(), ref_topk_length.cpu())
    torch.testing.assert_close(window_pos.cpu(), ref_window_pos.cpu())
    torch.testing.assert_close(disabled_indices.cpu(), ref_disabled_indices.cpu())
    torch.testing.assert_close(disabled_topk_length.cpu(), ref_disabled_topk_length.cpu())


@torch.inference_mode()
def test_build_decode_compressed_sparse_indices_matches_reference():
    from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import build_decode_compressed_sparse_indices

    logical_topk = torch.tensor(
        [[[0, 1, 4, -1, 99]], [[2, 7, 8, 11, -1]]],
        dtype=torch.int32,
        device='cuda')
    block_offsets = torch.tensor(
        [[7, 3, 9], [4, 8, 2]],
        dtype=torch.int64,
        device='cuda')
    block_size = 16
    compress_ratio = 4
    block = 4

    out = build_decode_compressed_sparse_indices(
        logical_topk, block_offsets, block_size, compress_ratio, block=block)

    bsz = logical_topk.size(0)
    safe_logical = logical_topk.clamp(min=0)
    token_positions = safe_logical * compress_ratio
    block_idx = torch.div(token_positions, block_size, rounding_mode='floor')
    max_block_idx = block_offsets.size(1)
    safe_block_idx = block_idx.clamp(max=max_block_idx - 1)
    block_idx_valid = block_idx < max_block_idx
    phys_block = block_offsets.gather(1, safe_block_idx.view(bsz, -1)).view_as(logical_topk)
    entries_per_block = block_size // compress_ratio
    block_off = torch.remainder(safe_logical, entries_per_block)
    phys_indices = phys_block * entries_per_block + block_off
    valid = (logical_topk >= 0) & block_idx_valid
    ref = torch.where(valid, phys_indices, phys_indices.new_full((), -1))
    ref = _pad_ref(ref, block=block).to(torch.int32)

    assert out.shape == (2, 1, 8)
    torch.testing.assert_close(out.cpu(), ref.cpu())


@torch.inference_mode()
def test_build_decode_prefix_compressed_sparse_indices_matches_reference():
    from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import build_decode_prefix_compressed_sparse_indices

    num_compressed = torch.tensor([3, 5], dtype=torch.int32, device='cuda')
    block_offsets = torch.tensor(
        [[7, 3, 9], [4, 8, 2]],
        dtype=torch.int64,
        device='cuda')
    block_size = 16
    compress_ratio = 4
    block = 4
    max_topk = 6

    out = build_decode_prefix_compressed_sparse_indices(
        num_compressed, block_offsets, block_size, compress_ratio, max_topk=max_topk, block=block)

    cols = torch.arange(8, dtype=torch.int64, device='cuda').unsqueeze(0)
    entries_per_block = block_size // compress_ratio
    token_positions = cols * compress_ratio
    block_idx = torch.div(token_positions, block_size, rounding_mode='floor')
    phys_block = block_offsets.gather(1, block_idx.clamp(max=block_offsets.size(1) - 1).expand(2, -1))
    block_off = torch.remainder(cols, entries_per_block)
    phys = phys_block * entries_per_block + block_off
    valid = cols < num_compressed.to(torch.int64).unsqueeze(1)
    ref = torch.where(valid, phys, phys.new_full((), -1)).unsqueeze(1).to(torch.int32)

    assert out.shape == (2, 1, 8)
    torch.testing.assert_close(out.cpu(), ref.cpu())


@torch.inference_mode()
@pytest.mark.parametrize('has_compress_offset', [False, True])
def test_assemble_prefill_sparse_indices_matches_reference(has_compress_offset):
    from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import assemble_prefill_sparse_indices

    window_topk = torch.tensor(
        [[0, 1, -1, -1, -1], [1, 2, 3, -1, -1], [0, 1, 2, 3, 4]],
        dtype=torch.int32,
        device='cuda')
    compress_topk = torch.tensor(
        [[0, 2, -1], [-1, 1, 3], [2, 4, 6]],
        dtype=torch.int32,
        device='cuda')
    repeat_cu = torch.tensor([0, 16, 32], dtype=torch.int32, device='cuda')
    compress_offset = torch.tensor([[5], [7], [11]], dtype=torch.int32, device='cuda')

    out = assemble_prefill_sparse_indices(
        window_topk,
        compress_topk,
        repeat_cu,
        compress_offset if has_compress_offset else None,
        block=4)

    comp_ref = compress_topk
    if has_compress_offset:
        comp_ref = torch.where(
            comp_ref >= 0,
            comp_ref + compress_offset,
            torch.full((), -1, dtype=torch.int32, device='cuda'))
    ref = torch.cat([window_topk, comp_ref], dim=-1)
    ref = torch.where(
        ref >= 0,
        ref + repeat_cu[:, None],
        torch.full((), -1, dtype=torch.int32, device='cuda'))
    ref = _pad_ref(ref.unsqueeze(1), block=4).to(torch.int32)

    assert out.shape == (3, 1, 8)
    torch.testing.assert_close(out.cpu(), ref.cpu())


@torch.inference_mode()
def test_assemble_prefill_sparse_indices_without_compress_matches_reference():
    from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import assemble_prefill_sparse_indices

    window_topk = torch.tensor(
        [[0, -1, -1], [0, 1, 2]],
        dtype=torch.int32,
        device='cuda')
    repeat_cu = torch.tensor([0, 8], dtype=torch.int32, device='cuda')

    out = assemble_prefill_sparse_indices(window_topk, None, repeat_cu, block=4)
    ref = torch.where(
        window_topk >= 0,
        window_topk + repeat_cu[:, None],
        torch.full((), -1, dtype=torch.int32, device='cuda'))
    ref = _pad_ref(ref.unsqueeze(1), block=4).to(torch.int32)

    assert out.shape == (2, 1, 4)
    torch.testing.assert_close(out.cpu(), ref.cpu())

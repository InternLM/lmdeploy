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


def _prefill_inputs(window_size: int, compress_ratio: int = 0):
    start_pos = torch.tensor([0, 5, 12], dtype=torch.int64, device='cuda')
    q_seqlens = torch.tensor([3, 2, 4], dtype=torch.int64, device='cuda')
    total_lens = start_pos + q_seqlens
    prev_window = start_pos.clamp(max=window_size)
    uncompressed_kv_lens = (prev_window + q_seqlens).to(torch.int64)
    compressed_lens = (
        torch.div(total_lens, compress_ratio, rounding_mode='floor')
        if compress_ratio else torch.zeros_like(total_lens))
    flat_lens = (uncompressed_kv_lens + compressed_lens).to(torch.int32)
    cu_seqlens_k = torch.zeros(4, dtype=torch.int32, device='cuda')
    torch.cumsum(flat_lens, dim=0, out=cu_seqlens_k[1:])

    cu_q = torch.zeros(4, dtype=torch.int64, device='cuda')
    torch.cumsum(q_seqlens, dim=0, out=cu_q[1:])
    token_id = torch.arange(int(cu_q[-1].item()), dtype=torch.int64, device='cuda')
    token_seq = torch.searchsorted(cu_q[1:], token_id, right=True)
    token_pos = token_id - cu_q[token_seq]
    return start_pos, total_lens, token_seq, token_pos, cu_seqlens_k, uncompressed_kv_lens


def _prefill_ref(
    start_pos,
    total_lens,
    token_seq,
    token_pos,
    cu_seqlens_k,
    uncompressed_kv_lens,
    window_size,
    compress_ratio=0,
    compress_topk=None,
    compress_width=0,
    block=4,
):
    abs_pos = start_pos[token_seq] + token_pos
    num_vis = (abs_pos + 1).clamp(max=window_size)
    window_start_abs = (start_pos - window_size).clamp(min=0)
    first_vis_abs = (abs_pos - window_size + 1).clamp(min=0)
    first_flat_pos = first_vis_abs - window_start_abs[token_seq]

    window_col = torch.arange(window_size, dtype=torch.int64, device='cuda').unsqueeze(0)
    window_vals = torch.where(
        window_col < num_vis.unsqueeze(1),
        first_flat_pos.unsqueeze(1) + window_col,
        torch.full((), -1, dtype=torch.int64, device='cuda'))

    if compress_ratio:
        comp_col = torch.arange(compress_width, dtype=torch.int64, device='cuda').unsqueeze(0)
        if compress_topk is not None:
            comp_vals = torch.where(
                compress_topk.to(torch.int64) >= 0,
                compress_topk.to(torch.int64) + uncompressed_kv_lens[token_seq].unsqueeze(1),
                torch.full((), -1, dtype=torch.int64, device='cuda'))
        else:
            num_compressed = torch.div(total_lens, compress_ratio, rounding_mode='floor')
            comp_vals = torch.where(
                comp_col < num_compressed[token_seq].unsqueeze(1),
                comp_col + uncompressed_kv_lens[token_seq].unsqueeze(1),
                torch.full((), -1, dtype=torch.int64, device='cuda'))
        vals = torch.cat([window_vals, comp_vals], dim=-1)
        padded_width = ((vals.size(-1) + block - 1) // block) * block
        topk_length = (window_size + torch.div(abs_pos + 1, compress_ratio, rounding_mode='floor')).clamp(
            max=padded_width).to(torch.int32)
    else:
        vals = window_vals
        topk_length = torch.full((token_seq.numel(),), window_size, dtype=torch.int32, device='cuda')

    vals = _pad_ref(vals.unsqueeze(1), block=block).squeeze(1)
    vals = torch.where(
        vals >= 0,
        vals + cu_seqlens_k[token_seq].to(torch.int64).unsqueeze(1),
        torch.full((), -1, dtype=torch.int64, device='cuda'))
    return vals.unsqueeze(1).to(torch.int32), topk_length


@torch.inference_mode()
def test_build_prefill_sparse_indices_without_compress_matches_reference():
    from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import build_prefill_sparse_indices

    window_size = 5
    args = _prefill_inputs(window_size)
    out, topk_length = build_prefill_sparse_indices(*args, window_size=window_size, block=4)
    ref, ref_topk_length = _prefill_ref(*args, window_size=window_size, block=4)

    assert out.shape == (9, 1, 8)
    torch.testing.assert_close(out.cpu(), ref.cpu())
    torch.testing.assert_close(topk_length.cpu(), ref_topk_length.cpu())


@torch.inference_mode()
def test_build_prefill_sparse_indices_prefix_compress_matches_reference():
    from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import build_prefill_sparse_indices

    window_size = 5
    compress_ratio = 4
    args = _prefill_inputs(window_size, compress_ratio)
    compress_width = 4
    out, topk_length = build_prefill_sparse_indices(
        *args,
        window_size=window_size,
        compress_ratio=compress_ratio,
        compress_width=compress_width,
        block=4)
    ref, ref_topk_length = _prefill_ref(
        *args,
        window_size=window_size,
        compress_ratio=compress_ratio,
        compress_width=compress_width,
        block=4)

    assert out.shape == (9, 1, 12)
    torch.testing.assert_close(out.cpu(), ref.cpu())
    torch.testing.assert_close(topk_length.cpu(), ref_topk_length.cpu())


@torch.inference_mode()
def test_build_prefill_sparse_indices_indexer_compress_matches_reference():
    from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import build_prefill_sparse_indices

    window_size = 5
    compress_ratio = 4
    args = _prefill_inputs(window_size, compress_ratio)
    compress_topk = torch.tensor(
        [[0, -1, -1], [0, 1, -1], [1, 0, -1],
         [0, 1, 2], [2, -1, 1], [1, 3, -1],
         [0, 2, 3], [3, 2, 1], [1, -1, 0]],
        dtype=torch.int32,
        device='cuda')

    out, topk_length = build_prefill_sparse_indices(
        *args,
        window_size=window_size,
        compress_ratio=compress_ratio,
        compress_topk=compress_topk,
        block=4)
    ref, ref_topk_length = _prefill_ref(
        *args,
        window_size=window_size,
        compress_ratio=compress_ratio,
        compress_topk=compress_topk,
        compress_width=compress_topk.size(1),
        block=4)

    assert out.shape == (9, 1, 8)
    torch.testing.assert_close(out.cpu(), ref.cpu())
    torch.testing.assert_close(topk_length.cpu(), ref_topk_length.cpu())

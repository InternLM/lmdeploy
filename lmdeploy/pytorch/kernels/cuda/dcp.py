# Copyright (c) OpenMMLab. All rights reserved.
"""CUDA kernels used by decode context parallel attention."""

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_dcp_lse_kernel(
    Lse,
    ValidRows,
    Out,
    numel,
    stride_lr,
    stride_lh,
    num_heads: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # ``offsets`` indexes the dense output, while ``Lse`` may be a view whose
    # physical row width is larger than ``num_heads``. Recover the logical
    # [row, head] coordinates and address the input with its actual strides.
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    rows = offsets // num_heads
    heads = offsets % num_heads
    lse = tl.load(Lse + rows * stride_lr + heads * stride_lh,
                  mask=mask).to(tl.float32)
    valid = tl.load(ValidRows + rows, mask=mask, other=0)
    invalid_lse = (lse != lse) | (lse == float('inf'))
    lse = tl.where(valid & ~invalid_lse, lse, -float('inf'))
    tl.store(Out + offsets, lse, mask=mask)


@triton.jit
def _correct_dcp_attention_output_kernel(
    LocalOutput,
    GatheredLse,
    CorrectedOutput,
    stride_ob,
    stride_oh,
    stride_od,
    stride_ln,
    stride_lb,
    stride_lh,
    stride_ch,
    stride_cb,
    stride_cd,
    dcp_rank: tl.constexpr,
    dcp_size: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    rank_offsets = tl.arange(0, BLOCK_N)
    rank_mask = rank_offsets < dcp_size
    lse_offsets = (rank_offsets * stride_ln + row * stride_lb +
                   head * stride_lh)
    lse = tl.load(GatheredLse + lse_offsets,
                  mask=rank_mask,
                  other=-float('inf')).to(tl.float32)

    max_lse = tl.max(lse, axis=0)
    safe_max_lse = tl.where(max_lse == -float('inf'), 0.0, max_lse)
    exp_lse = tl.exp(lse - safe_max_lse)
    denominator = tl.sum(exp_lse, axis=0)
    numerator = tl.sum(tl.where(rank_offsets == dcp_rank, exp_lse, 0.0),
                       axis=0)
    correction = tl.where(denominator > 0.0, numerator / denominator, 0.0)

    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < head_dim
    input_offsets = (row * stride_ob + head * stride_oh +
                     dim_offsets * stride_od)
    output_offsets = (head * stride_ch + row * stride_cb +
                      dim_offsets * stride_cd)
    output = tl.load(LocalOutput + input_offsets, mask=dim_mask,
                     other=0.0).to(tl.float32)
    output = tl.where(correction == 0.0, 0.0, output * correction)
    tl.store(CorrectedOutput + output_offsets, output, mask=dim_mask)


@triton.jit
def _merge_attention_states_kernel(
    PrefixOutput,
    PrefixLse,
    SuffixOutput,
    SuffixLse,
    Output,
    OutputLse,
    stride_pob,
    stride_poh,
    stride_pod,
    stride_plb,
    stride_plh,
    stride_sob,
    stride_soh,
    stride_sod,
    stride_slb,
    stride_slh,
    stride_ob,
    stride_oh,
    stride_od,
    stride_olb,
    stride_olh,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    prefix_lse = tl.load(PrefixLse + row * stride_plb +
                         head * stride_plh).to(tl.float32)
    suffix_lse = tl.load(SuffixLse + row * stride_slb +
                         head * stride_slh).to(tl.float32)
    prefix_lse = tl.where((prefix_lse == prefix_lse)
                          & (prefix_lse != float('inf')), prefix_lse,
                          -float('inf'))
    suffix_lse = tl.where((suffix_lse == suffix_lse)
                          & (suffix_lse != float('inf')), suffix_lse,
                          -float('inf'))
    max_lse = tl.maximum(prefix_lse, suffix_lse)
    valid = max_lse != -float('inf')
    safe_max = tl.where(valid, max_lse, 0.0)
    prefix_exp = tl.exp(prefix_lse - safe_max)
    suffix_exp = tl.exp(suffix_lse - safe_max)
    denominator = prefix_exp + suffix_exp
    prefix_scale = tl.where(valid, prefix_exp / denominator, 0.0)
    suffix_scale = tl.where(valid, suffix_exp / denominator, 0.0)

    dims = tl.arange(0, BLOCK_D)
    mask = dims < head_dim
    prefix_offset = (row * stride_pob + head * stride_poh +
                     dims * stride_pod)
    suffix_offset = (row * stride_sob + head * stride_soh +
                     dims * stride_sod)
    output_offset = row * stride_ob + head * stride_oh + dims * stride_od
    prefix = tl.load(PrefixOutput + prefix_offset, mask=mask, other=0.0)
    suffix = tl.load(SuffixOutput + suffix_offset, mask=mask, other=0.0)
    merged = (tl.where(prefix_scale == 0.0, 0.0, prefix * prefix_scale) +
              tl.where(suffix_scale == 0.0, 0.0, suffix * suffix_scale))
    tl.store(Output + output_offset, merged, mask=mask)
    tl.store(OutputLse + row * stride_olb + head * stride_olh,
             tl.where(valid, tl.log(denominator) + safe_max,
                      -float('inf')))


@triton.jit
def _scatter_dcp_prefill_kv_kernel(
    Gathered,
    Output,
    PrefixLens,
    KvStartLoc,
    LocalLens,
    chunk_rows,
    num_sequences,
    row_width,
    stride_gs,
    stride_gd,
    stride_os,
    stride_od,
    stride_lr,
    stride_ls,
    dcp_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    source_row = tl.program_id(0)
    rank = source_row // chunk_rows
    local_position = source_row % chunk_rows

    request = 0
    request_start = 0
    request_id = 0
    position_in_request = 0
    found = False
    while request < num_sequences:
        request_len = tl.load(LocalLens + rank * stride_lr +
                              request * stride_ls).to(tl.int32)
        owns_position = ((local_position >= request_start)
                         & (local_position < request_start + request_len))
        request_id = tl.where(owns_position, request, request_id)
        position_in_request = tl.where(owns_position,
                                       local_position - request_start,
                                       position_in_request)
        found |= owns_position
        request_start += request_len
        request += 1

    global_position = position_in_request * dcp_size + rank
    prefix_len = tl.load(PrefixLens + request_id, mask=found, other=0)
    output_start = tl.load(KvStartLoc + request_id, mask=found, other=0)
    valid_row = found & (global_position < prefix_len)

    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < row_width
    value = tl.load(Gathered + source_row * stride_gs +
                    dim_offsets * stride_gd,
                    mask=valid_row & dim_mask,
                    other=0.0)
    output_row = output_start + global_position
    tl.store(Output + output_row * stride_os + dim_offsets * stride_od,
             value,
             mask=valid_row & dim_mask)


def prepare_dcp_lse(local_lse: torch.Tensor,
                    valid_rows: torch.Tensor) -> torch.Tensor:
    """Normalize and compact LSE for DCP collectives or partition merging."""
    assert local_lse.dim() == 2
    assert valid_rows.shape == local_lse.shape[:1]
    output = torch.empty(local_lse.shape,
                         dtype=torch.float32,
                         device=local_lse.device)
    numel = local_lse.numel()
    block = 256
    # FlashMLA may pad the gathered query heads for kernel alignment and slice
    # its LSE result back to the original head count. The sliced tensor then
    # has the expected [tokens, heads] shape but retains the padded row stride.
    # Pass both input strides so this normalization kernel can also compact the
    # LSE for all-gather without an extra ``contiguous()`` copy.
    _prepare_dcp_lse_kernel[(triton.cdiv(numel, block), )](
        local_lse,
        valid_rows,
        output,
        numel,
        local_lse.stride(0),
        local_lse.stride(1),
        num_heads=local_lse.size(1),
        BLOCK=block,
    )
    return output


def correct_dcp_attention_output(local_output: torch.Tensor,
                                 gathered_lse: torch.Tensor,
                                 *, dcp_rank: int) -> torch.Tensor:
    """Apply global softmax correction and transpose for reduce-scatter.

    Args:
        local_output: Shard-local normalized output in ``[tokens, heads, dim]``.
        gathered_lse: Prepared LSE values in ``[dcp, tokens, heads]``.
        dcp_rank: Rank of ``local_output`` within the DCP group.

    Returns:
        FP32 corrected contributions in ``[heads, tokens, dim]``.
    """
    assert local_output.dim() == 3 and gathered_lse.dim() == 3
    dcp_size, num_tokens, num_heads = gathered_lse.shape
    assert local_output.shape[:2] == (num_tokens, num_heads)
    assert 0 <= dcp_rank < dcp_size

    corrected = torch.empty((num_heads, num_tokens, local_output.size(2)),
                            dtype=torch.float32,
                            device=local_output.device)
    block_n = triton.next_power_of_2(dcp_size)
    block_d = triton.next_power_of_2(local_output.size(2))
    _correct_dcp_attention_output_kernel[(num_tokens, num_heads)](
        local_output,
        gathered_lse,
        corrected,
        *local_output.stride(),
        *gathered_lse.stride(),
        *corrected.stride(),
        dcp_rank=dcp_rank,
        dcp_size=dcp_size,
        head_dim=local_output.size(2),
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return corrected


def merge_attention_states(
        prefix_output: torch.Tensor, prefix_lse: torch.Tensor,
        suffix_output: torch.Tensor,
        suffix_lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge attention partitions, retaining FP32 output for further merges."""
    assert prefix_output.shape == suffix_output.shape
    assert prefix_lse.shape == suffix_lse.shape == prefix_output.shape[:2]
    output = torch.empty_like(prefix_output, dtype=torch.float32)
    output_lse = torch.empty_like(prefix_lse, dtype=torch.float32)
    block_d = triton.next_power_of_2(prefix_output.size(2))
    _merge_attention_states_kernel[prefix_output.shape[:2]](
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        output,
        output_lse,
        *prefix_output.stride(),
        *prefix_lse.stride(),
        *suffix_output.stride(),
        *suffix_lse.stride(),
        *output.stride(),
        *output_lse.stride(),
        head_dim=prefix_output.size(2),
        BLOCK_D=block_d,
        num_warps=4,
    )
    return output, output_lse


def scatter_dcp_prefill_kv(gathered: torch.Tensor, output: torch.Tensor, *,
                           prefix_lens: torch.Tensor,
                           kv_start_loc: torch.Tensor,
                           local_lens: torch.Tensor) -> None:
    """Scatter rank-major KV into sequence order using chunk-relative
    lengths."""
    assert gathered.is_contiguous() and output.is_contiguous()
    assert gathered.dim() == output.dim()
    dcp_size, num_sequences = local_lens.shape
    assert prefix_lens.numel() == num_sequences
    assert kv_start_loc.numel() == num_sequences
    assert gathered.size(0) % dcp_size == 0
    chunk_rows = gathered.size(0) // dcp_size
    gathered_rows = gathered.view(gathered.size(0), -1)
    output_rows = output.view(output.size(0), -1)

    row_width = gathered_rows.size(1)
    block_d = triton.next_power_of_2(row_width)
    _scatter_dcp_prefill_kv_kernel[(gathered_rows.size(0), )](
        gathered_rows,
        output_rows,
        prefix_lens,
        kv_start_loc,
        local_lens,
        chunk_rows,
        num_sequences,
        row_width,
        *gathered_rows.stride(),
        *output_rows.stride(),
        *local_lens.stride(),
        dcp_size=dcp_size,
        BLOCK_D=block_d,
        num_warps=8,
    )

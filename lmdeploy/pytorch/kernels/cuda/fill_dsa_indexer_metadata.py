# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl


@triton.jit
def _fill_dsa_indexer_metadata_kernel(
    q_seqlens,
    kv_seqlens,
    cu_seqlens_q,
    block_offsets,
    indexer_kv_seqlens,
    expanded_block_offsets,
    batch_size,
    num_tokens,
    q_stride,
    kv_stride,
    cu_q_stride,
    block_row_stride,
    block_col_stride,
    expanded_row_stride,
    expanded_col_stride,
    MAX_QUERY_LEN: tl.constexpr,
    MAX_BLOCK_TABLE_LEN: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    EXPAND_BLOCKS: tl.constexpr,
):
    tl.static_assert(MAX_QUERY_LEN > 1, 'MTP query length must exceed one')
    tl.static_assert(MAX_BLOCK_TABLE_LEN > 0,
                     'MAX_BLOCK_TABLE_LEN must be positive')
    tl.static_assert(TILE_SIZE > 0, 'TILE_SIZE must be positive')

    batch_idx = tl.program_id(0)
    tile_idx = tl.program_id(1)
    offsets = tile_idx * TILE_SIZE + tl.arange(0, TILE_SIZE)

    q_len = tl.load(q_seqlens + batch_idx * q_stride).to(tl.int32)
    kv_len = tl.load(kv_seqlens + batch_idx * kv_stride).to(tl.int32)
    q_start = tl.load(cu_seqlens_q + batch_idx * cu_q_stride).to(
        tl.int32)
    tl.device_assert((q_len >= 0) & (q_len <= MAX_QUERY_LEN),
                     'q_seqlens exceeds graph query capacity')
    tl.device_assert(kv_len >= q_len,
                     'kv_seqlens must include all query tokens')
    tl.device_assert(q_start + q_len <= num_tokens,
                     'query range exceeds token capacity')
    if batch_idx == batch_size - 1:
        total_queries = tl.load(cu_seqlens_q +
                                batch_size * cu_q_stride)
        tl.device_assert(total_queries == num_tokens,
                         'q_seqlens do not sum to num_tokens')

    query_mask = offsets < q_len
    visible_kv_len = kv_len - q_len + offsets + 1
    tl.store(indexer_kv_seqlens + q_start + offsets,
             visible_kv_len,
             mask=query_mask)

    if EXPAND_BLOCKS:
        query_idx = offsets // MAX_BLOCK_TABLE_LEN
        block_idx = offsets % MAX_BLOCK_TABLE_LEN
        block_mask = query_idx < q_len
        block = tl.load(block_offsets + batch_idx * block_row_stride +
                        block_idx * block_col_stride,
                        mask=block_mask)
        output_row = q_start + query_idx
        tl.store(expanded_block_offsets +
                 output_row * expanded_row_stride +
                 block_idx * expanded_col_stride,
                 block,
                 mask=block_mask)


def fill_dsa_indexer_metadata(
        q_seqlens: torch.Tensor,
        kv_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        block_offsets: torch.Tensor,
        indexer_kv_seqlens: torch.Tensor,
        expanded_block_offsets: torch.Tensor | None,
        num_tokens: int,
        max_query_len: int,
) -> None:
    """Fill multi-token DSA indexer metadata into caller-owned tensors."""
    batch_size, block_table_len = block_offsets.size()
    expand_blocks = expanded_block_offsets is not None
    elements_per_batch = (max_query_len * block_table_len
                          if expand_blocks else max_query_len)
    tile_size = min(256, triton.next_power_of_2(elements_per_batch))
    grid = (batch_size, triton.cdiv(elements_per_batch, tile_size))
    if expanded_block_offsets is None:
        expanded_block_offsets = indexer_kv_seqlens
    _fill_dsa_indexer_metadata_kernel[grid](
        q_seqlens,
        kv_seqlens,
        cu_seqlens_q,
        block_offsets,
        indexer_kv_seqlens,
        expanded_block_offsets,
        batch_size,
        num_tokens,
        q_seqlens.stride(0),
        kv_seqlens.stride(0),
        cu_seqlens_q.stride(0),
        block_offsets.stride(0),
        block_offsets.stride(1),
        expanded_block_offsets.stride(0),
        expanded_block_offsets.stride(1) if expand_blocks else 0,
        MAX_QUERY_LEN=max_query_len,
        MAX_BLOCK_TABLE_LEN=block_table_len,
        TILE_SIZE=tile_size,
        EXPAND_BLOCKS=expand_blocks,
        num_warps=4,
    )

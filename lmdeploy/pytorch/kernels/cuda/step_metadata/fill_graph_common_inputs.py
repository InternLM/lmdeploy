# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl


@triton.jit
def _fill_graph_common_inputs_kernel(
    input_ids,
    position_ids,
    block_offsets,
    q_start_loc,
    q_seqlens,
    kv_seqlens,
    graph_input_ids,
    graph_position_ids,
    graph_block_offsets,
    qkv_lens,
    cu_seqlens,
    input_ids_stride,
    position_ids_stride,
    block_row_stride,
    block_col_stride,
    q_start_stride,
    q_seqlens_stride,
    kv_seqlens_stride,
    graph_input_ids_stride,
    graph_position_ids_stride,
    graph_block_row_stride,
    graph_block_col_stride,
    qkv_row_stride,
    qkv_col_stride,
    cu_row_stride,
    cu_col_stride,
    num_tokens,
    batch_size,
    num_blocks,
    q_batch_size,
    MAX_TOKENS: tl.constexpr,
    MAX_BATCHES: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    PAD_QUERY_LEN: tl.constexpr,
    BLOCK_COPY: tl.constexpr,
    BLOCK_BATCHES: tl.constexpr,
):
    tl.static_assert(MAX_TOKENS > 0, 'MAX_TOKENS must be positive')
    tl.static_assert(MAX_BATCHES > 0, 'MAX_BATCHES must be positive')
    tl.static_assert(MAX_BLOCKS > 0, 'MAX_BLOCKS must be positive')
    tl.static_assert(PAD_QUERY_LEN > 0, 'PAD_QUERY_LEN must be positive')
    tl.static_assert(BLOCK_COPY > 0, 'BLOCK_COPY must be positive')
    tl.static_assert(BLOCK_BATCHES >= MAX_BATCHES,
                     'BLOCK_BATCHES must cover MAX_BATCHES')
    tl.static_assert((BLOCK_BATCHES & (BLOCK_BATCHES - 1)) == 0,
                     'BLOCK_BATCHES must be a power of two')

    program_id = tl.program_id(0)

    if program_id == 0:
        # device_assert is compiled out unless TRITON_DEBUG is enabled.
        tl.device_assert((num_tokens >= 0) & (num_tokens <= MAX_TOKENS),
                         'num_tokens exceeds graph capacity')
        tl.device_assert((batch_size >= 0) &
                         (batch_size <= MAX_BATCHES),
                         'batch_size exceeds graph capacity')
        tl.device_assert((num_blocks >= 0) & (num_blocks <= MAX_BLOCKS),
                         'num_blocks exceeds graph capacity')
        tl.device_assert(q_batch_size == batch_size,
                         'sequence and block-table batches differ')

        q_offsets = tl.arange(0, BLOCK_BATCHES)
        valid = q_offsets < MAX_BATCHES
        real = q_offsets < q_batch_size

        q_start = tl.load(q_start_loc + q_offsets * q_start_stride,
                          mask=real,
                          other=0).to(tl.int32)
        q_len = tl.load(q_seqlens + q_offsets * q_seqlens_stride,
                        mask=real,
                        other=PAD_QUERY_LEN).to(tl.int32)
        kv_len = tl.load(kv_seqlens + q_offsets * kv_seqlens_stride,
                         mask=real,
                         other=PAD_QUERY_LEN).to(tl.int32)

        tl.store(qkv_lens + q_offsets * qkv_col_stride,
                 q_start,
                 mask=valid)
        tl.store(qkv_lens + qkv_row_stride + q_offsets * qkv_col_stride,
                 q_len,
                 mask=valid)
        tl.store(qkv_lens + 2 * qkv_row_stride + q_offsets * qkv_col_stride,
                 kv_len,
                 mask=valid)

        cu_q = tl.cumsum(q_len, 0)
        cu_k = tl.cumsum(kv_len, 0)
        last_real_q = real & (q_offsets == q_batch_size - 1)
        tl.device_assert(cu_q == num_tokens,
                         'q_seqlens do not sum to num_tokens',
                         mask=last_real_q)
        first = q_offsets == 0
        tl.store(cu_seqlens + q_offsets * cu_col_stride, 0, mask=first)
        tl.store(cu_seqlens + cu_row_stride + q_offsets * cu_col_stride,
                 0,
                 mask=first)
        tl.store(cu_seqlens + (q_offsets + 1) * cu_col_stride,
                 cu_q,
                 mask=valid)
        tl.store(cu_seqlens + cu_row_stride +
                 (q_offsets + 1) * cu_col_stride,
                 cu_k,
                 mask=valid)
    else:
        copy_offsets = ((program_id - 1) * BLOCK_COPY +
                        tl.arange(0, BLOCK_COPY))

        token_mask = ((copy_offsets < num_tokens) &
                      (copy_offsets < MAX_TOKENS))
        token_ids = tl.load(input_ids + copy_offsets * input_ids_stride,
                            mask=token_mask)
        positions = tl.load(position_ids +
                            copy_offsets * position_ids_stride,
                            mask=token_mask)
        tl.store(graph_input_ids +
                 copy_offsets * graph_input_ids_stride,
                 token_ids,
                 mask=token_mask)
        tl.store(graph_position_ids +
                 copy_offsets * graph_position_ids_stride,
                 positions,
                 mask=token_mask)

        block_mask = copy_offsets < MAX_BATCHES * MAX_BLOCKS
        block_rows = copy_offsets // MAX_BLOCKS
        block_cols = copy_offsets % MAX_BLOCKS
        real_blocks = (block_rows < batch_size) & (block_cols < num_blocks)
        blocks = tl.load(block_offsets + block_rows * block_row_stride +
                         block_cols * block_col_stride,
                         mask=real_blocks,
                         other=0).to(tl.int32)
        tl.store(graph_block_offsets +
                 block_rows * graph_block_row_stride +
                 block_cols * graph_block_col_stride,
                 blocks,
                 mask=block_mask)


def fill_graph_common_inputs(input_ids: torch.Tensor,
                             position_ids: torch.Tensor,
                             block_offsets: torch.Tensor,
                             q_start_loc: torch.Tensor,
                             q_seqlens: torch.Tensor,
                             kv_seqlens: torch.Tensor,
                             graph_input_ids: torch.Tensor,
                             graph_position_ids: torch.Tensor,
                             graph_block_offsets: torch.Tensor,
                             qkv_lens: torch.Tensor,
                             cu_seqlens: torch.Tensor,
                             pad_query_len: int) -> None:
    """Fill common token, block-table, and sequence graph buffers.

    The caller randomizes padded graph token IDs first. This operation only overwrites real token/position prefixes,
    while padding the block table and sequence lengths for safe graph replay.
    """
    num_tokens = input_ids.size(-1)
    batch_size, num_blocks = block_offsets.size()

    if not graph_input_ids.is_cuda:
        graph_input_ids[:, :num_tokens] = input_ids
        graph_position_ids[:, :num_tokens] = position_ids
        graph_block_offsets.zero_()
        graph_block_offsets[:batch_size, :num_blocks] = block_offsets
        qkv_lens.zero_()
        qkv_lens[1:].fill_(pad_query_len)
        qkv_lens[:, :q_seqlens.numel()] = torch.stack(
            (q_start_loc, q_seqlens, kv_seqlens))
        cu_seqlens[:, 0].zero_()
        cu_seqlens[:, 1:] = qkv_lens[1:].cumsum(1)
        return

    max_tokens = graph_input_ids.size(-1)
    max_batches, max_blocks = graph_block_offsets.size()
    block_copy = 256
    num_copy_programs = max(
        triton.cdiv(max_tokens, block_copy),
        triton.cdiv(max_batches * max_blocks, block_copy),
    )
    block_batches = triton.next_power_of_2(max_batches)
    _fill_graph_common_inputs_kernel[(num_copy_programs + 1, )](
        input_ids,
        position_ids,
        block_offsets,
        q_start_loc,
        q_seqlens,
        kv_seqlens,
        graph_input_ids,
        graph_position_ids,
        graph_block_offsets,
        qkv_lens,
        cu_seqlens,
        input_ids.stride(-1),
        position_ids.stride(-1),
        block_offsets.stride(0),
        block_offsets.stride(1),
        q_start_loc.stride(0),
        q_seqlens.stride(0),
        kv_seqlens.stride(0),
        graph_input_ids.stride(-1),
        graph_position_ids.stride(-1),
        graph_block_offsets.stride(0),
        graph_block_offsets.stride(1),
        qkv_lens.stride(0),
        qkv_lens.stride(1),
        cu_seqlens.stride(0),
        cu_seqlens.stride(1),
        num_tokens,
        batch_size,
        num_blocks,
        q_seqlens.numel(),
        MAX_TOKENS=max_tokens,
        MAX_BATCHES=max_batches,
        MAX_BLOCKS=max_blocks,
        PAD_QUERY_LEN=pad_query_len,
        BLOCK_COPY=block_copy,
        BLOCK_BATCHES=block_batches,
        num_warps=4,
    )

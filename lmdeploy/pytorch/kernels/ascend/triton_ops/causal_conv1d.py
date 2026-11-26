# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Tri Dao.
#
# causal_conv1d_fn         : Pure PyTorch prefill implementation (F.conv1d).
# causal_conv1d_update_npu : Decode-stage Triton kernel.
#   Adapted from https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/mamba/causal_conv1d.py
#   Original source: https://github.com/Dao-AILab/causal-conv1d
# mypy: ignore-errors

from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

PAD_SLOT_ID = -1


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = 'silu',
):
    """PyTorch reference implementation of causal conv1d.

    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)
    out: (batch, dim, seqlen)
    """
    if activation not in [None, 'silu', 'swish']:
        raise NotImplementedError('activation must be None, silu, or swish')
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape

    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]

    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = 'silu',
    conv_states: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    metadata: Any | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
    read_conv_offsets: torch.Tensor | None = None,
    write_conv_offsets: torch.Tensor | None = None,
):
    """Prefill-phase varlen causal conv1d using PyTorch reference
    implementation.

    x: (dim, cu_seq_len) for varlen
    weight: (dim, width)
    bias: (dim,)
    query_start_loc: (batch + 1) int32
    cache_indices: (batch) int32
    has_initial_state: (batch) bool
    conv_states: (..., dim, state_len) — state_len == width-1 in legacy linear
                 layout; state_len == width + num_spec_tokens in ring layout.

    Ring-buffer mode (used by MTP / speculative decoding) is enabled when
    `write_conv_offsets` is provided. Then `conv_states` is treated as a ring
    of size `state_len = conv_states.shape[-1]`:
      * initial state (when has_initial_state[i]) is gathered from
        read_conv_offsets[i] (shape (width-1,));
      * the trailing `width` tokens of (cached history ++ the sequence's
        own new tokens) are scattered to write_conv_offsets[i] (shape
        (width,)); zero padding replaces history for fresh requests.
    When `write_conv_offsets` is None the legacy semantics apply unchanged.
    """
    if activation not in [None, 'silu', 'swish']:
        raise NotImplementedError('activation must be None, silu, or swish')
    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    if query_start_loc is None:
        raise ValueError('query_start_loc is required for prefill mode')

    is_ring = write_conv_offsets is not None
    seqlens = query_start_loc[1:] - query_start_loc[:-1]
    seqlens = seqlens.tolist()
    splits = torch.split(x, seqlens, dim=-1)
    width = weight.shape[1]
    out_chunks = []
    for i in range(len(seqlens)):
        x_s = splits[i]
        if cache_indices[i] == PAD_SLOT_ID:
            continue

        if has_initial_state[i]:
            if is_ring:
                slot_idx = read_conv_offsets[i]
                init_state = conv_states[cache_indices[i]].index_select(-1, slot_idx)
            else:
                init_state = conv_states[cache_indices[i]][..., : (width - 1)]
        else:
            init_state = None

        out_ref_b = causal_conv1d_ref(
            x_s,
            weight,
            bias,
            activation=activation,
            return_final_states=not is_ring,
            final_states_out=(None if is_ring else conv_states[cache_indices[i]][..., : (width - 1)].unsqueeze(0)),
            initial_states=init_state,
        )
        out_chunks.append(out_ref_b[0])

        if is_ring and seqlens[i] > 0:
            # Seat the trailing `width` tokens of (history ++ new tokens) at
            # their circular slots. History must be gathered before this
            # scatter: for chunks shorter than `width` the missing tokens
            # live only in the ring (or are zero padding for fresh requests),
            # never in the flattened batch `x`.
            hist = init_state if init_state is not None else x_s.new_zeros(x_s.size(0), width - 1)
            tail = torch.cat([hist.to(x_s.dtype), x_s], dim=-1)[..., -width:]
            conv_states[cache_indices[i]].scatter_(
                -1,
                write_conv_offsets[i].to(torch.int64).unsqueeze(0).expand(conv_states.size(1), -1),
                tail.to(conv_states.dtype),
            )
    out = torch.cat(out_chunks, dim=-1)

    return out


@triton.jit
def _causal_conv1d_update_kernel_npu_tiled(
    x_ptr,
    w_ptr,
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    query_start_loc_ptr,
    block_idx_last_scheduled_token,
    initial_state_idx,
    cache_seqlens_ptr,
    o_ptr,
    batch: tl.int32,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    IS_CIRCULAR_BUFFER: tl.constexpr,
    BLOCK_N: tl.constexpr,
    B_TILE: tl.constexpr,
    T_CHUNK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    idx_feats = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_w = idx_feats < dim

    w_base = w_ptr + idx_feats * stride_w_dim
    w_col0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col4 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col5 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if KERNEL_WIDTH >= 1:
        w_col0 = tl.load(w_base + 0 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 2:
        w_col1 = tl.load(w_base + 1 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 3:
        w_col2 = tl.load(w_base + 2 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 4:
        w_col3 = tl.load(w_base + 3 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 5:
        w_col4 = tl.load(w_base + 4 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 6:
        w_col5 = tl.load(w_base + 5 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)

    if HAS_BIAS:
        acc_bias = tl.load(bias_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)
    else:
        acc_bias = tl.zeros((BLOCK_N,), dtype=tl.float32)

    tok_vec = tl.arange(0, T_CHUNK)

    for bi in tl.static_range(0, B_TILE):
        b = pid_b * B_TILE + bi
        lane_active = b < batch

        if IS_APC_ENABLED:
            conv_state_init = tl.load(initial_state_idx + b, mask=lane_active, other=0).to(tl.int32)
            current_last_index = tl.load(block_idx_last_scheduled_token + b, mask=lane_active, other=0).to(tl.int32)
        else:
            conv_state_init = tl.full((), 0, tl.int32)
            current_last_index = tl.full((), 0, tl.int32)

        conv_states_input_coord = tl.load(
            conv_state_indices_ptr + b * stride_state_indices + conv_state_init,
            mask=lane_active,
            other=0,
        ).to(tl.int64)

        if USE_PAD_SLOT:
            lane_active = lane_active & (conv_states_input_coord != pad_slot_id)

        if IS_VARLEN:
            qs = tl.load(query_start_loc_ptr + b, mask=lane_active, other=0).to(tl.int64)
            qe = tl.load(query_start_loc_ptr + (b + 1), mask=lane_active, other=0).to(tl.int64)
            seqlen_run = (qe - qs).to(tl.int32)
            state_len_run = (state_len - (seqlen - seqlen_run)).to(tl.int32)
            x_offset = (qs * stride_x_token).to(tl.int64)
            o_offset = (qs * stride_o_token).to(tl.int64)
        else:
            seqlen_run = tl.full((), seqlen, tl.int32)
            state_len_run = tl.full((), state_len, tl.int32)
            x_offset = (b * stride_x_seq).to(tl.int64)
            o_offset = (b * stride_o_seq).to(tl.int64)

        lane_active = lane_active & (seqlen_run > 0)

        if IS_CIRCULAR_BUFFER:
            cb_raw = tl.load(cache_seqlens_ptr + b, mask=lane_active, other=0).to(tl.int32)
            cb_pos = cb_raw % state_len  # write-start: wrapped current position
            cb_read_start = (cb_pos - (KERNEL_WIDTH - 1) + state_len) % state_len
            conv_state_token_offset = tl.full((), 0, tl.int64)
            shift = tl.full((), 0, tl.int32)
        else:
            cb_pos = tl.full((), 0, tl.int32)
            cb_read_start = tl.full((), 0, tl.int32)
            if IS_SPEC_DECODING:
                conv_state_token_offset = (
                    tl.load(num_accepted_tokens_ptr + b, mask=lane_active, other=1).to(tl.int64) - 1
                )
                shift = tl.full((), 1, tl.int32)
            else:
                conv_state_token_offset = tl.full((), 0, tl.int64)
                shift = seqlen_run

        conv_states_base = (
            conv_state_ptr + conv_states_input_coord * stride_conv_state_seq + idx_feats * stride_conv_state_dim
        )
        prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok

        col0 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col1 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col2 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col3 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col4 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        if IS_CIRCULAR_BUFFER:
            if KERNEL_WIDTH >= 2:
                col0 = tl.load(
                    conv_states_base + cb_read_start * stride_conv_state_tok,
                    mask=lane_active & mask_w,
                    other=0.0,
                ).to(tl.float16)
            if KERNEL_WIDTH >= 3:
                col1 = tl.load(
                    conv_states_base + ((cb_read_start + 1) % state_len) * stride_conv_state_tok,
                    mask=lane_active & mask_w,
                    other=0.0,
                ).to(tl.float16)
            if KERNEL_WIDTH >= 4:
                col2 = tl.load(
                    conv_states_base + ((cb_read_start + 2) % state_len) * stride_conv_state_tok,
                    mask=lane_active & mask_w,
                    other=0.0,
                ).to(tl.float16)
            if KERNEL_WIDTH >= 5:
                col3 = tl.load(
                    conv_states_base + ((cb_read_start + 3) % state_len) * stride_conv_state_tok,
                    mask=lane_active & mask_w,
                    other=0.0,
                ).to(tl.float16)
            if KERNEL_WIDTH >= 6:
                col4 = tl.load(
                    conv_states_base + ((cb_read_start + 4) % state_len) * stride_conv_state_tok,
                    mask=lane_active & mask_w,
                    other=0.0,
                ).to(tl.float16)
        else:
            if KERNEL_WIDTH >= 2:
                col0 = tl.load(
                    prior_tokens + 0 * stride_conv_state_tok,
                    mask=lane_active & mask_w,
                    other=0.0,
                ).to(tl.float16)
            if KERNEL_WIDTH >= 3:
                col1 = tl.load(
                    prior_tokens + 1 * stride_conv_state_tok,
                    mask=lane_active & mask_w,
                    other=0.0,
                ).to(tl.float16)
            if KERNEL_WIDTH >= 4:
                col2 = tl.load(
                    prior_tokens + 2 * stride_conv_state_tok,
                    mask=lane_active & mask_w,
                    other=0.0,
                ).to(tl.float16)
            if KERNEL_WIDTH >= 5:
                col3 = tl.load(
                    prior_tokens + 3 * stride_conv_state_tok,
                    mask=lane_active & mask_w,
                    other=0.0,
                ).to(tl.float16)
            if KERNEL_WIDTH >= 6:
                col4 = tl.load(
                    prior_tokens + 4 * stride_conv_state_tok,
                    mask=lane_active & mask_w,
                    other=0.0,
                ).to(tl.float16)

        if not IS_CIRCULAR_BUFFER:
            conv_states_offset = tl.load(
                conv_state_indices_ptr + b * stride_state_indices + current_last_index,
                mask=lane_active,
                other=0,
            ).to(tl.int64)
        else:
            conv_states_offset = tl.full((), 0, tl.int64)

        use_shift = seqlen_run < state_len_run
        use_tail = seqlen_run >= state_len_run
        zero_i32 = tl.full((), 0, tl.int32)
        keep_shift = tl.where(use_shift, (state_len_run - seqlen_run), zero_i32).to(tl.int32)
        tail_start = tl.where(use_tail, (seqlen_run - state_len_run), zero_i32).to(tl.int32)

        state_src_base = (
            conv_state_ptr
            + conv_states_input_coord * stride_conv_state_seq
            + conv_state_token_offset * stride_conv_state_tok
            + idx_feats * stride_conv_state_dim
        )
        state_dst_base = conv_state_ptr + conv_states_offset * stride_conv_state_seq + idx_feats * stride_conv_state_dim
        x_base = x_ptr + x_offset + idx_feats * stride_x_dim

        if IS_CIRCULAR_BUFFER:
            # Circular write: write each input token to its circular position.
            # No shift needed — positions are derived from cache_seqlens.
            for t0 in tl.static_range(0, seqlen, T_CHUNK):
                x_tok = (t0 + tok_vec).to(tl.int32)
                write_tok = (cb_pos + x_tok) % state_len
                m = (
                    (lane_active & (x_tok < seqlen_run))[:, None]
                    & mask_w[None, :]
                    & (conv_states_input_coord < num_cache_lines)
                )
                x_ptrs = x_base[None, :] + x_tok[:, None] * stride_x_token
                dst_ptrs = conv_states_base[None, :] + write_tok[:, None] * stride_conv_state_tok
                x_vals = tl.load(x_ptrs, mask=m, other=0.0)
                tl.store(dst_ptrs, x_vals, mask=m)
        else:
            for t0 in tl.static_range(0, NP2_STATELEN, T_CHUNK):
                dst_tok = (t0 + tok_vec).to(tl.int32)
                src_tok = (dst_tok + shift).to(tl.int32)
                m_tok = use_shift & (dst_tok < keep_shift) & (src_tok < state_len_run) & (dst_tok < state_len_run)
                m = (
                    (lane_active & m_tok)[:, None]
                    & mask_w[None, :]
                    & (conv_states_input_coord < num_cache_lines)
                    & (conv_states_offset < num_cache_lines)
                )
                src_ptrs = state_src_base[None, :] + src_tok[:, None] * stride_conv_state_tok
                dst_ptrs = state_dst_base[None, :] + dst_tok[:, None] * stride_conv_state_tok
                vals = tl.load(src_ptrs, mask=m, other=0.0)
                tl.store(dst_ptrs, vals, mask=m)

            for t0 in tl.static_range(0, seqlen, T_CHUNK):
                x_tok = (t0 + tok_vec).to(tl.int32)
                dst_tok = (keep_shift + x_tok).to(tl.int32)
                m_tok = use_shift & (x_tok < seqlen_run) & (dst_tok < state_len_run)
                m = (lane_active & m_tok)[:, None] & mask_w[None, :] & (conv_states_offset < num_cache_lines)
                x_ptrs = x_base[None, :] + x_tok[:, None] * stride_x_token
                dst_ptrs = state_dst_base[None, :] + dst_tok[:, None] * stride_conv_state_tok
                x_vals = tl.load(x_ptrs, mask=m, other=0.0)
                tl.store(dst_ptrs, x_vals, mask=m)

            for t0 in tl.static_range(0, NP2_STATELEN, T_CHUNK):
                dst_tok = (t0 + tok_vec).to(tl.int32)
                x_tok = (tail_start + dst_tok).to(tl.int32)
                m_tok = use_tail & (dst_tok < state_len_run) & (x_tok < seqlen_run)
                m = (lane_active & m_tok)[:, None] & mask_w[None, :] & (conv_states_offset < num_cache_lines)
                x_ptrs = x_base[None, :] + x_tok[:, None] * stride_x_token
                dst_ptrs = state_dst_base[None, :] + dst_tok[:, None] * stride_conv_state_tok
                x_vals = tl.load(x_ptrs, mask=m, other=0.0)
                tl.store(dst_ptrs, x_vals, mask=m)

        x_base_1d = x_base
        o_base_1d = o_ptr + o_offset + idx_feats * stride_o_dim
        acc_preload = acc_bias

        for idx_token in tl.range(seqlen_run):
            acc = acc_preload
            matrix_w = w_col0
            matrix_x = col0
            for j in tl.static_range(KERNEL_WIDTH):
                if KERNEL_WIDTH == 1:
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0).to(tl.float16)
                    matrix_w = w_col0
                elif KERNEL_WIDTH == 2:
                    if j == 1:
                        matrix_w = w_col1
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0).to(tl.float16)
                elif KERNEL_WIDTH == 3:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0).to(tl.float16)
                elif KERNEL_WIDTH == 4:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0).to(tl.float16)
                elif KERNEL_WIDTH == 5:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        matrix_x = col3
                    elif j == 4:
                        matrix_w = w_col4
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0).to(tl.float16)
                elif KERNEL_WIDTH == 6:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        matrix_x = col3
                    elif j == 4:
                        matrix_w = w_col4
                        matrix_x = col4
                    elif j == 5:
                        matrix_w = w_col5
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0).to(tl.float16)
                acc += matrix_x.to(tl.float32) * matrix_w

            if KERNEL_WIDTH == 2:
                col0 = matrix_x
            elif KERNEL_WIDTH == 3:
                col0 = col1
                col1 = matrix_x
            elif KERNEL_WIDTH == 4:
                col0 = col1
                col1 = col2
                col2 = matrix_x
            elif KERNEL_WIDTH == 5:
                col0 = col1
                col1 = col2
                col2 = col3
                col3 = matrix_x
            elif KERNEL_WIDTH == 6:
                col0 = col1
                col1 = col2
                col2 = col3
                col3 = col4
                col4 = matrix_x

            if SILU_ACTIVATION:
                acc = acc / (1.0 + tl.exp(-acc))
            o_ptrs = o_base_1d + idx_token * stride_o_token
            tl.store(o_ptrs, acc, mask=lane_active & mask_w)


def causal_conv1d_update_npu(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = 'silu',
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    cache_seqlens: torch.Tensor | None = None,
    validate_data=False,
):
    if validate_data:
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = 'silu' if activation is True else None
    elif activation is not None:
        assert activation in ['silu', 'swish']

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    unsqueeze = query_start_loc is None and x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(1)

    if query_start_loc is None:
        batch, seqlen, dim = x.shape
    else:
        assert conv_state_indices is not None
        batch = conv_state_indices.size(0)
        dim = x.size(1)
        seqlen = max_query_len

    width, _ = weight.shape
    num_cache_lines, state_len_total, _ = conv_state.size()

    out = x

    stride_w_width, stride_w_dim = weight.stride()
    if query_start_loc is None:
        stride_x_seq, stride_x_token, stride_x_dim = x.stride()
        stride_o_seq, stride_o_token, stride_o_dim = out.stride()
    else:
        stride_x_token, stride_x_dim = x.stride()
        stride_x_seq = 0
        stride_o_token, stride_o_dim = out.stride()
        stride_o_seq = 0

    stride_istate_seq, stride_istate_token, stride_istate_dim = conv_state.stride()
    stride_state_indices = conv_state_indices.stride(0) if conv_state_indices is not None else 0

    if cache_seqlens is not None:
        # Circular buffer: use the full allocated state size as the modulus.
        eff_state_len = state_len_total
    elif num_accepted_tokens is not None:
        eff_state_len = width - 1 + (seqlen - 1)
    else:
        eff_state_len = width - 1
    np2_statelen = triton.next_power_of_2(eff_state_len)

    CORE_HINT = 40
    block_n = 512 if dim >= 512 else 256
    g = triton.cdiv(dim, block_n)
    target = 2 * CORE_HINT
    b_tile_raw = max(1, (batch * g + target - 1) // target)
    if b_tile_raw <= 1:
        b_tile = 1
    elif b_tile_raw <= 2:
        b_tile = 2
    elif b_tile_raw <= 4:
        b_tile = 4
    else:
        b_tile = 8
    t_chunk = 1 if block_n == 512 else 48

    def grid(META):
        return (triton.cdiv(batch, META['B_TILE']), triton.cdiv(dim, META['BLOCK_N']))

    _causal_conv1d_update_kernel_npu_tiled[grid](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        block_idx_last_scheduled_token,
        initial_state_idx,
        cache_seqlens,
        out,
        batch,
        dim,
        seqlen,
        eff_state_len,
        num_cache_lines,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ['silu', 'swish'],
        IS_VARLEN=query_start_loc is not None,
        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        IS_CIRCULAR_BUFFER=cache_seqlens is not None,
        BLOCK_N=block_n,
        B_TILE=b_tile,
        T_CHUNK=t_chunk,
    )

    if unsqueeze:
        out = out.squeeze(1)
    return out.to(original_x_dtype)

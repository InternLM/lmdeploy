# Copyright (c) OpenMMLab. All rights reserved.
"""Triton gather/scatter kernels for MiMo-V2-Flash BF16 SWA state rings."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_mimo_swa_ring_kernel(
    ring_ptr,
    output_ptr,
    state_slots_ptr,
    start_positions_ptr,
    history_lens_ptr,
    cu_history_lens_ptr,
    stride_ring_slot,
    stride_ring_pos,
    stride_ring_head,
    stride_ring_dim,
    stride_output_token,
    stride_output_head,
    stride_output_dim,
    NUM_STATE_SLOTS: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    batch_id = tl.program_id(0)
    history_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    history_len = tl.load(history_lens_ptr + batch_id)
    if history_idx >= history_len:
        return

    state_slot = tl.load(state_slots_ptr + batch_id)
    if state_slot < 0 or state_slot >= NUM_STATE_SLOTS:
        return

    start_position = tl.load(start_positions_ptr + batch_id)
    absolute_position = start_position - history_len + history_idx
    ring_position = absolute_position % WINDOW_SIZE
    output_token = tl.load(cu_history_lens_ptr + batch_id) + history_idx

    dim_offsets = tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offsets < HEAD_DIM
    source_offsets = (
        state_slot.to(tl.int64) * stride_ring_slot
        + ring_position.to(tl.int64) * stride_ring_pos
        + head_idx * stride_ring_head
        + dim_offsets * stride_ring_dim
    )
    output_offsets = (
        output_token.to(tl.int64) * stride_output_token
        + head_idx * stride_output_head
        + dim_offsets * stride_output_dim
    )
    payload = tl.load(ring_ptr + source_offsets, mask=dim_mask)
    tl.store(output_ptr + output_offsets, payload, mask=dim_mask)


@triton.jit
def _scatter_mimo_swa_ring_kernel(
    tokens_ptr,
    ring_ptr,
    state_slots_ptr,
    start_positions_ptr,
    q_seqlens_ptr,
    cu_q_seqlens_ptr,
    stride_tokens_token,
    stride_tokens_head,
    stride_tokens_dim,
    stride_ring_slot,
    stride_ring_pos,
    stride_ring_head,
    stride_ring_dim,
    NUM_STATE_SLOTS: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    batch_id = tl.program_id(0)
    write_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    state_slot = tl.load(state_slots_ptr + batch_id)
    if state_slot < 0 or state_slot >= NUM_STATE_SLOTS:
        return

    start_position = tl.load(start_positions_ptr + batch_id)
    if start_position < 0:
        return

    q_seqlen = tl.load(q_seqlens_ptr + batch_id)
    write_len = tl.minimum(q_seqlen, WINDOW_SIZE)
    if write_idx >= write_len:
        return

    # If a chunk exceeds the ring capacity, its early tokens cannot be part of
    # the final state.  Starting at q_seqlen - write_len also prevents two
    # programs from racing to write the same modulo slot.
    local_position = q_seqlen - write_len + write_idx
    token_idx = tl.load(cu_q_seqlens_ptr + batch_id) + local_position
    absolute_position = start_position + local_position
    ring_position = absolute_position % WINDOW_SIZE

    dim_offsets = tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offsets < HEAD_DIM
    token_offsets = (
        token_idx.to(tl.int64) * stride_tokens_token
        + head_idx * stride_tokens_head
        + dim_offsets * stride_tokens_dim
    )
    ring_offsets = (
        state_slot.to(tl.int64) * stride_ring_slot
        + ring_position.to(tl.int64) * stride_ring_pos
        + head_idx * stride_ring_head
        + dim_offsets * stride_ring_dim
    )
    payload = tl.load(tokens_ptr + token_offsets, mask=dim_mask)
    tl.store(ring_ptr + ring_offsets, payload, mask=dim_mask)


@triton.jit
def _flatten_mimo_swa_ring_kernel(
    ring_ptr,
    current_ptr,
    output_ptr,
    state_slots_ptr,
    start_positions_ptr,
    q_seqlens_ptr,
    cu_q_seqlens_ptr,
    history_lens_ptr,
    cu_kv_seqlens_ptr,
    stride_ring_slot,
    stride_ring_pos,
    stride_ring_head,
    stride_ring_dim,
    stride_current_token,
    stride_current_head,
    stride_current_dim,
    stride_output_token,
    stride_output_head,
    stride_output_dim,
    NUM_STATE_SLOTS: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """Pack each sequence as chronological history followed by current KV."""
    batch_id = tl.program_id(0)
    sequence_token = tl.program_id(1)
    head_idx = tl.program_id(2)

    history_len = tl.load(history_lens_ptr + batch_id)
    q_seqlen = tl.load(q_seqlens_ptr + batch_id)
    if sequence_token >= history_len + q_seqlen:
        return

    output_token = tl.load(cu_kv_seqlens_ptr + batch_id) + sequence_token
    dim_offsets = tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offsets < HEAD_DIM
    output_offsets = (
        output_token.to(tl.int64) * stride_output_token
        + head_idx * stride_output_head
        + dim_offsets * stride_output_dim
    )

    if sequence_token < history_len:
        state_slot = tl.load(state_slots_ptr + batch_id)
        start_position = tl.load(start_positions_ptr + batch_id)
        absolute_position = start_position - history_len + sequence_token
        ring_position = absolute_position % WINDOW_SIZE
        source_offsets = (
            state_slot.to(tl.int64) * stride_ring_slot
            + ring_position.to(tl.int64) * stride_ring_pos
            + head_idx * stride_ring_head
            + dim_offsets * stride_ring_dim
        )
        payload = tl.load(ring_ptr + source_offsets, mask=dim_mask)
    else:
        current_token = tl.load(cu_q_seqlens_ptr + batch_id) + sequence_token - history_len
        source_offsets = (
            current_token.to(tl.int64) * stride_current_token
            + head_idx * stride_current_head
            + dim_offsets * stride_current_dim
        )
        payload = tl.load(current_ptr + source_offsets, mask=dim_mask)
    tl.store(output_ptr + output_offsets, payload, mask=dim_mask)


def _validate_ring(ring: torch.Tensor) -> None:
    if ring.dim() != 4:
        raise ValueError(f'MiMo SWA ring must be [state_slots, window, heads, dim], got {tuple(ring.shape)}.')
    if ring.dtype != torch.bfloat16:
        raise TypeError(f'MiMo SWA ring must use torch.bfloat16, got {ring.dtype}.')
    if not ring.is_cuda:
        raise ValueError('MiMo SWA ring kernels require CUDA tensors.')
    if ring.size(1) <= 1:
        raise ValueError(f'MiMo SWA ring window must be greater than one, got {ring.size(1)}.')


def _validate_batch_vector(name: str, tensor: torch.Tensor, batch_size: int, device: torch.device) -> None:
    if tensor.dim() != 1 or tensor.numel() != batch_size:
        raise ValueError(f'{name} must have shape [{batch_size}], got {tuple(tensor.shape)}.')
    if tensor.device != device:
        raise ValueError(f'{name} must be on {device}, got {tensor.device}.')
    if tensor.dtype not in (torch.int32, torch.int64):
        raise TypeError(f'{name} must use int32 or int64, got {tensor.dtype}.')


def gather_mimo_swa_ring(
    ring: torch.Tensor,
    state_slots: torch.Tensor,
    start_positions: torch.Tensor,
    history_limit: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather previous MiMo SWA state in chronological order.

    Args:
        ring: One layer's state view with shape
            ``[num_state_slots, window_size, local_kv_heads, head_dim]``.
        state_slots: ``[batch]`` state row IDs.  A negative or out-of-range ID
            is a dummy sequence and contributes zero history tokens.
        start_positions: ``[batch]`` absolute positions of each current chunk.
        history_limit: Maximum previous-token count.  MiMo uses
            ``window_size - 1`` so the current query plus its history contains
            at most ``window_size`` tokens.

    Returns:
        A capacity-sized packed payload buffer, actual per-sequence history
        lengths, and their cumulative offsets.  Only rows before
        ``cu_history_lens[-1]`` contain payload.
    """
    _validate_ring(ring)
    batch_size = state_slots.numel()
    _validate_batch_vector('state_slots', state_slots, batch_size, ring.device)
    _validate_batch_vector('start_positions', start_positions, batch_size, ring.device)

    window_size = ring.size(1)
    if history_limit is None:
        history_limit = window_size - 1
    if history_limit < 0 or history_limit > window_size:
        raise ValueError(f'history_limit must be in [0, {window_size}], got {history_limit}.')

    valid_slots = (state_slots >= 0) & (state_slots < ring.size(0)) & (start_positions >= 0)
    history_lens = start_positions.clamp(min=0, max=history_limit).to(torch.int32)
    history_lens = torch.where(valid_slots, history_lens, 0)
    cu_history_lens = torch.zeros(batch_size + 1, dtype=torch.int32, device=ring.device)
    torch.cumsum(history_lens, dim=0, out=cu_history_lens[1:])

    num_heads = ring.size(2)
    head_dim = ring.size(3)
    output_capacity = batch_size * history_limit
    output = torch.empty(
        (output_capacity, num_heads, head_dim), dtype=ring.dtype, device=ring.device
    )
    if batch_size == 0 or history_limit == 0:
        return output, history_lens, cu_history_lens

    block_dim = triton.next_power_of_2(head_dim)
    grid = (batch_size, history_limit, num_heads)
    _gather_mimo_swa_ring_kernel[grid](
        ring,
        output,
        state_slots,
        start_positions,
        history_lens,
        cu_history_lens,
        ring.stride(0),
        ring.stride(1),
        ring.stride(2),
        ring.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        NUM_STATE_SLOTS=ring.size(0),
        WINDOW_SIZE=window_size,
        HEAD_DIM=head_dim,
        BLOCK_DIM=block_dim,
    )
    return output, history_lens, cu_history_lens


def flatten_mimo_swa_ring(
    ring: torch.Tensor,
    current: torch.Tensor,
    state_slots: torch.Tensor,
    start_positions: torch.Tensor,
    q_seqlens: torch.Tensor,
    cu_q_seqlens: torch.Tensor,
    history_lens: torch.Tensor,
    cu_kv_seqlens: torch.Tensor,
    max_q_seqlen: int,
) -> torch.Tensor:
    """Pack chronological ring history and current tokens for varlen attention.

    The output order is ``history_i + current_i`` for every sequence ``i``.
    Its allocation uses a synchronization-free upper bound; only entries
    addressed by ``cu_kv_seqlens`` are consumed by varlen attention.
    """
    _validate_ring(ring)
    if current.dim() != 3:
        raise ValueError(f'current must be [total_q, heads, dim], got {tuple(current.shape)}.')
    if current.dtype != ring.dtype or current.device != ring.device:
        raise ValueError('current and ring must have the same BF16 dtype and CUDA device.')
    if current.shape[1:] != ring.shape[2:]:
        raise ValueError(
            f'current head shape {tuple(current.shape[1:])} does not match ring {tuple(ring.shape[2:])}.'
        )
    if not isinstance(max_q_seqlen, int) or max_q_seqlen < 0:
        raise ValueError(f'max_q_seqlen must be a non-negative int, got {max_q_seqlen!r}.')

    batch_size = state_slots.numel()
    _validate_batch_vector('state_slots', state_slots, batch_size, ring.device)
    _validate_batch_vector('start_positions', start_positions, batch_size, ring.device)
    _validate_batch_vector('q_seqlens', q_seqlens, batch_size, ring.device)
    _validate_batch_vector('history_lens', history_lens, batch_size, ring.device)
    if cu_q_seqlens.dim() != 1 or cu_q_seqlens.numel() != batch_size + 1:
        raise ValueError(f'cu_q_seqlens must have shape [{batch_size + 1}], got {tuple(cu_q_seqlens.shape)}.')
    if cu_kv_seqlens.dim() != 1 or cu_kv_seqlens.numel() != batch_size + 1:
        raise ValueError(f'cu_kv_seqlens must have shape [{batch_size + 1}], got {tuple(cu_kv_seqlens.shape)}.')
    for name, cumulative in (('cu_q_seqlens', cu_q_seqlens), ('cu_kv_seqlens', cu_kv_seqlens)):
        if cumulative.device != ring.device or cumulative.dtype not in (torch.int32, torch.int64):
            raise ValueError(f'{name} must be an int32/int64 tensor on the ring device.')

    history_limit = ring.size(1) - 1
    output_capacity = current.size(0) + batch_size * history_limit
    output = torch.empty(
        (output_capacity, ring.size(2), ring.size(3)),
        dtype=ring.dtype,
        device=ring.device,
    )
    if batch_size == 0 or output_capacity == 0:
        return output

    block_dim = triton.next_power_of_2(ring.size(3))
    grid = (batch_size, history_limit + max_q_seqlen, ring.size(2))
    _flatten_mimo_swa_ring_kernel[grid](
        ring,
        current,
        output,
        state_slots,
        start_positions,
        q_seqlens,
        cu_q_seqlens,
        history_lens,
        cu_kv_seqlens,
        ring.stride(0),
        ring.stride(1),
        ring.stride(2),
        ring.stride(3),
        current.stride(0),
        current.stride(1),
        current.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        NUM_STATE_SLOTS=ring.size(0),
        WINDOW_SIZE=ring.size(1),
        HEAD_DIM=ring.size(3),
        BLOCK_DIM=block_dim,
    )
    return output


def scatter_mimo_swa_ring(
    tokens: torch.Tensor,
    ring: torch.Tensor,
    state_slots: torch.Tensor,
    start_positions: torch.Tensor,
    q_seqlens: torch.Tensor,
    cu_q_seqlens: torch.Tensor,
) -> None:
    """Write each chunk's newest tokens into one layer's MiMo SWA ring.

    ``tokens`` is packed by sequence.  Valid state slot IDs must be unique
    within the batch.  At most the last ``window_size`` tokens of each chunk
    are written; attention must gather the old state before this function is
    called.
    """
    _validate_ring(ring)
    if tokens.dim() != 3:
        raise ValueError(f'tokens must be [total_q, heads, dim], got {tuple(tokens.shape)}.')
    if tokens.dtype != ring.dtype or tokens.device != ring.device:
        raise ValueError('tokens and ring must have the same BF16 dtype and CUDA device.')
    if tokens.size(1) != ring.size(2) or tokens.size(2) != ring.size(3):
        raise ValueError(
            f'tokens head shape {tuple(tokens.shape[1:])} does not match ring {tuple(ring.shape[2:])}.'
        )

    batch_size = state_slots.numel()
    _validate_batch_vector('state_slots', state_slots, batch_size, ring.device)
    _validate_batch_vector('start_positions', start_positions, batch_size, ring.device)
    _validate_batch_vector('q_seqlens', q_seqlens, batch_size, ring.device)
    if cu_q_seqlens.dim() != 1 or cu_q_seqlens.numel() != batch_size + 1:
        raise ValueError(
            f'cu_q_seqlens must have shape [{batch_size + 1}], got {tuple(cu_q_seqlens.shape)}.'
        )
    if cu_q_seqlens.device != ring.device or cu_q_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError('cu_q_seqlens must be an int32/int64 tensor on the ring device.')
    if batch_size == 0 or tokens.numel() == 0:
        return

    window_size = ring.size(1)
    num_heads = ring.size(2)
    head_dim = ring.size(3)
    block_dim = triton.next_power_of_2(head_dim)
    grid = (batch_size, window_size, num_heads)
    _scatter_mimo_swa_ring_kernel[grid](
        tokens,
        ring,
        state_slots,
        start_positions,
        q_seqlens,
        cu_q_seqlens,
        tokens.stride(0),
        tokens.stride(1),
        tokens.stride(2),
        ring.stride(0),
        ring.stride(1),
        ring.stride(2),
        ring.stride(3),
        NUM_STATE_SLOTS=ring.size(0),
        WINDOW_SIZE=window_size,
        HEAD_DIM=head_dim,
        BLOCK_DIM=block_dim,
    )

# Copyright (c) OpenMMLab. All rights reserved.
"""ConceptLM runtime kernels."""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_chunk_state_update_kernel(
    state_cache,
    current_states,
    state_ids,
    position_ids,
    concept_inputs,
    next_rows,
    update_mask,
    state_stride_n,
    state_stride_s,
    state_stride_h,
    cur_stride_b,
    cur_stride_s,
    cur_stride_h,
    out_stride_b,
    out_stride_s,
    out_stride_h,
    next_stride_b,
    next_stride_s,
    next_stride_h,
    HIDDEN: tl.constexpr,
    TOTAL_ELEMS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    MERGE_METHOD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)
    offs = tile_id * BLOCK + tl.arange(0, BLOCK)
    valid_elem = offs < TOTAL_ELEMS

    source_id = offs // HIDDEN
    hidden_id = offs - source_id * HIDDEN

    state_id = tl.load(state_ids + batch_id)
    valid_state = state_id >= 0
    safe_state_id = tl.maximum(state_id, 0)
    pos = tl.load(position_ids + batch_id)
    chunk_pos = pos % CHUNK_SIZE
    is_boundary = ((pos + 1) % CHUNK_SIZE) == 0
    is_first_token = chunk_pos == 0

    current_ptrs = (current_states + batch_id * cur_stride_b + source_id * cur_stride_s +
                    hidden_id * cur_stride_h)
    state_ptrs = state_cache + safe_state_id * state_stride_n + source_id * state_stride_s + hidden_id * state_stride_h

    current = tl.load(current_ptrs, mask=valid_elem, other=0.0).to(tl.float32)
    previous = tl.load(state_ptrs, mask=valid_elem, other=0.0).to(tl.float32)

    if MERGE_METHOD == 1:  # first
        merged = tl.where(is_first_token, current, previous)
        concept = merged
    elif MERGE_METHOD == 2:  # last
        merged = current
        concept = current
    else:  # meanpooling
        merged = previous + current
        concept = merged / CHUNK_SIZE

    zero = tl.zeros((BLOCK, ), dtype=tl.float32)
    next_value = tl.where(is_boundary, zero, merged)
    concept_value = tl.where(valid_state & is_boundary, concept, zero)
    next_debug_value = tl.where(valid_state, next_value, previous)

    concept_ptrs = concept_inputs + batch_id * out_stride_b + source_id * out_stride_s + hidden_id * out_stride_h
    next_ptrs = next_rows + batch_id * next_stride_b + source_id * next_stride_s + hidden_id * next_stride_h
    tl.store(concept_ptrs, concept_value, mask=valid_elem)
    tl.store(next_ptrs, next_debug_value, mask=valid_elem)
    tl.store(state_ptrs, next_value, mask=valid_elem & valid_state)

    if tile_id == 0:
        tl.store(update_mask + batch_id, valid_state & is_boundary)


def _flatten_decode_position_ids(position_ids: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    """Normalize decode position ids to one absolute position per batch row."""
    if position_ids.dim() == 0:
        position_ids = position_ids.view(1)
    if position_ids.dim() == 1:
        return position_ids.to(device=device, dtype=torch.long)
    position_ids = position_ids.reshape(-1)
    if position_ids.numel() == batch_size:
        return position_ids.to(device=device, dtype=torch.long)
    assert position_ids.numel() % batch_size == 0, (
        f'Cannot map position_ids with {position_ids.numel()} elements to batch size {batch_size}.')
    return position_ids.reshape(-1, batch_size)[-1].to(device=device, dtype=torch.long)


def _merge_method_id(merge_method: str) -> int:
    """Map ConceptLM merge method string to kernel constexpr id."""
    merge_method = str(merge_method)
    if merge_method == 'first':
        return 1
    if merge_method == 'last':
        return 2
    return 0


def decode_chunk_state_update(
    chunk_source_state_cache: torch.Tensor,
    current_source_states: torch.Tensor,
    state_ids: torch.Tensor,
    position_ids: torch.Tensor,
    chunk_size: int,
    merge_method: str,
    block: int = 1024,
):
    """Update ConceptLM decode chunk-source state in-place.

    Args:
        chunk_source_state_cache: ``[num_state_slots, num_sources, hidden]``.
        current_source_states: ``[batch, num_sources, hidden]``.
        state_ids: ``[batch]`` with ``-1`` for padded graph rows.
        position_ids: absolute decode positions.
        chunk_size: ConceptLM chunk size.
        merge_method: ``meanpooling``, ``first``, or ``last``.
        block: Triton vector width.

    Returns:
        Tuple ``(concept_inputs, next_rows, update_mask)``. ``concept_inputs``
        is zero for non-boundary rows. ``next_rows`` is a debug/reference copy
        of the per-batch rows written to state cache. ``update_mask`` is
        ``True`` only for valid boundary rows.
    """
    assert chunk_source_state_cache.is_cuda, 'ConceptLM chunk-state kernel requires CUDA state cache.'
    assert current_source_states.is_cuda, 'ConceptLM chunk-state kernel requires CUDA current states.'
    assert current_source_states.dim() == 3
    assert chunk_source_state_cache.dim() == 3
    assert current_source_states.shape[1:] == chunk_source_state_cache.shape[1:]

    batch_size, num_sources, hidden = current_source_states.shape
    total_elems = num_sources * hidden
    state_ids = state_ids.to(device=current_source_states.device, dtype=torch.long)
    position_ids = _flatten_decode_position_ids(position_ids, batch_size, current_source_states.device)
    concept_inputs = torch.empty_like(current_source_states)
    next_rows = torch.empty_like(current_source_states)
    update_mask = torch.empty((batch_size, ), dtype=torch.bool, device=current_source_states.device)
    grid = (triton.cdiv(total_elems, block), batch_size)
    _decode_chunk_state_update_kernel[grid](
        chunk_source_state_cache,
        current_source_states,
        state_ids,
        position_ids,
        concept_inputs,
        next_rows,
        update_mask,
        *chunk_source_state_cache.stride(),
        *current_source_states.stride(),
        *concept_inputs.stride(),
        *next_rows.stride(),
        HIDDEN=hidden,
        TOTAL_ELEMS=total_elems,
        CHUNK_SIZE=int(chunk_size),
        MERGE_METHOD=_merge_method_id(merge_method),
        BLOCK=block,
        num_warps=8,
    )
    return concept_inputs, next_rows, update_mask

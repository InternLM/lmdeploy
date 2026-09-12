# Copyright (c) OpenMMLab. All rights reserved.
"""ConceptLM runtime kernels."""

import torch
import triton
import triton.language as tl


@triton.jit
def _prefill_chunk_state_update_kernel(
    source_states,
    concept_states,
    token_start_ids,
    token_counts,
    source_stride_t,
    source_stride_s,
    source_stride_h,
    out_stride_c,
    out_stride_s,
    out_stride_h,
    HIDDEN: tl.constexpr,
    TOTAL_ELEMS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    MERGE_METHOD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Merge contiguous token chunks into compact concept-source rows."""
    tile_id = tl.program_id(0)
    concept_id = tl.program_id(1)
    offs = tile_id * BLOCK + tl.arange(0, BLOCK)
    valid_elem = offs < TOTAL_ELEMS

    source_id = offs // HIDDEN
    hidden_id = offs - source_id * HIDDEN

    token_start = tl.load(token_start_ids + concept_id)
    token_count = tl.load(token_counts + concept_id)
    has_token = token_count > 0

    acc = tl.zeros((BLOCK, ), dtype=tl.float32)
    for token_offset in range(CHUNK_SIZE):
        load_mask = valid_elem & (token_offset < token_count)
        ptrs = (source_states + (token_start + token_offset) * source_stride_t + source_id * source_stride_s +
                hidden_id * source_stride_h)
        values = tl.load(ptrs, mask=load_mask, other=0.0).to(tl.float32)
        acc += values

    denom = tl.maximum(token_count, 1).to(tl.float32)
    mean_values = acc / denom
    if MERGE_METHOD == 1:  # first; short prompts keep reference mean-pooling
        first_ptrs = (source_states + token_start * source_stride_t + source_id * source_stride_s +
                      hidden_id * source_stride_h)
        first_values = tl.load(first_ptrs, mask=valid_elem & has_token, other=0.0).to(tl.float32)
        out_values = tl.where(token_count < CHUNK_SIZE, mean_values, first_values)
    elif MERGE_METHOD == 2:  # last; short prompts keep reference mean-pooling
        last_token = token_start + tl.maximum(token_count, 1) - 1
        last_ptrs = (source_states + last_token * source_stride_t + source_id * source_stride_s +
                     hidden_id * source_stride_h)
        last_values = tl.load(last_ptrs, mask=valid_elem & has_token, other=0.0).to(tl.float32)
        out_values = tl.where(token_count < CHUNK_SIZE, mean_values, last_values)
    else:
        out_values = mean_values

    out_ptrs = concept_states + concept_id * out_stride_c + source_id * out_stride_s + hidden_id * out_stride_h
    tl.store(out_ptrs, out_values, mask=valid_elem)


@triton.jit
def _prefill_state_cache_update_kernel(
    chunk_state_cache,
    last_raw_state_cache,
    last_final_state_cache,
    source_states,
    predicted_vectors,
    raw_states,
    state_ids,
    token_q_start_loc,
    token_q_seqlens,
    concept_q_start_loc,
    concept_q_seqlens,
    chunk_state_stride_n,
    chunk_state_stride_s,
    chunk_state_stride_h,
    raw_cache_stride_n,
    raw_cache_stride_l,
    raw_cache_stride_h,
    final_cache_stride_n,
    final_cache_stride_h,
    source_stride_t,
    source_stride_s,
    source_stride_h,
    pred_stride_c,
    pred_stride_h,
    raw_stride_c,
    raw_stride_l,
    raw_stride_h,
    HIDDEN: tl.constexpr,
    SOURCE_ELEMS: tl.constexpr,
    RAW_ELEMS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    MERGE_METHOD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Seed decode state caches directly from prefill rows."""
    tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)
    state_id = tl.load(state_ids + batch_id)
    if state_id < 0:
        return

    offs = tile_id * BLOCK + tl.arange(0, BLOCK)
    q_start = tl.load(token_q_start_loc + batch_id).to(tl.int64)
    q_len = tl.load(token_q_seqlens + batch_id).to(tl.int64)
    tail_len = q_len % CHUNK_SIZE
    tail_len = tl.where(q_len < CHUNK_SIZE, q_len, tail_len)
    tail_len = tl.where(q_len > 0, tail_len, 0)
    has_tail = tail_len > 0

    source_mask = offs < SOURCE_ELEMS
    source_id = offs // HIDDEN
    source_hidden_id = offs - source_id * HIDDEN
    tail_start = q_start + q_len - tail_len

    if MERGE_METHOD == 1:  # first
        first_ptrs = (source_states + tail_start * source_stride_t + source_id * source_stride_s +
                      source_hidden_id * source_stride_h)
        source_values = tl.load(first_ptrs, mask=source_mask & has_tail, other=0.0).to(tl.float32)
    elif MERGE_METHOD == 2:  # last
        last_ptrs = (source_states + (q_start + q_len - 1) * source_stride_t + source_id * source_stride_s +
                     source_hidden_id * source_stride_h)
        source_values = tl.load(last_ptrs, mask=source_mask & has_tail, other=0.0).to(tl.float32)
    else:
        source_values = tl.zeros((BLOCK, ), dtype=tl.float32)
        for token_offset in range(CHUNK_SIZE):
            load_mask = source_mask & (token_offset < tail_len)
            ptrs = (source_states + (tail_start + token_offset) * source_stride_t + source_id * source_stride_s +
                    source_hidden_id * source_stride_h)
            values = tl.load(ptrs, mask=load_mask, other=0.0).to(tl.float32)
            source_values += values

    chunk_ptrs = (chunk_state_cache + state_id * chunk_state_stride_n + source_id * chunk_state_stride_s +
                  source_hidden_id * chunk_state_stride_h)
    tl.store(chunk_ptrs, source_values, mask=source_mask)

    concept_count = tl.load(concept_q_seqlens + batch_id).to(tl.int64)
    has_concept = concept_count > 0
    concept_start = tl.load(concept_q_start_loc + batch_id).to(tl.int64)
    last_concept_id = concept_start + concept_count - 1

    final_hidden_id = offs
    final_mask = (offs < HIDDEN) & has_concept
    pred_ptrs = predicted_vectors + last_concept_id * pred_stride_c + final_hidden_id * pred_stride_h
    final_ptrs = last_final_state_cache + state_id * final_cache_stride_n + final_hidden_id * final_cache_stride_h
    final_values = tl.load(pred_ptrs, mask=final_mask, other=0.0)
    tl.store(final_ptrs, final_values, mask=final_mask)

    raw_mask = (offs < RAW_ELEMS) & has_concept
    raw_layer_id = offs // HIDDEN
    raw_hidden_id = offs - raw_layer_id * HIDDEN
    raw_ptrs = raw_states + last_concept_id * raw_stride_c + raw_layer_id * raw_stride_l + raw_hidden_id * raw_stride_h
    raw_cache_ptrs = (last_raw_state_cache + state_id * raw_cache_stride_n + raw_layer_id * raw_cache_stride_l +
                      raw_hidden_id * raw_cache_stride_h)
    raw_values = tl.load(raw_ptrs, mask=raw_mask, other=0.0)
    tl.store(raw_cache_ptrs, raw_values, mask=raw_mask)


@triton.jit
def _decode_chunk_state_update_kernel(
    state_cache,
    current_states,
    state_ids,
    position_ids,
    concept_inputs,
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

    concept_ptrs = concept_inputs + batch_id * out_stride_b + source_id * out_stride_s + hidden_id * out_stride_h
    tl.store(concept_ptrs, concept_value, mask=valid_elem)
    tl.store(state_ptrs, next_value, mask=valid_elem & valid_state)

    if tile_id == 0:
        tl.store(update_mask + batch_id, valid_state & is_boundary)


@triton.jit
def _decode_kv_cache_snapshot_kernel(
    k_cache,
    v_cache,
    block_offsets,
    kv_seqlens,
    saved_k,
    saved_v,
    k_stride_n,
    k_stride_b,
    k_stride_h,
    k_stride_d,
    v_stride_n,
    v_stride_b,
    v_stride_h,
    v_stride_d,
    boff_stride_b,
    boff_stride_n,
    sk_stride_b,
    sk_stride_h,
    sk_stride_d,
    sv_stride_b,
    sv_stride_h,
    sv_stride_d,
    HEAD_DIM: tl.constexpr,
    TOTAL_ELEMS: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Snapshot one paged decode KV slot per batch row."""
    tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)
    offs = tile_id * BLOCK + tl.arange(0, BLOCK)
    valid_elem = offs < TOTAL_ELEMS

    head_id = offs // HEAD_DIM
    dim_id = offs - head_id * HEAD_DIM

    kv_seqlen = tl.maximum(tl.load(kv_seqlens + batch_id), 1)
    slot_id = kv_seqlen - 1
    block_idx = slot_id // KV_BLOCK_SIZE
    page_offset = slot_id - block_idx * KV_BLOCK_SIZE
    block_id = tl.load(block_offsets + batch_id * boff_stride_b + block_idx * boff_stride_n).to(tl.int64)

    k_ptrs = (k_cache + block_id * k_stride_n + page_offset * k_stride_b + head_id * k_stride_h +
              dim_id * k_stride_d)
    v_ptrs = (v_cache + block_id * v_stride_n + page_offset * v_stride_b + head_id * v_stride_h +
              dim_id * v_stride_d)
    sk_ptrs = saved_k + batch_id * sk_stride_b + head_id * sk_stride_h + dim_id * sk_stride_d
    sv_ptrs = saved_v + batch_id * sv_stride_b + head_id * sv_stride_h + dim_id * sv_stride_d
    tl.store(sk_ptrs, tl.load(k_ptrs, mask=valid_elem), mask=valid_elem)
    tl.store(sv_ptrs, tl.load(v_ptrs, mask=valid_elem), mask=valid_elem)


@triton.jit
def _decode_kv_cache_restore_kernel(
    k_cache,
    v_cache,
    saved_k,
    saved_v,
    block_offsets,
    kv_seqlens,
    restore_mask,
    k_stride_n,
    k_stride_b,
    k_stride_h,
    k_stride_d,
    v_stride_n,
    v_stride_b,
    v_stride_h,
    v_stride_d,
    sk_stride_b,
    sk_stride_h,
    sk_stride_d,
    sv_stride_b,
    sv_stride_h,
    sv_stride_d,
    boff_stride_b,
    boff_stride_n,
    HEAD_DIM: tl.constexpr,
    TOTAL_ELEMS: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Restore one paged decode KV slot for masked batch rows."""
    tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)
    do_restore = tl.load(restore_mask + batch_id)
    if not do_restore:
        return

    offs = tile_id * BLOCK + tl.arange(0, BLOCK)
    valid_elem = offs < TOTAL_ELEMS

    head_id = offs // HEAD_DIM
    dim_id = offs - head_id * HEAD_DIM

    kv_seqlen = tl.maximum(tl.load(kv_seqlens + batch_id), 1)
    slot_id = kv_seqlen - 1
    block_idx = slot_id // KV_BLOCK_SIZE
    page_offset = slot_id - block_idx * KV_BLOCK_SIZE
    block_id = tl.load(block_offsets + batch_id * boff_stride_b + block_idx * boff_stride_n).to(tl.int64)

    k_ptrs = (k_cache + block_id * k_stride_n + page_offset * k_stride_b + head_id * k_stride_h +
              dim_id * k_stride_d)
    v_ptrs = (v_cache + block_id * v_stride_n + page_offset * v_stride_b + head_id * v_stride_h +
              dim_id * v_stride_d)
    sk_ptrs = saved_k + batch_id * sk_stride_b + head_id * sk_stride_h + dim_id * sk_stride_d
    sv_ptrs = saved_v + batch_id * sv_stride_b + head_id * sv_stride_h + dim_id * sv_stride_d
    tl.store(k_ptrs, tl.load(sk_ptrs, mask=valid_elem), mask=valid_elem)
    tl.store(v_ptrs, tl.load(sv_ptrs, mask=valid_elem), mask=valid_elem)


@triton.jit
def _decode_concept_state_update_kernel(
    last_raw_state_cache,
    last_final_state_cache,
    predicted_vectors,
    raw_states,
    state_ids,
    update_mask,
    raw_cache_stride_n,
    raw_cache_stride_l,
    raw_cache_stride_h,
    final_cache_stride_n,
    final_cache_stride_h,
    pred_stride_b,
    pred_stride_h,
    raw_stride_b,
    raw_stride_l,
    raw_stride_h,
    HIDDEN: tl.constexpr,
    RAW_ELEMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write final/raw concept state caches for valid boundary rows."""
    tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)
    state_id = tl.load(state_ids + batch_id)
    do_update = (state_id >= 0) & tl.load(update_mask + batch_id)
    if not do_update:
        return

    offs = tile_id * BLOCK + tl.arange(0, BLOCK)
    hidden_id = offs % HIDDEN

    final_mask = offs < HIDDEN
    pred_ptrs = predicted_vectors + batch_id * pred_stride_b + hidden_id * pred_stride_h
    final_ptrs = last_final_state_cache + state_id * final_cache_stride_n + hidden_id * final_cache_stride_h
    final_values = tl.load(pred_ptrs, mask=final_mask)
    tl.store(final_ptrs, final_values, mask=final_mask)

    raw_mask = offs < RAW_ELEMS
    layer_id = offs // HIDDEN
    raw_ptrs = raw_states + batch_id * raw_stride_b + layer_id * raw_stride_l + hidden_id * raw_stride_h
    raw_cache_ptrs = (last_raw_state_cache + state_id * raw_cache_stride_n + layer_id * raw_cache_stride_l +
                      hidden_id * raw_cache_stride_h)
    raw_values = tl.load(raw_ptrs, mask=raw_mask)
    tl.store(raw_cache_ptrs, raw_values, mask=raw_mask)


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


def prefill_chunk_state_update(
    source_states: torch.Tensor,
    token_start_ids: torch.Tensor,
    token_counts: torch.Tensor,
    num_concepts_total: int,
    chunk_size: int,
    merge_method: str,
    block: int = 1024,
):
    """Merge prefill source states into compact concept-source rows.

    Args:
        source_states: ``[num_tokens, num_sources, hidden]``.
        token_start_ids: first token row for each compact concept row.
        token_counts: number of tokens merged into each concept row.
        num_concepts_total: compact concept row count.
        chunk_size: ConceptLM chunk size.
        merge_method: ``meanpooling``, ``first``, or ``last``.
        block: Triton vector width over ``num_sources * hidden``.
    """
    assert source_states.is_cuda, 'ConceptLM prefill merge requires CUDA source states.'
    assert source_states.dim() == 3
    num_sources = source_states.size(1)
    hidden = source_states.size(2)
    concept_states = source_states.new_empty((num_concepts_total, num_sources, hidden))
    if num_concepts_total == 0:
        return concept_states

    total_elems = num_sources * hidden
    token_start_ids = token_start_ids.to(device=source_states.device, dtype=torch.long)
    token_counts = token_counts.to(device=source_states.device, dtype=torch.int32)
    grid = (triton.cdiv(total_elems, block), num_concepts_total)
    _prefill_chunk_state_update_kernel[grid](
        source_states,
        concept_states,
        token_start_ids,
        token_counts,
        *source_states.stride(),
        *concept_states.stride(),
        HIDDEN=hidden,
        TOTAL_ELEMS=total_elems,
        CHUNK_SIZE=int(chunk_size),
        MERGE_METHOD=_merge_method_id(merge_method),
        BLOCK=block,
        num_warps=8,
    )
    return concept_states


def prefill_state_cache_update(
    chunk_state_cache: torch.Tensor,
    last_raw_state_cache: torch.Tensor,
    last_final_state_cache: torch.Tensor,
    source_states: torch.Tensor,
    predicted_vectors: torch.Tensor,
    raw_states: torch.Tensor,
    state_ids: torch.Tensor,
    token_q_start_loc: torch.Tensor,
    token_q_seqlens: torch.Tensor,
    concept_q_start_loc: torch.Tensor,
    concept_q_seqlens: torch.Tensor,
    chunk_size: int,
    merge_method: str,
    block: int = 1024,
) -> None:
    """Seed ConceptLM decode state caches from prefill in one CUDA op."""
    assert chunk_state_cache.is_cuda, 'ConceptLM prefill state-cache update requires CUDA caches.'
    assert last_raw_state_cache.is_cuda and last_final_state_cache.is_cuda
    assert source_states.is_cuda and predicted_vectors.is_cuda and raw_states.is_cuda
    assert source_states.dim() == 3 and predicted_vectors.dim() == 2 and raw_states.dim() == 3
    assert chunk_state_cache.shape[1:] == source_states.shape[1:]
    assert last_final_state_cache.size(1) == predicted_vectors.size(1)
    assert last_raw_state_cache.shape[1:] == raw_states.shape[1:]

    batch_size = token_q_seqlens.numel()
    if batch_size == 0:
        return

    hidden = source_states.size(2)
    source_elems = source_states.size(1) * hidden
    raw_elems = raw_states.size(1) * raw_states.size(2)
    max_elems = max(source_elems, hidden, raw_elems)
    state_ids = state_ids.to(device=source_states.device, dtype=torch.long).reshape(-1)
    token_q_start_loc = token_q_start_loc.to(device=source_states.device)
    token_q_seqlens = token_q_seqlens.to(device=source_states.device)
    concept_q_start_loc = concept_q_start_loc.to(device=source_states.device)
    concept_q_seqlens = concept_q_seqlens.to(device=source_states.device)
    grid = (triton.cdiv(max_elems, block), batch_size)
    _prefill_state_cache_update_kernel[grid](
        chunk_state_cache,
        last_raw_state_cache,
        last_final_state_cache,
        source_states,
        predicted_vectors,
        raw_states,
        state_ids,
        token_q_start_loc,
        token_q_seqlens,
        concept_q_start_loc,
        concept_q_seqlens,
        *chunk_state_cache.stride(),
        *last_raw_state_cache.stride(),
        *last_final_state_cache.stride(),
        *source_states.stride(),
        *predicted_vectors.stride(),
        *raw_states.stride(),
        HIDDEN=hidden,
        SOURCE_ELEMS=source_elems,
        RAW_ELEMS=raw_elems,
        CHUNK_SIZE=int(chunk_size),
        MERGE_METHOD=_merge_method_id(merge_method),
        BLOCK=block,
        num_warps=8,
    )


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
        Tuple ``(concept_inputs, update_mask)``. ``concept_inputs`` is zero for
        non-boundary rows. ``update_mask`` is ``True`` only for valid boundary
        rows.
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
    update_mask = torch.empty((batch_size, ), dtype=torch.bool, device=current_source_states.device)
    grid = (triton.cdiv(total_elems, block), batch_size)
    _decode_chunk_state_update_kernel[grid](
        chunk_source_state_cache,
        current_source_states,
        state_ids,
        position_ids,
        concept_inputs,
        update_mask,
        *chunk_source_state_cache.stride(),
        *current_source_states.stride(),
        *concept_inputs.stride(),
        HIDDEN=hidden,
        TOTAL_ELEMS=total_elems,
        CHUNK_SIZE=int(chunk_size),
        MERGE_METHOD=_merge_method_id(merge_method),
        BLOCK=block,
        num_warps=8,
    )
    return concept_inputs, update_mask


def decode_kv_cache_snapshot(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_offsets: torch.Tensor,
    kv_seqlens: torch.Tensor,
    block: int = 1024,
):
    """Snapshot the current decode KV slot for each batch row.

    The slot is ``max(kv_seqlen, 1) - 1`` in the paged cache. This is used by
    ConceptLM graph-safe decode to undo dummy all-row concept predictor writes
    for non-boundary rows.
    """
    assert k_cache.is_cuda and v_cache.is_cuda, 'ConceptLM KV snapshot requires CUDA caches.'
    assert k_cache.dim() == 4 and v_cache.dim() == 4
    assert k_cache.shape[:3] == v_cache.shape[:3]
    assert k_cache.shape[-1] == v_cache.shape[-1]
    assert block_offsets.is_cuda and kv_seqlens.is_cuda

    batch_size = kv_seqlens.numel()
    num_heads = k_cache.size(2)
    head_dim = k_cache.size(3)
    total_elems = num_heads * head_dim
    saved_k = torch.empty((batch_size, num_heads, head_dim), dtype=k_cache.dtype, device=k_cache.device)
    saved_v = torch.empty((batch_size, num_heads, head_dim), dtype=v_cache.dtype, device=v_cache.device)
    grid = (triton.cdiv(total_elems, block), batch_size)
    _decode_kv_cache_snapshot_kernel[grid](
        k_cache,
        v_cache,
        block_offsets,
        kv_seqlens,
        saved_k,
        saved_v,
        *k_cache.stride(),
        *v_cache.stride(),
        *block_offsets.stride(),
        *saved_k.stride(),
        *saved_v.stride(),
        HEAD_DIM=head_dim,
        TOTAL_ELEMS=total_elems,
        KV_BLOCK_SIZE=k_cache.size(1),
        BLOCK=block,
        num_warps=8,
    )
    return saved_k, saved_v


def decode_kv_cache_restore(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    saved_k: torch.Tensor,
    saved_v: torch.Tensor,
    block_offsets: torch.Tensor,
    kv_seqlens: torch.Tensor,
    restore_mask: torch.Tensor,
    block: int = 1024,
) -> None:
    """Restore the current decode KV slot for masked batch rows."""
    assert k_cache.is_cuda and v_cache.is_cuda, 'ConceptLM KV restore requires CUDA caches.'
    assert saved_k.is_cuda and saved_v.is_cuda
    assert saved_k.shape == (kv_seqlens.numel(), k_cache.size(2), k_cache.size(3))
    assert saved_v.shape == (kv_seqlens.numel(), v_cache.size(2), v_cache.size(3))
    assert restore_mask.numel() == kv_seqlens.numel()

    batch_size = kv_seqlens.numel()
    num_heads = k_cache.size(2)
    head_dim = k_cache.size(3)
    total_elems = num_heads * head_dim
    restore_mask = restore_mask.to(device=k_cache.device, dtype=torch.bool)
    grid = (triton.cdiv(total_elems, block), batch_size)
    _decode_kv_cache_restore_kernel[grid](
        k_cache,
        v_cache,
        saved_k,
        saved_v,
        block_offsets,
        kv_seqlens,
        restore_mask,
        *k_cache.stride(),
        *v_cache.stride(),
        *saved_k.stride(),
        *saved_v.stride(),
        *block_offsets.stride(),
        HEAD_DIM=head_dim,
        TOTAL_ELEMS=total_elems,
        KV_BLOCK_SIZE=k_cache.size(1),
        BLOCK=block,
        num_warps=8,
    )


def decode_concept_state_update(
    last_raw_state_cache: torch.Tensor,
    last_final_state_cache: torch.Tensor,
    predicted_vectors: torch.Tensor,
    raw_states: torch.Tensor,
    state_ids: torch.Tensor,
    update_mask: torch.Tensor,
    block: int = 1024,
) -> None:
    """Write final/raw concept states for valid boundary rows."""
    assert last_raw_state_cache.is_cuda, 'ConceptLM concept-state update requires CUDA caches.'
    assert last_final_state_cache.is_cuda
    assert predicted_vectors.is_cuda and raw_states.is_cuda
    assert raw_states.dim() == 3
    assert last_raw_state_cache.shape[1:] == raw_states.shape[1:]
    assert last_final_state_cache.size(1) == predicted_vectors.size(1)
    assert predicted_vectors.size(0) == raw_states.size(0) == state_ids.numel() == update_mask.numel()

    batch_size = predicted_vectors.size(0)
    hidden = predicted_vectors.size(1)
    raw_elems = raw_states.size(1) * raw_states.size(2)
    max_elems = max(hidden, raw_elems)
    state_ids = state_ids.to(device=predicted_vectors.device, dtype=torch.long)
    update_mask = update_mask.to(device=predicted_vectors.device, dtype=torch.bool)
    grid = (triton.cdiv(max_elems, block), batch_size)
    _decode_concept_state_update_kernel[grid](
        last_raw_state_cache,
        last_final_state_cache,
        predicted_vectors,
        raw_states,
        state_ids,
        update_mask,
        *last_raw_state_cache.stride(),
        *last_final_state_cache.stride(),
        *predicted_vectors.stride(),
        *raw_states.stride(),
        HIDDEN=hidden,
        RAW_ELEMS=raw_elems,
        BLOCK=block,
        num_warps=8,
    )

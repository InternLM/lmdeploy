# Copyright (c) OpenMMLab. All rights reserved.
"""Triton kernels for linear-chain GDN MTP ReplaySSM verification."""

import torch
import triton
import triton.language as tl


def _get_mtp_reconstruct_policy(B: int, H: int, HV: int, K: int, V: int, T: int,
                               channelwise: bool) -> tuple[bool, bool]:
    """Measured checkpoint eviction/grid policy; unrelated shapes keep
    defaults."""
    # Limit the checkpoint streaming hint to the measured small-batch geometry;
    # applying it to larger batches can regress performance.
    evict_checkpoint = channelwise and B in (8, 16) and (H, HV, K, V, T) == (64, 64, 128, 128, 6)
    # Small grids retain the batch-major order; larger grids benefit from
    # traversing contiguous heads before moving to the next state slot.
    head_major = channelwise and (K, V, T) == (128, 128, 6) and B * HV >= 512
    return evict_checkpoint, head_major


@triton.jit
def _snapshot_replayssm_raw_kernel(
    raw_v,
    raw_k,
    gates,
    betas,
    cursor,
    indices,
    snap_v,
    snap_k,
    snap_g,
    snap_beta,
    stride_v,
    stride_k,
    stride_g,
    stride_beta,
    stride_cursor,
    stride_indices,
    H: tl.constexpr,
    HV: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    CHANNELWISE_G: tl.constexpr,
):
    row = tl.program_id(0)
    hv = tl.program_id(1)
    h = hv // (HV // H)
    slot = tl.load(indices + row * stride_indices).to(tl.int64)
    if slot < 0:
        return
    if tl.load(cursor + slot * stride_cursor + 2) == 0:
        return
    t = tl.arange(0, R)
    ov = tl.arange(0, BV)
    vv = tl.load(raw_v + slot * stride_v + (hv * R + t[:, None]) * V + ov[None, :], mask=ov[None, :] < V, other=0)
    tl.store(snap_v + ((row * HV + hv) * R + t[:, None]) * V + ov[None, :], vv, mask=ov[None, :] < V)
    if CHANNELWISE_G:
        ok = tl.arange(0, BK)
        gg = tl.load(gates + slot * stride_g + (hv * R + t[:, None]) * K + ok[None, :],
                     mask=ok[None, :] < K, other=0.0)
        tl.store(snap_g + ((row * HV + hv) * R + t[:, None]) * K + ok[None, :],
                 gg, mask=ok[None, :] < K)
    else:
        gg = tl.load(gates + slot * stride_g + hv * R + t)
        tl.store(snap_g + (row * HV + hv) * R + t, gg)
    bb = tl.load(betas + slot * stride_beta + hv * R + t)
    tl.store(snap_beta + (row * HV + hv) * R + t, bb)
    if hv % (HV // H) == 0:
        ok = tl.arange(0, BK)
        kk = tl.load(raw_k + slot * stride_k + (h * R + t[:, None]) * K + ok[None, :], mask=ok[None, :] < K, other=0)
        tl.store(snap_k + ((row * H + h) * R + t[:, None]) * K + ok[None, :], kk, mask=ok[None, :] < K)


@triton.jit
def gdn_replayssm_exact_fold_kernel(
    checkpoint,
    raw_v_ring,
    raw_k_ring,
    g_ring,
    beta_ring,
    cursor,
    state_indices,
    stride_state_slot,
    stride_raw_v_slot,
    stride_raw_k_slot,
    stride_g_slot,
    stride_beta_slot,
    stride_cursor_slot,
    stride_indices,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    R: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    FOLD_PARTIAL: tl.constexpr,
    CHANNELWISE_G: tl.constexpr,
):
    """Fold only previously committed raw history for rows marked flush."""
    i_v = tl.program_id(0)
    i_b = tl.program_id(1)
    i_hv = tl.program_id(2)
    i_h = i_hv // (HV // H)
    state_idx = tl.load(state_indices + i_b * stride_indices).to(tl.int64)
    if state_idx < 0:
        return
    p_cursor = cursor + state_idx * stride_cursor_slot
    if (not FOLD_PARTIAL) and tl.load(p_cursor + 2) == 0:
        return

    write_pos = tl.load(p_cursor).to(tl.int32)
    if write_pos == 0:
        return
    cache_base = tl.load(p_cursor + 1).to(tl.int32)
    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, BK)
    mask_v = o_v < V
    mask_k = o_k < K
    mask_h = mask_v[:, None] & mask_k[None, :]
    p_h = (checkpoint + state_idx * stride_state_slot + i_hv * V * K
           + o_v[:, None] * K + o_k[None, :])
    h = tl.load(p_h, mask=mask_h, other=0.0).to(tl.float32)

    for t in range(R):
        if t < write_pos:
            phys = (cache_base + t) & (R - 1)
            raw_k = tl.load(raw_k_ring + state_idx * stride_raw_k_slot
                            + (i_h * R + phys) * K + o_k,
                            mask=mask_k, other=0.0).to(tl.float32)
            raw_v = tl.load(raw_v_ring + state_idx * stride_raw_v_slot
                            + (i_hv * R + phys) * V + o_v,
                            mask=mask_v, other=0.0).to(tl.float32)
            if CHANNELWISE_G:
                gate = tl.load(g_ring + state_idx * stride_g_slot + (i_hv * R + phys) * K + o_k,
                               mask=mask_k, other=0.0).to(tl.float32)[None, :]
            else:
                gate = tl.load(g_ring + state_idx * stride_g_slot + i_hv * R + phys).to(tl.float32)
            beta = tl.load(beta_ring + state_idx * stride_beta_slot + i_hv * R + phys).to(tl.float32)
            key = raw_k / tl.sqrt(tl.sum(raw_k * raw_k) + 1e-6)
            h *= tl.exp(gate)
            delta = (raw_v - tl.sum(h * key[None, :], axis=1)) * beta
            h += delta[:, None] * key[None, :]

    tl.store(p_h, h, mask=mask_h)


@triton.jit
def gdn_replayssm_spec_circular_kernel(
    q,
    k,
    v,
    g,
    beta,
    out,
    checkpoint,
    d_ring,
    k_ring,
    g_ring,
    raw_v_ring,
    raw_k_ring,
    beta_ring,
    cursor,
    state_indices,
    q_seqlens,
    snap_v, snap_k, snap_g, snap_beta,
    stride_q_b,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_k_b,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_v_b,
    stride_v_t,
    stride_v_h,
    stride_v_d,
    stride_g_b,
    stride_g_t,
    stride_beta_b,
    stride_beta_t,
    stride_out_b,
    stride_out_t,
    stride_state_slot,
    stride_d_slot,
    stride_k_ring_slot,
    stride_g_ring_slot,
    stride_raw_v_slot,
    stride_raw_k_slot,
    stride_beta_ring_slot,
    stride_cursor_slot,
    stride_indices,
    stride_qlen,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    R: tl.constexpr,
    TMAX: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BC: tl.constexpr,
    CHANNELWISE_G: tl.constexpr,
    EVICT_CHECKPOINT: tl.constexpr = False,
    HEAD_MAJOR: tl.constexpr = False,
):
    """Fold/reconstruct and verify after the raw-history snapshot barrier.

    Flush CTAs read the immutable snapshot, never the live raw rings: another value tile or grouped value head may
    already be writing the new speculative window into the same ring positions.
    """
    i_v = tl.program_id(0)
    i_b = tl.program_id(2) if HEAD_MAJOR else tl.program_id(1)
    i_hv = tl.program_id(1) if HEAD_MAJOR else tl.program_id(2)
    i_h = i_hv // (HV // H)
    state_idx = tl.load(state_indices + i_b * stride_indices).to(tl.int64)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_v = o_v < V
    # Triton requires arange widths to be powers of two. TMAX is a public
    # speculative-decoding geometry and may be any positive integer, so use a
    # padded vector axis and mask it back to the logical verify width.
    o_t = tl.arange(0, BT)
    token_in_bounds = o_t < TMAX
    qlen = tl.load(q_seqlens + i_b * stride_qlen).to(tl.int32)
    valid_t = token_in_bounds & (o_t < qlen)
    if state_idx < 0:
        p_out = out + i_b * stride_out_b + o_t[:, None] * stride_out_t + i_hv * V + o_v[None, :]
        tl.store(p_out, 0.0,
                 mask=token_in_bounds[:, None] & mask_v[None, :])
        return

    p_cursor = cursor + state_idx * stride_cursor_slot
    row_flush = tl.load(p_cursor + 2) != 0
    stored_write_pos = tl.load(p_cursor).to(tl.int32)
    stored_cache_base = tl.load(p_cursor + 1).to(tl.int32)
    write_pos = tl.where(row_flush, 0, stored_write_pos)
    cache_base = tl.where(row_flush, 0, stored_cache_base)
    p_out_all = out + i_b * stride_out_b + o_t[:, None] * stride_out_t + i_hv * V + o_v[None, :]
    tl.store(p_out_all, 0.0,
             mask=(token_in_bounds & ~valid_t)[:, None] & mask_v[None, :])

    o_k = tl.arange(0, BK)
    mask_k = o_k < K
    mask_h = mask_v[:, None] & mask_k[None, :]
    p_h = (checkpoint + state_idx * stride_state_slot + i_hv * V * K
           + o_v[:, None] * K + o_k[None, :])
    h = tl.load(p_h, mask=mask_h, other=0.0,
                eviction_policy='evict_first' if EVICT_CHECKPOINT else '').to(tl.float32)

    if row_flush:
        # Materialize only the committed raw prefix into the checkpoint.  The
        # current TMAX candidates remain tentative and are written to the
        # fresh ring below; rejection sampling owns the later cursor commit.
        for t in range(R):
            if t < stored_write_pos:
                phys = (stored_cache_base + t) & (R - 1)
                raw_k = tl.load(snap_k + i_b * H * R * K
                                + (i_h * R + phys) * K + o_k,
                                mask=mask_k, other=0.0).to(tl.float32)
                raw_v = tl.load(snap_v + i_b * HV * R * V
                                + (i_hv * R + phys) * V + o_v,
                                mask=mask_v, other=0.0).to(tl.float32)
                if CHANNELWISE_G:
                    gate = tl.load(snap_g + ((i_b * HV + i_hv) * R + phys) * K + o_k,
                                   mask=mask_k, other=0.0).to(tl.float32)[None, :]
                else:
                    gate = tl.load(snap_g + i_b * HV * R
                                   + i_hv * R + phys).to(tl.float32)
                beta_raw = tl.load(snap_beta + i_b * HV * R
                                   + i_hv * R + phys).to(tl.float32)
                key = raw_k / tl.sqrt(tl.sum(raw_k * raw_k) + 1e-6)
                h *= tl.exp(gate)
                delta = (raw_v - tl.sum(h * key[None, :], axis=1)) * beta_raw
                h += delta[:, None] * key[None, :]
        tl.store(p_h, h, mask=mask_h)
    else:
        o_c = tl.arange(0, BC)
        cache_valid = o_c < stored_write_pos
        phys_c = (stored_cache_base + o_c) & (R - 1)
        if CHANNELWISE_G:
            # Reconstruct only the committed prefix from FP32 corrections.
            # This changes association, not the raw exact-fold authority.
            for history_index in range(stored_write_pos):
                history_slot = (stored_cache_base + history_index) & (R - 1)
                gate_hist = tl.load(g_ring + state_idx * stride_g_ring_slot
                                    + (i_hv * R + history_slot) * K + o_k,
                                    mask=mask_k, other=0.0).to(tl.float32)
                delta_hist = tl.load(d_ring + state_idx * stride_d_slot
                                     + (i_hv * R + history_slot) * V + o_v,
                                     mask=mask_v, other=0.0).to(tl.float32)
                key_hist = tl.load(k_ring + state_idx * stride_k_ring_slot
                                   + (i_h * R + history_slot) * K + o_k,
                                   mask=mask_k, other=0.0).to(tl.float32)
                h = h * tl.exp(gate_hist)[None, :] + delta_hist[:, None] * key_hist[None, :]
        else:
            gates = tl.load(g_ring + state_idx * stride_g_ring_slot
                            + i_hv * R + phys_c,
                            mask=cache_valid, other=0.0).to(tl.float32)
            prefix = tl.cumsum(gates, axis=0)
            total = tl.sum(gates, axis=0)
            decay = tl.where(cache_valid, tl.exp(total - prefix), 0.0)
            d_hist = tl.load(d_ring + state_idx * stride_d_slot
                             + (i_hv * R + phys_c[None, :]) * V + o_v[:, None],
                             mask=mask_v[:, None] & cache_valid[None, :],
                             other=0.0).to(q.dtype.element_ty)
            k_hist = tl.load(k_ring + state_idx * stride_k_ring_slot
                             + (i_h * R + phys_c[:, None]) * K + o_k[None, :],
                             mask=cache_valid[:, None] & mask_k[None, :],
                             other=0.0).to(q.dtype.element_ty)
            h = h * tl.exp(total) + tl.dot(
                (d_hist * decay[None, :]).to(q.dtype.element_ty),
                k_hist,
                input_precision='ieee').to(tl.float32)

    write_k = (i_v == 0) and (i_hv == i_h * (HV // H))
    for t in range(TMAX):
        if t < qlen:
            q_raw = tl.load(q + i_b * stride_q_b + t * stride_q_t
                            + i_h * stride_q_h + o_k * stride_q_d,
                            mask=mask_k, other=0.0).to(tl.float32)
            k_raw = tl.load(k + i_b * stride_k_b + t * stride_k_t
                            + i_h * stride_k_h + o_k * stride_k_d,
                            mask=mask_k, other=0.0).to(tl.float32)
            query_t = q_raw / tl.sqrt(tl.sum(q_raw * q_raw) + 1e-6) * scale
            key_t = k_raw / tl.sqrt(tl.sum(k_raw * k_raw) + 1e-6)
            value_t = tl.load(v + i_b * stride_v_b + t * stride_v_t
                              + i_hv * stride_v_h + o_v * stride_v_d,
                              mask=mask_v, other=0.0).to(tl.float32)
            if CHANNELWISE_G:
                gate_t = tl.load(g + i_b * stride_g_b + t * stride_g_t + i_hv * K + o_k,
                                 mask=mask_k, other=0.0).to(tl.float32)
            else:
                gate_t = tl.load(g + i_b * stride_g_b + t * stride_g_t + i_hv).to(tl.float32)
            beta_t = tl.load(beta + i_b * stride_beta_b + t * stride_beta_t + i_hv).to(tl.float32)
            if CHANNELWISE_G:
                h *= tl.exp(gate_t)[None, :]
            else:
                h *= tl.exp(gate_t)
            delta = (value_t - tl.sum(h * key_t[None, :], axis=1)) * beta_t
            h += delta[:, None] * key_t[None, :]
            result = tl.sum(h * query_t[None, :], axis=1)
            tl.store(out + i_b * stride_out_b + t * stride_out_t + i_hv * V + o_v,
                     result.to(out.dtype.element_ty), mask=mask_v)

            phys = (cache_base + write_pos + t) & (R - 1)
            tl.store(d_ring + state_idx * stride_d_slot + (i_hv * R + phys) * V + o_v,
                     delta.to(d_ring.dtype.element_ty), mask=mask_v)
            tl.store(raw_v_ring + state_idx * stride_raw_v_slot + (i_hv * R + phys) * V + o_v,
                     value_t.to(raw_v_ring.dtype.element_ty), mask=mask_v)
            if write_k:
                tl.store(k_ring + state_idx * stride_k_ring_slot + (i_h * R + phys) * K + o_k,
                         key_t.to(k_ring.dtype.element_ty), mask=mask_k)
                tl.store(raw_k_ring + state_idx * stride_raw_k_slot + (i_h * R + phys) * K + o_k,
                         k_raw.to(raw_k_ring.dtype.element_ty), mask=mask_k)
            if i_v == 0:
                if CHANNELWISE_G:
                    tl.store(g_ring + state_idx * stride_g_ring_slot + (i_hv * R + phys) * K + o_k,
                             gate_t, mask=mask_k)
                else:
                    tl.store(g_ring + state_idx * stride_g_ring_slot + i_hv * R + phys, gate_t)
                tl.store(beta_ring + state_idx * stride_beta_ring_slot + i_hv * R + phys, beta_t)


@triton.jit
def commit_gdn_replayssm_spec_cursor_kernel(
    cursor,
    state_indices,
    q_seqlens,
    num_rejected_tokens,
    stride_cursor_slot,
    stride_indices,
    stride_qlen,
    stride_rejected,
    n_rows,
    R: tl.constexpr,
    TMAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = row < n_rows
    state_idx = tl.load(state_indices + row * stride_indices, mask=mask, other=-1).to(tl.int64)
    valid = mask & (state_idx >= 0)
    p = cursor + state_idx * stride_cursor_slot
    was_flush = tl.load(p + 2, mask=valid, other=0) != 0
    old_pos = tl.load(p, mask=valid, other=0).to(tl.int32)
    qlen = tl.load(q_seqlens + row * stride_qlen, mask=mask, other=0).to(tl.int32)
    rejected = tl.load(num_rejected_tokens + row * stride_rejected,
                       mask=mask, other=0).to(tl.int32)
    accepted = tl.maximum(0, qlen - rejected)
    new_pos = tl.where(was_flush, accepted, old_pos + accepted)
    tl.store(p, new_pos, mask=valid)
    tl.store(p + 1, 0, mask=valid)
    tl.store(p + 2, new_pos > (R - TMAX), mask=valid)


def _validate_spec_tensors(q, k, v, g, beta, state, q_seqlens):
    B, T, H, K = q.shape
    _, Tv, HV, V = v.shape
    gate_shape = (B, T, HV, K) if g.ndim == 4 else (B, T, HV)
    if T != Tv or k.shape != q.shape or g.shape != gate_shape or beta.shape != (B, T, HV):
        raise ValueError('invalid q/k/v/g/beta shapes for MTP ReplaySSM')
    if T != state.cursor.Tmax:
        raise ValueError(f'MTP verify width {T} does not match configured Tmax={state.cursor.Tmax}')
    if q_seqlens.shape != (B, ):
        raise ValueError('q_seqlens must have shape [B]')
    if state.checkpoint.dtype != torch.float32:
        raise ValueError('MTP ReplaySSM checkpoint must be float32')
    return B, T, H, K, HV, V


def make_gdn_replayssm_spec_workspace(state, batch: int) -> tuple[torch.Tensor, ...]:
    """Allocate transient raw-history buffers, not persistent cache state.

    The CUDA backend owns one workspace per bounded capacity/geometry and retains it for the lifetime of the model. No
    new workspace is allocated in the production steady-state path. Standalone callers may pass their own workspace or
    use the allocation fallback in the wrapper.
    """
    _, hv, vdim, kdim = state.checkpoint.shape
    h = state.raw_k_ring.shape[1]
    r = state.cursor.R
    kwargs = dict(device=state.checkpoint.device)
    return (
        torch.empty((batch, hv, r, vdim), dtype=state.raw_v_ring.dtype, **kwargs),
        torch.empty((batch, h, r, kdim), dtype=state.raw_k_ring.dtype, **kwargs),
        torch.empty((batch, *state.g_ring.shape[1:]), dtype=state.g_ring.dtype, **kwargs),
        torch.empty((batch, hv, r), dtype=state.beta_ring.dtype, **kwargs),
    )


def fused_recurrent_gdn_replayssm_spec(q: torch.Tensor,
                                       k: torch.Tensor,
                                       v: torch.Tensor,
                                       g: torch.Tensor,
                                       beta: torch.Tensor,
                                       state,
                                       state_indices: torch.Tensor,
                                       q_seqlens: torch.Tensor,
                                       scale: float | None = None,
                                       workspace: tuple[torch.Tensor, ...] | None = None):
    """Snapshot committed raw inputs, then fold/reconstruct and verify."""
    B, T, H, K, HV, V = _validate_spec_tensors(q, k, v, g, beta, state, q_seqlens)
    R = state.cursor.R
    BT = triton.next_power_of_2(T)
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    BC = max(16, triton.next_power_of_2(R))
    if scale is None:
        scale = K**-0.5
    out = torch.empty_like(v, memory_format=torch.contiguous_format)
    if workspace is None:
        workspace = make_gdn_replayssm_spec_workspace(state, B)
    snap_v, snap_k, snap_g, snap_beta = workspace
    shapes = ((HV, R, V), (H, R, K), tuple(state.g_ring.shape[1:]), (HV, R))
    if state.g_ring.shape[1:] != ((HV, R, K) if g.ndim == 4 else (HV, R)):
        raise ValueError('ReplaySSM gate and history geometry disagree')
    dtypes = (state.raw_v_ring.dtype, state.raw_k_ring.dtype, state.g_ring.dtype, state.beta_ring.dtype)
    if any(tensor.shape[0] < B or tensor.shape[1:] != shape
           or tensor.dtype != dtype or tensor.device != q.device or not tensor.is_contiguous()
           for tensor, shape, dtype in zip(workspace, shapes, dtypes)):
        raise ValueError('MTP ReplaySSM workspace does not match the verify batch')
    _snapshot_replayssm_raw_kernel[(B, HV)](
        state.raw_v_ring, state.raw_k_ring, state.g_ring, state.beta_ring,
        state.cursor.packed_cursor, state_indices, snap_v, snap_k, snap_g, snap_beta,
        state.raw_v_ring.stride(0), state.raw_k_ring.stride(0), state.g_ring.stride(0),
        state.beta_ring.stride(0), state.cursor.packed_cursor.stride(0), state_indices.stride(0),
        H=H, HV=HV, R=R, K=K, V=V, BK=BK, BV=triton.next_power_of_2(V),
        CHANNELWISE_G=g.ndim == 4, num_warps=4)
    args = (
        q, k, v, g, beta, out, state.checkpoint, state.d_ring,
        state.k_ring, state.g_ring, state.raw_v_ring, state.raw_k_ring,
        state.beta_ring, state.cursor.packed_cursor, state_indices, q_seqlens,
        snap_v, snap_k, snap_g, snap_beta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        g.stride(0), g.stride(1), beta.stride(0), beta.stride(1),
        out.stride(0), out.stride(1), state.checkpoint.stride(0),
        state.d_ring.stride(0), state.k_ring.stride(0), state.g_ring.stride(0),
        state.raw_v_ring.stride(0), state.raw_k_ring.stride(0),
        state.beta_ring.stride(0), state.cursor.packed_cursor.stride(0),
        state_indices.stride(0), q_seqlens.stride(0), scale,
    )
    evict_checkpoint, head_major = _get_mtp_reconstruct_policy(B, H, HV, K, V, T, g.ndim == 4)
    grid = (triton.cdiv(V, BV), HV, B) if head_major else (triton.cdiv(V, BV), B, HV)
    gdn_replayssm_spec_circular_kernel[grid](
        *args, H=H, HV=HV, K=K, V=V, R=R, TMAX=T, BT=BT,
        BK=BK, BV=BV, BC=BC, CHANNELWISE_G=g.ndim == 4, EVICT_CHECKPOINT=evict_checkpoint, HEAD_MAJOR=head_major,
        num_warps=1,
        num_stages=1)
    return out, state.checkpoint


def fold_gdn_replayssm_mtp_history(state, state_indices: torch.Tensor) -> None:
    """Exactly materialize committed raw history before continuation
    prefill."""
    _, hv, vdim, kdim = state.checkpoint.shape
    h = state.raw_k_ring.shape[1]
    ring_len = state.cursor.R
    bk = triton.next_power_of_2(kdim)
    bv, num_warps, num_stages = (32, 2, 3) if state_indices.numel() <= 1 else (16, 1, 1)
    bv = min(triton.next_power_of_2(vdim), bv)
    grid = (triton.cdiv(vdim, bv), state_indices.numel(), hv)
    gdn_replayssm_exact_fold_kernel[grid](
        state.checkpoint, state.raw_v_ring, state.raw_k_ring, state.g_ring,
        state.beta_ring, state.cursor.packed_cursor, state_indices,
        state.checkpoint.stride(0), state.raw_v_ring.stride(0),
        state.raw_k_ring.stride(0), state.g_ring.stride(0),
        state.beta_ring.stride(0), state.cursor.packed_cursor.stride(0),
        state_indices.stride(0), H=h, HV=hv, K=kdim, V=vdim, R=ring_len,
        BK=bk, BV=bv, FOLD_PARTIAL=True, CHANNELWISE_G=state.g_ring.ndim == 4,
        num_warps=num_warps, num_stages=num_stages)


def commit_gdn_replayssm_spec_cursor(packed_cursor: torch.Tensor,
                                     state_indices: torch.Tensor,
                                     q_seqlens: torch.Tensor,
                                     num_rejected_tokens: torch.Tensor,
                                     buffer_len: int,
                                     tmax: int) -> None:
    """Commit the rejection-owned accepted target-input prefix on device."""
    n_rows = state_indices.numel()
    block = 256
    commit_gdn_replayssm_spec_cursor_kernel[(triton.cdiv(n_rows, block), )](
        packed_cursor, state_indices, q_seqlens, num_rejected_tokens,
        packed_cursor.stride(0), state_indices.stride(0), q_seqlens.stride(0),
        num_rejected_tokens.stride(0), n_rows,
        R=buffer_len, TMAX=tmax, BLOCK=block)

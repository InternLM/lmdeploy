# Copyright (c) OpenMMLab. All rights reserved.
"""Triton kernels for standard Gated DeltaNet ReplaySSM decode."""

import torch
import triton
import triton.language as tl

from .gated_delta_replayssm_config import (
    RESET_CURSOR_SM90_CONFIG,
    STANDARD_CURSOR_SM90_CONFIG,
    get_standard_replayssm_config,
)


@triton.jit
def _gdn_replayssm_decode_kernel(
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
    cursor,
    state_indices,
    force_flush,
    stride_q_b,
    stride_q_h,
    stride_k_b,
    stride_k_h,
    stride_v_b,
    stride_v_h,
    stride_g_b,
    stride_beta_b,
    stride_state_slot,
    stride_d_slot,
    stride_k_ring_slot,
    stride_g_ring_slot,
    stride_cursor_slot,
    stride_indices,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BC: tl.constexpr,
    NK: tl.constexpr,
    BKT: tl.constexpr,
    HAS_FORCE_FLUSH: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_b = tl.program_id(1)
    i_hv = tl.program_id(2)
    i_h = i_hv // (HV // H)

    o_v = i_v * BV + tl.arange(0, BV)
    o_c = tl.arange(0, BC)
    mask_v = o_v < V
    state_idx = tl.load(state_indices + i_b * stride_indices).to(tl.int64)
    p_out = out + (i_b * HV + i_hv) * V + o_v
    if state_idx < 0:
        tl.store(p_out, 0.0, mask=mask_v)
        return

    write_pos = tl.load(cursor + state_idx * stride_cursor_slot).to(tl.int32)
    is_flush = write_pos == L - 1
    if HAS_FORCE_FLUSH:
        is_flush = is_flush | (tl.load(force_flush + i_b) != 0)
    cache_valid = o_c < write_pos

    g_cur = tl.load(g + i_b * stride_g_b + i_hv).to(tl.float32)
    beta_cur = tl.load(beta + i_b * stride_beta_b + i_hv).to(tl.float32)
    alpha_cur = tl.exp(g_cur)

    p_g_hist = g_ring + state_idx * stride_g_ring_slot + i_hv * L + o_c
    g_hist = tl.load(p_g_hist, mask=cache_valid, other=0.0).to(tl.float32)
    g_prefix = tl.cumsum(g_hist, axis=0)
    g_total = tl.sum(g_hist, axis=0)
    replay_decay = tl.where(cache_valid, tl.exp(g_total - g_prefix), 0.0)
    total_decay = tl.exp(g_total)

    p_d_hist = (d_ring + state_idx * stride_d_slot
                + (i_hv * L + o_c[None, :]) * V + o_v[:, None])
    d_hist = tl.load(p_d_hist,
                     mask=mask_v[:, None] & cache_valid[None, :],
                     other=0.0).to(tl.float32)
    d_tc = (d_hist * replay_decay[None, :]).to(q.dtype.element_ty)

    p_q = q + i_b * stride_q_b + i_h * stride_q_h
    p_k = k + i_b * stride_k_b + i_h * stride_k_h
    o_k_full = tl.arange(0, BK)
    mask_k_full = o_k_full < K
    q_full = tl.load(p_q + o_k_full, mask=mask_k_full, other=0.0).to(tl.float32)
    k_full = tl.load(p_k + o_k_full, mask=mask_k_full, other=0.0).to(tl.float32)
    q_rnorm = 1.0 / tl.sqrt(tl.sum(q_full * q_full) + 1e-6)
    k_rnorm = 1.0 / tl.sqrt(tl.sum(k_full * k_full) + 1e-6)

    b_v = tl.load(v + i_b * stride_v_b + i_hv * stride_v_h + o_v,
                  mask=mask_v, other=0.0).to(tl.float32)
    state_q = tl.zeros([BV], dtype=tl.float32)
    state_k = tl.zeros([BV], dtype=tl.float32)
    cur_kq = tl.zeros([1], dtype=tl.float32)
    write_k = (not is_flush) and (i_v == 0) and (i_hv == i_h * (HV // H))

    for kk in range(NK):
        o_kt = kk * BKT + tl.arange(0, BKT)
        mask_kt = o_kt < K
        q_cur = tl.load(p_q + o_kt, mask=mask_kt, other=0.0).to(tl.float32) * q_rnorm * scale
        k_cur = tl.load(p_k + o_kt, mask=mask_kt, other=0.0).to(tl.float32) * k_rnorm
        cur_kq += tl.sum(q_cur * k_cur)

        p_h0 = (checkpoint + state_idx * stride_state_slot + i_hv * V * K
                + o_v[:, None] * K + o_kt[None, :])
        h0 = tl.load(p_h0, mask=mask_v[:, None] & mask_kt[None, :], other=0.0).to(tl.float32)
        p_k_hist = (k_ring + state_idx * stride_k_ring_slot
                    + (i_h * L + o_c[:, None]) * K + o_kt[None, :])
        k_hist = tl.load(p_k_hist,
                         mask=cache_valid[:, None] & mask_kt[None, :],
                         other=0.0).to(q.dtype.element_ty)
        h = h0 * total_decay + tl.dot(d_tc, k_hist, input_precision=DOT_PRECISION).to(tl.float32)
        state_q += tl.sum(h * q_cur[None, :], axis=1)
        state_k += tl.sum(h * k_cur[None, :], axis=1)

        if write_k:
            p_k_new = (k_ring + state_idx * stride_k_ring_slot
                       + (i_h * L + write_pos) * K + o_kt)
            tl.store(p_k_new, k_cur.to(p_k_new.dtype.element_ty), mask=mask_kt)

    state_q *= alpha_cur
    state_k *= alpha_cur
    d_cur = beta_cur * (b_v - state_k)
    result = state_q + d_cur * tl.sum(cur_kq)
    tl.store(p_out, result.to(p_out.dtype.element_ty), mask=mask_v)

    if is_flush:
        for kk in range(NK):
            o_kt = kk * BKT + tl.arange(0, BKT)
            mask_kt = o_kt < K
            k_cur = tl.load(p_k + o_kt, mask=mask_kt, other=0.0).to(tl.float32) * k_rnorm
            p_h0 = (checkpoint + state_idx * stride_state_slot + i_hv * V * K
                    + o_v[:, None] * K + o_kt[None, :])
            h0 = tl.load(p_h0, mask=mask_v[:, None] & mask_kt[None, :], other=0.0).to(tl.float32)
            p_k_hist = (k_ring + state_idx * stride_k_ring_slot
                        + (i_h * L + o_c[:, None]) * K + o_kt[None, :])
            k_hist = tl.load(p_k_hist,
                             mask=cache_valid[:, None] & mask_kt[None, :],
                             other=0.0).to(q.dtype.element_ty)
            h = h0 * total_decay + tl.dot(d_tc, k_hist, input_precision=DOT_PRECISION).to(tl.float32)
            h = alpha_cur * h + d_cur[:, None] * k_cur[None, :]
            tl.store(p_h0, h.to(p_h0.dtype.element_ty),
                     mask=mask_v[:, None] & mask_kt[None, :])
    else:
        p_d_new = (d_ring + state_idx * stride_d_slot
                   + (i_hv * L + write_pos) * V + o_v)
        tl.store(p_d_new, d_cur.to(p_d_new.dtype.element_ty), mask=mask_v)
        if i_v == 0:
            tl.store(g_ring + state_idx * stride_g_ring_slot + i_hv * L + write_pos,
                     g_cur)


@triton.jit
def _advance_standard_cursor_kernel(cursor, state_indices, force_flush,
                                    stride_cursor_slot, stride_indices,
                                    n_rows, L: tl.constexpr,
                                    HAS_FORCE_FLUSH: tl.constexpr,
                                    BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = row < n_rows
    state_idx = tl.load(state_indices + row * stride_indices, mask=mask, other=-1).to(tl.int64)
    valid = mask & (state_idx >= 0)
    p = cursor + state_idx * stride_cursor_slot
    write_pos = tl.load(p, mask=valid, other=0).to(tl.int32)
    flush = write_pos == L - 1
    if HAS_FORCE_FLUSH:
        flush = flush | (tl.load(force_flush + row, mask=mask, other=0) != 0)
    tl.store(p, tl.where(flush, 0, write_pos + 1), mask=valid)
    tl.store(p + 1, 0, mask=valid)
    tl.store(p + 2, 0, mask=valid)


@triton.jit
def _fold_standard_history_kernel(checkpoint, d_ring, k_ring, g_ring, cursor,
                                  state_indices, stride_state_slot,
                                  stride_d_slot, stride_k_slot, stride_g_slot,
                                  stride_cursor_slot, stride_indices,
                                  H: tl.constexpr, HV: tl.constexpr,
                                  K: tl.constexpr, V: tl.constexpr,
                                  L: tl.constexpr, BK: tl.constexpr,
                                  BV: tl.constexpr, BC: tl.constexpr):
    """Materialize a partial standard ring before a continuation prefill."""
    i_v = tl.program_id(0)
    i_b = tl.program_id(1)
    i_hv = tl.program_id(2)
    i_h = i_hv // (HV // H)
    state_idx = tl.load(state_indices + i_b * stride_indices).to(tl.int64)
    if state_idx < 0:
        return
    write_pos = tl.load(cursor + state_idx * stride_cursor_slot).to(tl.int32)
    if write_pos == 0:
        return

    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, BK)
    o_c = tl.arange(0, BC)
    mask_v = o_v < V
    mask_k = o_k < K
    cache_valid = o_c < write_pos
    gates = tl.load(g_ring + state_idx * stride_g_slot + i_hv * L + o_c,
                    mask=cache_valid, other=0.0).to(tl.float32)
    prefix = tl.cumsum(gates, axis=0)
    total = tl.sum(gates, axis=0)
    decay = tl.where(cache_valid, tl.exp(total - prefix), 0.0)
    d_hist = tl.load(d_ring + state_idx * stride_d_slot
                     + (i_hv * L + o_c[None, :]) * V + o_v[:, None],
                     mask=mask_v[:, None] & cache_valid[None, :], other=0.0)
    k_hist = tl.load(k_ring + state_idx * stride_k_slot
                     + (i_h * L + o_c[:, None]) * K + o_k[None, :],
                     mask=cache_valid[:, None] & mask_k[None, :], other=0.0)
    p_state = (checkpoint + state_idx * stride_state_slot + i_hv * V * K
               + o_v[:, None] * K + o_k[None, :])
    state = tl.load(p_state, mask=mask_v[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
    state = state * tl.exp(total) + tl.dot(
        (d_hist * decay[None, :]).to(d_hist.dtype), k_hist,
        input_precision='ieee').to(tl.float32)
    tl.store(p_state, state, mask=mask_v[:, None] & mask_k[None, :])


@triton.jit
def _reset_cursor_kernel(cursor, slot_ids, stride_cursor_slot, n_slots,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = row < n_slots
    slot = tl.load(slot_ids + row, mask=mask, other=0).to(tl.int64)
    valid = mask & (slot >= 0)
    p = cursor + slot * stride_cursor_slot
    tl.store(p, 0, mask=valid)
    tl.store(p + 1, 0, mask=valid)
    tl.store(p + 2, 0, mask=valid)


def fused_recurrent_gdn_replayssm_decode(q: torch.Tensor,
                                         k: torch.Tensor,
                                         v: torch.Tensor,
                                         g: torch.Tensor,
                                         beta: torch.Tensor,
                                         checkpoint: torch.Tensor,
                                         d_ring: torch.Tensor,
                                         k_ring: torch.Tensor,
                                         g_ring: torch.Tensor,
                                         packed_cursor: torch.Tensor,
                                         state_indices: torch.Tensor,
                                         force_flush: torch.Tensor | None = None,
                                         scale: float | None = None,
                                         dot_precision: str = 'ieee'):
    """Run one-token standard ReplaySSM without advancing the shared cursor."""
    B, T, H, K = q.shape
    _, Tv, HV, V = v.shape
    if T != 1 or Tv != 1:
        raise ValueError('standard ReplaySSM decode requires exactly one token per row')
    if k.shape != q.shape or g.shape != (B, 1, HV) or beta.shape != (B, 1, HV):
        raise ValueError('invalid q/k/g/beta shape for standard ReplaySSM')
    if checkpoint.shape[1:] != (HV, V, K):
        raise ValueError('invalid ReplaySSM checkpoint shape')
    L = d_ring.shape[2]
    if d_ring.shape[1:] != (HV, L, V) or k_ring.shape[1:] != (H, L, K):
        raise ValueError('invalid ReplaySSM d/k ring shape')
    if g_ring.shape[1:] != (HV, L) or g_ring.dtype != torch.float32:
        raise ValueError('invalid ReplaySSM g ring shape or dtype')
    if packed_cursor.ndim != 2 or packed_cursor.shape[1] != 3 or packed_cursor.dtype != torch.int32:
        raise ValueError('packed_cursor must have shape [slots, 3] and dtype int32')
    if state_indices.shape != (B, ):
        raise ValueError('state_indices must have shape [B]')
    if force_flush is not None and force_flush.shape != (B, ):
        raise ValueError('force_flush must have shape [B]')
    if scale is None:
        scale = K**-0.5

    BK = triton.next_power_of_2(K)
    config = get_standard_replayssm_config(B)
    nk = min(config['NK'], BK)
    BKT = BK // nk
    BV = min(triton.next_power_of_2(V), config['BV'])
    BC = max(16, triton.next_power_of_2(L))
    out = torch.empty_like(v)
    grid = (triton.cdiv(V, BV), B, HV)
    _gdn_replayssm_decode_kernel[grid](
        q, k, v, g, beta, out, checkpoint, d_ring, k_ring, g_ring,
        packed_cursor, state_indices,
        force_flush if force_flush is not None else state_indices,
        q.stride(0), q.stride(2), k.stride(0), k.stride(2),
        v.stride(0), v.stride(2), g.stride(0), beta.stride(0),
        checkpoint.stride(0), d_ring.stride(0), k_ring.stride(0),
        g_ring.stride(0), packed_cursor.stride(0), state_indices.stride(0),
        scale,
        H=H, HV=HV, K=K, V=V, L=L, BK=BK, BV=BV, BC=BC,
        NK=nk, BKT=BKT, HAS_FORCE_FLUSH=force_flush is not None,
        DOT_PRECISION=dot_precision, num_warps=config['num_warps'],
        num_stages=config['num_stages'])
    return out, checkpoint


def advance_gdn_replayssm_standard_cursor(packed_cursor: torch.Tensor,
                                          state_indices: torch.Tensor,
                                          buffer_len: int,
                                          force_flush: torch.Tensor | None = None) -> None:
    """Advance valid standard-decode rows exactly once after all GDN layers."""
    n_rows = state_indices.numel()
    block = STANDARD_CURSOR_SM90_CONFIG['BLOCK']
    _advance_standard_cursor_kernel[(triton.cdiv(n_rows, block), )](
        packed_cursor, state_indices,
        force_flush if force_flush is not None else state_indices,
        packed_cursor.stride(0), state_indices.stride(0), n_rows,
        L=buffer_len, HAS_FORCE_FLUSH=force_flush is not None, BLOCK=block)


def fold_gdn_replayssm_standard_history(checkpoint: torch.Tensor,
                                        d_ring: torch.Tensor,
                                        k_ring: torch.Tensor,
                                        g_ring: torch.Tensor,
                                        packed_cursor: torch.Tensor,
                                        state_indices: torch.Tensor) -> None:
    """Materialize any committed partial ring without changing its cursor."""
    slots, hv, vdim, kdim = checkpoint.shape
    del slots
    h = k_ring.shape[1]
    length = d_ring.shape[2]
    bk = triton.next_power_of_2(kdim)
    config = get_standard_replayssm_config(state_indices.numel())
    bv = min(triton.next_power_of_2(vdim), config['BV'])
    bc = max(16, triton.next_power_of_2(length))
    grid = (triton.cdiv(vdim, bv), state_indices.numel(), hv)
    _fold_standard_history_kernel[grid](
        checkpoint, d_ring, k_ring, g_ring, packed_cursor, state_indices,
        checkpoint.stride(0), d_ring.stride(0), k_ring.stride(0),
        g_ring.stride(0), packed_cursor.stride(0), state_indices.stride(0),
        H=h, HV=hv, K=kdim, V=vdim, L=length, BK=bk, BV=bv, BC=bc,
        num_warps=config['num_warps'],
        num_stages=config['num_stages'])


def reset_gdn_replayssm_cursor(packed_cursor: torch.Tensor,
                               slot_ids: torch.Tensor) -> None:
    """Reset selected device-resident cursor rows."""
    n_slots = slot_ids.numel()
    block = RESET_CURSOR_SM90_CONFIG['BLOCK']
    _reset_cursor_kernel[(triton.cdiv(n_slots, block), )](
        packed_cursor, slot_ids, packed_cursor.stride(0), n_slots, BLOCK=block)

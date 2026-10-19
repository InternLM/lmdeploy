# Copyright (c) OpenMMLab. All rights reserved.
# Forward propagation adapted from fla-core 0.5.2, ops/common/chunk_delta_h.py.
# Only addressing and observational checkpoint stores change; preserve chunk math.
# MIT License
#
# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import triton
import triton.language as tl
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.utils import (
    IS_NVIDIA_BLACKWELL,
    autotune_cache_kwargs,
    check_shared_mem,
)

# TODO: Triton mainline fixes a Blackwell tl.dot recurrence race.
# Keep this kernel on num_warps=2 for Blackwell until Triton 3.8 is released
# and we re-validate the wider config space.
GATED_DELTA_RULE_FWD_H_NUM_WARPS = [2] if IS_NVIDIA_BLACKWELL else [2, 4]


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_INITIAL_STATE': lambda args: True,
    'STORE_FINAL_STATE': lambda args: True,
    'STORE_CHECKPOINT': lambda args: args['checkpoint_ids'] is not None,
    'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T', 'NUM_UNITS'])
def _indexed_chunk_state_kernel(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    pool,
    input_ids,
    final_ids,
    checkpoint_ids,
    checkpoint_offsets,
    stride_unit: tl.constexpr,
    stride_head: tl.constexpr,
    stride_k: tl.constexpr,
    stride_v: tl.constexpr,
    NUM_UNITS,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    STORE_CHECKPOINT: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ALLOW_INPUT_ALIAS: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    if STATE_V_FIRST:
        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * HV + i_h).to(tl.int64) * K*V
    v += (bos * HV + i_h).to(tl.int64) * V
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * HV + i_h).to(tl.int64) * V

    input_id = tl.load(input_ids + i_n).to(tl.int64)
    final_id = tl.load(final_ids + i_n).to(tl.int64)
    input_valid = (input_id >= 0) & (input_id < NUM_UNITS)
    # Each CTA owns a disjoint value tile and loads its input tile before
    # updating it. Only an exclusive final destination may reuse that input.
    final_valid = (input_valid & (final_id > 0) & (final_id < NUM_UNITS)
                   & ((final_id != input_id) | ALLOW_INPUT_ALIAS) & (T > 0))
    h0 = pool + input_id * stride_unit + i_h * stride_head
    ht = pool + final_id * stride_unit + i_h * stride_head
    if STORE_CHECKPOINT:
        checkpoint_id = tl.load(checkpoint_ids + i_n).to(tl.int64)
        checkpoint_offset = tl.load(checkpoint_offsets + i_n)
        # c=e with a shared final ID is written only by the final store.
        checkpoint_valid = (input_valid & (checkpoint_id > 0) & (checkpoint_id < NUM_UNITS)
                            & (checkpoint_id != input_id) & (checkpoint_id != final_id)
                            & (checkpoint_offset > 0) & (checkpoint_offset <= T))
        hc = pool + checkpoint_id * stride_unit + i_h * stride_head

    # load initial state
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_k3 = 128 + o_k1
    m_k3 = o_k3 < K
    o_k4 = 192 + o_k1
    m_k4 = o_k4 < K
    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            p_h0_1 = h0 + o_v[:, None] * stride_v + o_k1[None, :] * stride_k
            m_h0_1 = m_v[:, None] & m_k1[None, :]
        else:
            p_h0_1 = h0 + o_k1[:, None] * stride_k + o_v[None, :] * stride_v
            m_h0_1 = m_k1[:, None] & m_v[None, :]
        b_h1 += tl.load(p_h0_1, mask=m_h0_1 & input_valid, other=0.0).to(tl.float32)
        if K > 64:
            if STATE_V_FIRST:
                p_h0_2 = h0 + o_v[:, None] * stride_v + o_k2[None, :] * stride_k
                m_h0_2 = m_v[:, None] & m_k2[None, :]
            else:
                p_h0_2 = h0 + o_k2[:, None] * stride_k + o_v[None, :] * stride_v
                m_h0_2 = m_k2[:, None] & m_v[None, :]
            b_h2 += tl.load(p_h0_2, mask=m_h0_2 & input_valid, other=0.0).to(tl.float32)
        if K > 128:
            if STATE_V_FIRST:
                p_h0_3 = h0 + o_v[:, None] * stride_v + o_k3[None, :] * stride_k
                m_h0_3 = m_v[:, None] & m_k3[None, :]
            else:
                p_h0_3 = h0 + o_k3[:, None] * stride_k + o_v[None, :] * stride_v
                m_h0_3 = m_k3[:, None] & m_v[None, :]
            b_h3 += tl.load(p_h0_3, mask=m_h0_3 & input_valid, other=0.0).to(tl.float32)
        if K > 192:
            if STATE_V_FIRST:
                p_h0_4 = h0 + o_v[:, None] * stride_v + o_k4[None, :] * stride_k
                m_h0_4 = m_v[:, None] & m_k4[None, :]
            else:
                p_h0_4 = h0 + o_k4[:, None] * stride_k + o_v[None, :] * stride_v
                m_h0_4 = m_k4[:, None] & m_v[None, :]
            b_h4 += tl.load(p_h0_4, mask=m_h0_4 & input_valid, other=0.0).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        if STATE_V_FIRST:
            p_h1 = h + i_t_int64 * HV*K*V + o_v[:, None] * K + o_k1[None, :]
            m_h1 = m_v[:, None] & m_k1[None, :]
        else:
            p_h1 = h + i_t_int64 * HV*K*V + o_k1[:, None] * V + o_v[None, :]
            m_h1 = m_k1[:, None] & m_v[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_h1)
        if K > 64:
            if STATE_V_FIRST:
                p_h2 = h + i_t_int64 * HV*K*V + o_v[:, None] * K + o_k2[None, :]
                m_h2 = m_v[:, None] & m_k2[None, :]
            else:
                p_h2 = h + i_t_int64 * HV*K*V + o_k2[:, None] * V + o_v[None, :]
                m_h2 = m_k2[:, None] & m_v[None, :]
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_h2)
        if K > 128:
            if STATE_V_FIRST:
                p_h3 = h + i_t_int64 * HV*K*V + o_v[:, None] * K + o_k3[None, :]
                m_h3 = m_v[:, None] & m_k3[None, :]
            else:
                p_h3 = h + i_t_int64 * HV*K*V + o_k3[:, None] * V + o_v[None, :]
                m_h3 = m_k3[:, None] & m_v[None, :]
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_h3)
        if K > 192:
            if STATE_V_FIRST:
                p_h4 = h + i_t_int64 * HV*K*V + o_v[:, None] * K + o_k4[None, :]
                m_h4 = m_v[:, None] & m_k4[None, :]
            else:
                p_h4 = h + i_t_int64 * HV*K*V + o_k4[:, None] * V + o_v[None, :]
                m_h4 = m_k4[:, None] & m_v[None, :]
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_h4)

        p_w = w + o_t[:, None] * (HV*K) + o_k1[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        if STATE_V_FIRST:
            b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        else:
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = w + o_t[:, None] * (HV*K) + o_k2[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = w + o_t[:, None] * (HV*K) + o_k3[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = w + o_t[:, None] * (HV*K) + o_k4[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = v + o_t[:, None] * (HV*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0) - b_v

        if SAVE_NEW_VALUE:
            p_v = v_new + o_t[:, None] * (HV*V) + o_v[None, :]
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(tl.float32)
            p_g = g + (bos * HV + i_h).to(tl.int64) + o_t * HV
            b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
            b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
            b_g_last = exp2(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * HV*K + i_h * K + o_k1,
                mask=(o_k1 < K), other=0.).to(tl.float32)
            if STATE_V_FIRST:
                b_h1 *= exp2(b_gk_last1)[None, :]
            else:
                b_h1 *= exp2(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                gk + (bos + last_idx) * HV*K + i_h * K + o_k2,
                mask=(o_k2 < K), other=0.).to(tl.float32)
                if STATE_V_FIRST:
                    b_h2 *= exp2(b_gk_last2)[None, :]
                else:
                    b_h2 *= exp2(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                gk + (bos + last_idx) * HV*K + i_h * K + o_k3,
                mask=(o_k3 < K), other=0.).to(tl.float32)
                if STATE_V_FIRST:
                    b_h3 *= exp2(b_gk_last3)[None, :]
                else:
                    b_h3 *= exp2(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                gk + (bos + last_idx) * HV*K + i_h * K + o_k4,
                mask=(o_k4 < K), other=0.).to(tl.float32)
                if STATE_V_FIRST:
                    b_h4 *= exp2(b_gk_last4)[None, :]
                else:
                    b_h4 *= exp2(b_gk_last4)[:, None]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = k + o_k1[:, None] + o_t[None, :] * (H*K)
        b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        if STATE_V_FIRST:
            b_h1 += tl.trans(tl.dot(b_k, b_v))
        else:
            b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = k + o_k2[:, None] + o_t[None, :] * (H*K)
            b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            if STATE_V_FIRST:
                b_h2 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = k + o_k3[:, None] + o_t[None, :] * (H*K)
            b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
            if STATE_V_FIRST:
                b_h3 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = k + o_k4[:, None] + o_t[None, :] * (H*K)
            b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
            if STATE_V_FIRST:
                b_h4 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h4 += tl.dot(b_k, b_v)

        if STORE_CHECKPOINT:
            if checkpoint_valid and min((i_t + 1) * BT, T) == checkpoint_offset:
                if STATE_V_FIRST:
                    p_hc = hc + o_v[:, None] * stride_v + o_k1[None, :] * stride_k
                    m_ht = m_v[:, None] & m_k1[None, :]
                else:
                    p_hc = hc + o_k1[:, None] * stride_k + o_v[None, :] * stride_v
                    m_ht = m_k1[:, None] & m_v[None, :]
                tl.store(p_hc, b_h1.to(p_hc.dtype.element_ty), mask=m_ht & checkpoint_valid)
                if K > 64:
                    if STATE_V_FIRST:
                        p_hc = hc + o_v[:, None] * stride_v + o_k2[None, :] * stride_k
                        m_ht = m_v[:, None] & m_k2[None, :]
                    else:
                        p_hc = hc + o_k2[:, None] * stride_k + o_v[None, :] * stride_v
                        m_ht = m_k2[:, None] & m_v[None, :]
                    tl.store(p_hc, b_h2.to(p_hc.dtype.element_ty), mask=m_ht & checkpoint_valid)
                if K > 128:
                    if STATE_V_FIRST:
                        p_hc = hc + o_v[:, None] * stride_v + o_k3[None, :] * stride_k
                        m_ht = m_v[:, None] & m_k3[None, :]
                    else:
                        p_hc = hc + o_k3[:, None] * stride_k + o_v[None, :] * stride_v
                        m_ht = m_k3[:, None] & m_v[None, :]
                    tl.store(p_hc, b_h3.to(p_hc.dtype.element_ty), mask=m_ht & checkpoint_valid)
                if K > 192:
                    if STATE_V_FIRST:
                        p_hc = hc + o_v[:, None] * stride_v + o_k4[None, :] * stride_k
                        m_ht = m_v[:, None] & m_k4[None, :]
                    else:
                        p_hc = hc + o_k4[:, None] * stride_k + o_v[None, :] * stride_v
                        m_ht = m_k4[:, None] & m_v[None, :]
                    tl.store(p_hc, b_h4.to(p_hc.dtype.element_ty), mask=m_ht & checkpoint_valid)

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = ht + o_v[:, None] * stride_v + o_k1[None, :] * stride_k
            m_ht = m_v[:, None] & m_k1[None, :]
        else:
            p_ht = ht + o_k1[:, None] * stride_k + o_v[None, :] * stride_v
            m_ht = m_k1[:, None] & m_v[None, :]
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), mask=m_ht & final_valid)
        if K > 64:
            if STATE_V_FIRST:
                p_ht = ht + o_v[:, None] * stride_v + o_k2[None, :] * stride_k
                m_ht = m_v[:, None] & m_k2[None, :]
            else:
                p_ht = ht + o_k2[:, None] * stride_k + o_v[None, :] * stride_v
                m_ht = m_k2[:, None] & m_v[None, :]
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), mask=m_ht & final_valid)
        if K > 128:
            if STATE_V_FIRST:
                p_ht = ht + o_v[:, None] * stride_v + o_k3[None, :] * stride_k
                m_ht = m_v[:, None] & m_k3[None, :]
            else:
                p_ht = ht + o_k3[:, None] * stride_k + o_v[None, :] * stride_v
                m_ht = m_k3[:, None] & m_v[None, :]
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), mask=m_ht & final_valid)
        if K > 192:
            if STATE_V_FIRST:
                p_ht = ht + o_v[:, None] * stride_v + o_k4[None, :] * stride_k
                m_ht = m_v[:, None] & m_k4[None, :]
            else:
                p_ht = ht + o_k4[:, None] * stride_k + o_v[None, :] * stride_v
                m_ht = m_k4[:, None] & m_v[None, :]
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), mask=m_ht & final_valid)



# Ordinary disjoint inputs may be benchmarked repeatedly without changing their source.
_autotuned_indexed_chunk_state = fla_cache_autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in GATED_DELTA_RULE_FWD_H_NUM_WARPS
        for num_stages in ([2, 3, 4] if check_shared_mem('ampere') else [2, 1])
        for BV in ([32, 64] if check_shared_mem('ada') else [32])
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'STATE_V_FIRST'],
    **autotune_cache_kwargs,
)(_indexed_chunk_state_kernel)


def indexed_chunk_state_fwd(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    pool: torch.Tensor,
    input_ids: torch.Tensor,
    final_ids: torch.Tensor,
    *,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    checkpoint_ids: torch.Tensor | None = None,
    checkpoint_offsets: torch.Tensor | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    allow_input_alias: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FLA-0.5.2 chunk propagation with direct unit input/output addressing.

    ``k/w/u`` are the original intra-chunk transformed inputs; ``g/gk`` are
    log2 chunk-cumulative gates. No state gather, final-state staging or scatter.
    h/v_new are the mathematical workspace used by the chunk output algorithm.

    The planner excludes shared holders and other rows' inputs from private
    destinations. Only explicit exclusive reuse permits this row's own input
    as its final destination. Internal checkpoints must be relative multiples of64;
    final boundaries may be partial. Invalid/dummy IDs never write pool zero.
    Varlen callers supply prebuilt chunk metadata (no new D2H/synchronization).
    Explicit input aliasing is only for owner-proven private destinations.
    It bypasses all autotune trial launches; checkpoints remain disjoint and
    empty rows do not write. This primitive does not select compact serving mode.
    """
    batch, length, heads, dim = k.shape
    value_heads, value_dim = u.shape[2:]
    if not (0 < dim <= 256) or value_heads % heads:
        raise ValueError('Indexed chunk propagation requires K<=256 and integral value-head grouping.')
    expected = (value_heads, value_dim, dim) if state_v_first else (value_heads, dim, value_dim)
    if pool.ndim != 4 or tuple(pool.shape[1:]) != expected:
        raise ValueError(f'Expected a rank-4 single-boundary state pool with payload {expected}.')
    if pool.dtype not in (torch.float16, torch.bfloat16, torch.float32) or pool.device != k.device:
        raise ValueError('State pool must use FP16/BF16/FP32 on the compute device.')
    if not all(x.is_contiguous() for x in (k, w, u)):
        raise ValueError('Chunk math inputs must be contiguous; state pool may be strided.')
    if tuple(w.shape) != (batch, length, value_heads, dim) or u.shape[:2] != k.shape[:2]:
        raise ValueError('Inconsistent chunk input geometry.')
    count = batch if cu_seqlens is None else cu_seqlens.numel() - 1
    if cu_seqlens is not None:
        if batch != 1 or chunk_indices is None or chunk_offsets is None:
            raise ValueError('Packed varlen inputs require prebuilt chunk indices and offsets.')
        if chunk_offsets.shape != (count + 1,) or chunk_indices.ndim != 2 or chunk_indices.shape[1] != 2:
            raise ValueError('Invalid chunk metadata shapes.')
        for metadata in (cu_seqlens, chunk_indices, chunk_offsets):
            if metadata.device != k.device or not metadata.is_contiguous() or metadata.dtype not in (
                    torch.int32, torch.int64):
                raise ValueError('Chunk metadata must be contiguous device integers.')
    if (checkpoint_ids is None) != (checkpoint_offsets is None):
        raise ValueError('Checkpoint IDs and relative offsets must be supplied together.')
    for ids in (input_ids, final_ids, checkpoint_ids, checkpoint_offsets):
        if ids is not None and (ids.shape != (count,) or ids.device != k.device
                                or not ids.is_contiguous() or ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError('State IDs and offsets must be contiguous per-request device integers.')
    for gate, shape in ((g, (batch, length, value_heads)), (gk, (batch, length, value_heads, dim))):
        if gate is not None and (gate.shape != shape or gate.device != k.device or not gate.is_contiguous()):
            raise ValueError('Invalid cumulative gate layout.')
    chunks = triton.cdiv(length, 64) if cu_seqlens is None else len(chunk_indices)
    h = k.new_empty((batch, chunks, *expected))
    v_new = torch.empty_like(u)
    if count == 0:
        return h, v_new
    stride_k, stride_v = pool.stride()[-2:]
    if state_v_first:
        stride_k, stride_v = stride_v, stride_k
    def grid(meta):
        return (triton.cdiv(value_dim, meta['BV']), count * value_heads)

    # Online autotuning would repeatedly advance an aliased persistent input.
    # This explicit single launch uses the conservative supported configuration;
    # never clone/restore the pool to make benchmarking appear side-effect free.
    kernel = _indexed_chunk_state_kernel if allow_input_alias else _autotuned_indexed_chunk_state
    launch = dict(BV=32, num_warps=2, num_stages=2) if allow_input_alias else {}
    kernel[grid](
        k, u, w, v_new, g, gk, h, pool, input_ids, final_ids, checkpoint_ids, checkpoint_offsets,
        pool.stride(0), pool.stride(1), stride_k, stride_v, pool.shape[0], cu_seqlens, chunk_offsets,
        T=length, H=heads, HV=value_heads, K=dim, V=value_dim, BT=64, STATE_V_FIRST=state_v_first,
        ALLOW_INPUT_ALIAS=allow_input_alias, **launch)
    return h, v_new

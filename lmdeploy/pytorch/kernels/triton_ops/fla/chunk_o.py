# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# ruff: noqa: E501
# mypy: ignore-errors
from typing import Optional

import torch
import triton
import triton.language as tl

from .utils import prepare_chunk_offsets, safe_exp


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    T_max = T

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(
            tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # offset calculation
    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V

    for i_t in range(NT):
        i_tg = boh + i_t
        h_base = h + (i_tg * H + i_h).to(tl.int64) * K * V
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_A = tl.zeros([BT, BT], dtype=tl.float32)

        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(q, (T, K), (Hg * K, 1),
                                    (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k, (K, T), (1, Hg * K),
                                    (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_h = tl.make_block_ptr(h_base, (K, V), (V, 1),
                                    (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            # [BT, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            # [BK, BT]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BK, BV]
            b_h = tl.load(p_h, boundary_check=(0, 1))

            # [BT, BK] @ [BK, BV] -> [BT, BV]
            b_o += tl.dot(b_q, b_h)
            # [BT, BK] @ [BK, BT] -> [BT, BT]
            b_A += tl.dot(b_q, b_k)

        if USE_G:
            offs_t = i_t * BT + tl.arange(0, BT)
            mask_t = offs_t < T
            g_ptr = g + bos + i_h * T_max
            b_g = tl.load(g_ptr + offs_t, mask=mask_t, other=0.0)

            b_o = b_o * tl.exp(b_g)[:, None]
            b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])

        o_i = tl.arange(0, BT).to(tl.float32)
        m_A = o_i[:, None] >= o_i[None, :]
        b_A = tl.where(m_A, b_A, 0)

        p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV),
                                (BT, BV), (1, 0))
        p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV),
                                (BT, BV), (1, 0))

        b_v = tl.load(p_v, boundary_check=(0, 1))
        # to fix mma -> mma layout conversion
        # already solved by fla v3.2 or higher
        b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    BT = chunk_size

    if scale is None:
        scale = k.shape[-1]**-0.5

    o = torch.empty_like(v)
    if cu_seqlens is None:
        N, chunk_offsets = B, None
    else:
        N, chunk_offsets = (
            len(cu_seqlens) - 1,
            prepare_chunk_offsets(cu_seqlens, BT),
        )

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), N * H)

    g = g.transpose(1, 2).contiguous()
    chunk_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=128,
        BV=128,
        num_warps=4,
        num_stages=2,
    )
    return o

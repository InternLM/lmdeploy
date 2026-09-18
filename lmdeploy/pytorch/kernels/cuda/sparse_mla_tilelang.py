# Copyright (c) OpenMMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TileLang sparse MLA decode attention for CUDA BF16 tensors.

The kernel schedule in this file is adapted from SGLang's
sparse_attention_fwd_kernel_v1 at commit 9e692c9216c3, distributed under the
Apache License 2.0:

https://github.com/sgl-project/sglang/blob/9e692c9216c3/python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py

Only the CUDA BF16 path used by GLM-5.3-Flash is retained. The public wrapper
uses generic sparse-MLA names and has no SGLang runtime dependency.
"""

import tilelang
import tilelang.language as T
import torch

tilelang.set_log_level('WARNING')


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def _sparse_mla_bf16_fwd_kernel(
    num_heads,
    dim,
    topk,
    *,
    storage_dim=None,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=64,
    num_stages=2,
    threads=256,
):
    if storage_dim is None:
        storage_dim = dim
    assert storage_dim >= dim
    assert (
        dim == tilelang.math.next_power_of_2(dim) or dim % 64 == 0
    ), f"dim={dim} must be a power of 2 or a multiple of 64"
    assert is_causal, "non-causal is not supported"
    assert (
        topk % block_I == 0
    ), "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    seq_len_kv = T.symbolic("seq_len_kv")

    head_kv = num_heads // kv_group
    q_shape = [batch, seq_len, num_heads, dim]
    kv_shape = [batch, seq_len_kv, kv_group, storage_dim]
    o_shape = [batch, seq_len, num_heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"

    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (
            bx,
            by,
            bz,
        ):
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            O_shared = T.alloc_shared([H_per_block, D], dtype)
            mask = T.alloc_fragment([BI], "bool")

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)

            for i_i in T.Pipelined(NI, num_stages=num_stages):

                for bi_i in T.Parallel(BI):
                    mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] >= 0

                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[
                        b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i
                    ]

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        mask[bi_i], 0, -T.infinity(acc_s.dtype)
                    )
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(
                        acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

            # Rescale
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

            T.copy(acc_o, O_shared)
            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])

    return main


def sparse_mla_bf16_fwd(q: torch.Tensor,
                        kv: torch.Tensor,
                        indices: torch.Tensor,
                        sm_scale: float) -> torch.Tensor:
    """Run sparse MLA decode attention on CUDA BF16 tensors.

    Args:
        q: Query tensor with shape [tokens, local_heads, 512].
        kv: Paged latent KV tensor with shape [slots, 1, storage_dim]. The
            supported storage widths are 512 and 576; only the first 512
            elements participate in attention.
        indices: Physical slot indices with shape [tokens, 1, topk].
            Invalid entries must be -1 and topk must be padded to a
            multiple of 64.
        sm_scale: Attention scale before the kernel's base-2 conversion.

    Returns:
        A BF16 tensor with shape [tokens, local_heads, 512].
    """
    if torch.version.hip is not None:
        raise RuntimeError('sparse_mla_bf16_fwd only supports CUDA')
    if not (q.is_cuda and kv.is_cuda and indices.is_cuda):
        raise ValueError('q, kv, and indices must be CUDA tensors')
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise TypeError('q and kv must have dtype torch.bfloat16')
    if indices.dtype != torch.int32:
        raise TypeError('indices must have dtype torch.int32')
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] != 512:
        raise ValueError(
            'q must have shape [tokens, positive_local_heads, 512], '
            f'got {tuple(q.shape)}')
    if kv.ndim != 3 or kv.shape[1] != 1 or kv.shape[2] not in (512, 576):
        raise ValueError(
            'kv must have shape [slots, 1, storage_dim] with storage_dim '
            f'in (512, 576), got {tuple(kv.shape)}')
    if (indices.ndim != 3 or indices.shape[0] != q.shape[0]
            or indices.shape[1] != 1):
        raise ValueError(
            'indices must have shape [tokens, 1, topk] matching q, '
            f'got {tuple(indices.shape)}')
    if q.device != kv.device or q.device != indices.device:
        raise ValueError('q, kv, and indices must be on the same CUDA device')

    topk = indices.shape[-1]
    if topk == 0 or topk % 64 != 0:
        raise ValueError(f'topk must be a positive multiple of 64, got {topk}')

    if not kv.is_contiguous():
        raise ValueError(
            'kv must be a zero-copy contiguous view of the complete paged cache')
    q = q.contiguous()
    indices = indices.contiguous()
    kernel = _sparse_mla_bf16_fwd_kernel(
        num_heads=q.shape[1],
        dim=512,
        topk=topk,
        storage_dim=kv.shape[-1],
        sm_scale=sm_scale,
    )
    output = kernel(q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0))
    return output.squeeze(0)

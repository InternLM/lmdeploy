# Copyright (c) OpenMMLab. All rights reserved.
# ruff: noqa
"""LMDeploy adapter for XTuner's TileLang SparseMLA forward kernel.

XTuner source (pinned):
https://github.com/InternLM/xtuner/blob/6751212f032f941e72580b2afbb0c1440671cf7f/xtuner/v1/ops/sparse_mla/tilelang_sparse_mla_fwd.py

TileLang official example:
https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/sparse_mla_fwd.py
"""

import torch
import tilelang
from tilelang import language as T


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mla_fwd(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=64,
    num_stages=2,
    threads=256,
):
    assert dim == tilelang.math.next_power_of_2(dim), 'dim must be a power of two'
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), (
        'tail_dim must be a power of two'
    )
    assert is_causal, 'non-causal attention is not supported'
    assert topk % block_I == 0, 'top-k must be divisible by block_I'
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    batch = T.dynamic('batch')
    seq_len = T.dynamic('seq_len')
    seq_len_kv = T.dynamic('seq_len_kv')

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1, 'automatic head padding requires one KV group'
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > 64:
        assert head_kv % 64 == 0, 'head_kv should be a multiple of 64'
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
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (
            bx,
            by,
            bz,
        ):
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            mask = T.alloc_fragment([BI], 'bool')

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
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                for bi_i in T.Parallel(BI):
                    # Invalid DSA slots are padded with -1.
                    mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] != -1

                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i]
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    K_tail_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, D + d_i]

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.gemm(
                    Q_tail_shared,
                    K_tail_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Rescale
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
            T.copy(sumexp, Lse[b_i, s_i, H0:H1])

    return main

_HEAD_DIM = 576
_VALUE_DIM = 512
_INDEX_BLOCK = 64


def _validate_common(q: torch.Tensor, kv: torch.Tensor,
                     indices: torch.Tensor) -> None:
    if not q.is_cuda or not kv.is_cuda or not indices.is_cuda:
        raise RuntimeError('TileLang SparseMLA requires CUDA tensors.')
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise RuntimeError(
            'TileLang SparseMLA requires BF16 Q and KV tensors.')
    if q.size(-1) != _HEAD_DIM or kv.size(-1) != _HEAD_DIM:
        raise RuntimeError('TileLang SparseMLA supports head_dim=576 only.')
    if indices.size(-1) % _INDEX_BLOCK:
        raise RuntimeError(
            'TileLang SparseMLA requires top-k divisible by 64.')


def _launch(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor,
            sm_scale: float) -> torch.Tensor:
    """Launch the vendored XTuner kernel on its native batched layout."""
    _validate_common(q, kv, indices)
    q = q.contiguous()
    kv = kv.contiguous()
    indices = indices.to(torch.int32).contiguous()

    batch, seq_len, heads, head_dim = q.shape
    if kv.dim() != 4 or indices.dim() != 4:
        raise RuntimeError(
            'TileLang SparseMLA expects batched 4D KV and indices.')
    if kv.size(0) != batch or indices.shape[:2] != (batch, seq_len):
        raise RuntimeError(
            'TileLang SparseMLA batch and sequence dimensions must match.')
    kv_group = kv.size(2)
    if kv_group != 1 or indices.size(2) != kv_group:
        raise RuntimeError('TileLang SparseMLA supports one KV group only.')

    kernel = sparse_mla_fwd(
        heads=heads,
        dim=_VALUE_DIM,
        tail_dim=head_dim - _VALUE_DIM,
        topk=indices.size(-1),
        kv_group=kv_group,
        sm_scale=sm_scale,
        is_causal=True,
        block_I=_INDEX_BLOCK,
        num_stages=2,
        threads=256,
    )
    output, _ = kernel(q, kv, indices)
    return output


def tilelang_sparse_mla_forward(q: torch.Tensor, kv: torch.Tensor,
                                indices: torch.Tensor,
                                sm_scale: float) -> torch.Tensor:
    """Run SparseMLA over contiguous query rows and index-addressable KV."""
    if q.dim() != 3 or kv.dim() != 3 or indices.dim() != 3:
        raise RuntimeError(
            'Packed TileLang SparseMLA expects Q, KV and indices to be 3D.')
    return _launch(q[None], kv[None], indices[None], sm_scale)[0]

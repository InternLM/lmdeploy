# Copyright (c) OpenMMLab. All rights reserved.
"""Materialize KDA's channelwise log decay and beta for ReplaySSM rings."""
import torch
import triton
import triton.language as tl
from triton.language.extra import cuda as tl_cuda


@triton.jit
def _prepare_kda_replay_gates(raw_g, raw_b, a_log, dt_bias, g, beta,
                             N: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                             BH: tl.constexpr, BK: tl.constexpr,
                             LOWER_BOUND: tl.constexpr):
    # Group adjacent token/head rows to amortize the small per-head CTA.
    row = tl.program_id(0) * BH + tl.arange(0, BH)
    head = row % H
    k = tl.arange(0, BK)
    offsets = row[:, None] * K + k[None, :]
    mask = (row[:, None] < N) & (k[None, :] < K)
    value = tl.load(raw_g + offsets, mask=mask, other=0.0).to(tl.float32)
    value += tl.load(dt_bias + head[:, None] * K + k[None, :], mask=k[None, :] < K, other=0.0)
    gate = LOWER_BOUND / (1.0 + tl.exp(-tl.exp(tl.load(a_log + head))[:, None] * value))
    tl.store(g + offsets, gate, mask=mask)
    b = tl.load(raw_b + row, mask=row < N, other=0.0).to(tl.float32)
    # Match the existing fused KDA path's PyTorch sigmoid rounding.
    tl.store(beta + row, tl.div_rn(1.0, 1.0 + tl_cuda.libdevice.exp(-b)), mask=row < N)


def prepare_kda_replay_gates(raw_g, raw_beta, a_log, dt_bias, lower_bound):
    """Inputs are contiguous [B,T,H,K]/[B,T,H]; outputs remain FP32."""
    if lower_bound is None or lower_bound >= 0:
        raise ValueError('KDA ReplaySSM requires a negative bounded gate.')
    b, t, h, k = raw_g.shape
    g = torch.empty_like(raw_g, dtype=torch.float32)
    beta = torch.empty_like(raw_beta, dtype=torch.float32)
    _prepare_kda_replay_gates[(triton.cdiv(b * t * h, 4),)](
        raw_g, raw_beta, a_log, dt_bias, g, beta,
        N=b * t * h, H=h, K=k, BH=4, BK=triton.next_power_of_2(k), LOWER_BOUND=lower_bound,
        num_warps=4)
    return g, beta

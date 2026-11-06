# Copyright (c) OpenMMLab. All rights reserved.
"""Small-M BF16 projections using a local Triton SIMT reduction kernel."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class SkinnyGemmConfig:
    block_n: int
    block_k: int
    num_warps: int = 4
    k_unroll: int = 1


def warmup_skinny_gemm(n: int, k: int, m: int, config: SkinnyGemmConfig) -> None:
    """Compile a schedule without allocating another projection matrix."""
    dummy = torch.empty(0, dtype=torch.bfloat16, device='cuda')
    block_m = triton.next_power_of_2(m)
    _skinny_gemm_kernel.warmup(
        dummy,
        dummy,
        dummy,
        k,
        n,
        k,
        m,
        block_m,
        config.block_n,
        config.block_k,
        config.k_unroll,
        grid=(1, ),
        num_warps=config.num_warps,
        num_stages=1,
    )


@triton.jit
def _skinny_gemm_kernel(A, W, Y, stride_am, N: tl.constexpr, K: tl.constexpr, M: tl.constexpr, BLOCK_M: tl.constexpr,
                        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, K_UNROLL: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = tl.arange(0, BLOCK_K)
    accum = tl.full((BLOCK_M, BLOCK_N, BLOCK_K), 0, tl.float32)
    for tile in tl.range(0, tl.cdiv(K, BLOCK_K), loop_unroll_factor=K_UNROLL):
        kk = tile * BLOCK_K + offsets
        a = tl.load(A + rows[:, None] * stride_am + kk[None, :], (rows[:, None] < M) & (kk[None, :] < K),
                    other=0).to(tl.float32)
        w = tl.load(W + cols[:, None] * K + kk[None, :], (cols[:, None] < N) & (kk[None, :] < K),
                    other=0,
                    cache_modifier='.cg').to(tl.float32)
        accum = tl.fma(a[:, None, :], w[None, :, :], accum)
    result = tl.sum(accum, axis=2)
    tl.store(Y + rows[:, None] * N + cols[None, :], result, (rows[:, None] < M) & (cols[None, :] < N))


def skinny_gemm(x: torch.Tensor, weight: torch.Tensor, config: SkinnyGemmConfig) -> torch.Tensor:
    """Compute ``x @ weight.T`` for two-dimensional BF16 CUDA inputs.

    Each CTA owns its complete K reduction. There are no global atomics or extra reduction launches. A wider activation
    row stride is supported.
    """
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError('Expected [M,K] input and [N,K] weight.')
    if x.dtype != torch.bfloat16 or weight.dtype != x.dtype:
        raise ValueError('Skinny GEMM requires BF16 inputs and weights.')
    if not x.is_cuda or x.device != weight.device or not weight.is_contiguous() or x.stride(1) != 1:
        raise ValueError('Expected CUDA tensors on one device and contiguous weight rows.')
    m, k = x.shape
    n = weight.shape[0]
    if not 1 <= m <= 16:
        raise ValueError('Skinny GEMM supports 1..16 token rows.')
    output = x.new_empty((m, n))
    block_m = triton.next_power_of_2(m)
    _skinny_gemm_kernel[(triton.cdiv(n, config.block_n), )](
        x,
        weight,
        output,
        x.stride(0),
        n,
        k,
        m,
        block_m,
        config.block_n,
        config.block_k,
        config.k_unroll,
        num_warps=config.num_warps,
        num_stages=1,
    )
    return output

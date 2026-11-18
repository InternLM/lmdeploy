# Copyright (c) OpenMMLab. All rights reserved.
# Sinkhorn equations follow hc_split_sinkhorn.py, ported from DeepSeek-V4.
# Original copyright (c) DeepSeek
import functools

import tilelang
import tilelang.language as T
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .hc_split_sinkhorn import pass_configs


@triton.jit
def _hc_project_partial_kernel(X, W, P):
    row, split = tl.program_id(0), tl.program_id(1)
    n = tl.arange(0, 32)
    value = tl.full((32,), 0, tl.float32)
    # Preserve the 128 ordered FMAs in each original FP32 split-K partition.
    for j in range(128):
        k = split * 128 + j
        x = tl.load(X + row * 16384 + k)
        w = tl.load(W + k * 24 + n, n < 24, 0)
        value = tl.fma(x, w, value)
    tl.store(P + (row * 128 + split) * 32 + n, value)


@triton.jit
def _hc_project_reduce_kernel(P, Y):
    row = tl.program_id(0)
    n = tl.arange(0, 32)
    value = tl.full((32,), 0, tl.float32)
    # A parallel tree changes the projection; retain the serial partition sum.
    for j in range(128):
        value += tl.load(P + (row * 128 + j) * 32 + n)
    tl.store(Y + row * 24 + n, value, n < 24)


def hc_project_mixes(x_fp32: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    rows = x_fp32.numel() // 16384
    partials = x_fp32.new_empty((rows, 128, 32))
    out = x_fp32.new_empty((*x_fp32.shape[:-1], 24))
    _hc_project_partial_kernel[(rows, 128)](x_fp32, weight, partials, num_warps=1)
    _hc_project_reduce_kernel[(rows,)](partials, out, num_warps=1, enable_fp_fusion=False)
    return out


@functools.lru_cache
def _get_scale_mixes_kernel():
    # Keep the TileLang and ordered projection paths importable on Triton
    # versions without the experimental Gluon API. The backend gates this path.
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    @gluon.jit
    def _hc_scale_mixes_kernel(A, X, Y, EPS: gl.constexpr):
        row = gl.program_id(0)
        layout: gl.constexpr = gl.BlockedLayout([1], [32], [16], [0])
        r = gl.arange(0, 2048, layout=layout)
        acc0 = gl.full((2048,), 0, gl.float32, layout=layout)
        acc1 = gl.full((2048,), 0, gl.float32, layout=layout)
        # Keep the two 8192-wide partial sums and their reduction order. Merging
        # their launches must not merge the FP32 accumulation trees.
        for offset in range(0, 8192, 2048):
            x0 = gl.load(X + row * 16384 + offset + r)
            x1 = gl.load(X + row * 16384 + 8192 + offset + r)
            acc0 += x0 * x0
            acc1 += x1 * x1
        mean = (gl.sum(acc0, 0) + gl.sum(acc1, 0)) * (1.0 / 16384)
        stabilized = gl.inline_asm_elementwise(
            'add.rn.f32 $0, $1, $2;', constraints='=f,f,f', args=[mean, EPS],
            dtype=gl.float32, is_pure=True, pack=1)
        factor = libdevice.rsqrt(stabilized)
        n = gl.arange(0, 32, layout=layout)
        a = gl.load(A + row * 24 + n, n < 24, 0)
        gl.store(Y + row * 24 + n, a * factor, n < 24)

    return _hc_scale_mixes_kernel


def hc_scale_mixes(mixes: torch.Tensor, x_fp32: torch.Tensor, eps: float) -> torch.Tensor:
    """Fuse input RMS statistics and projection scaling for small GLM
    batches."""
    out = torch.empty_like(mixes)
    _get_scale_mixes_kernel()[(mixes.numel() // 24,)](
        mixes, x_fp32, out, eps, num_warps=16, enable_fp_fusion=False)
    return out


@tilelang.jit(pass_configs=pass_configs)
def _hc_pre_fused_kernel(dim: int, hc: int, sinkhorn_iters: int, eps: float, norm_eps: float):
    n = T.symbolic('n')
    channels = (2 + hc) * hc

    @T.prim_func
    def hc_pre_fused_kernel(
        x: T.Tensor[(n, hc, dim), T.bfloat16],
        mixes: T.Tensor[(n, channels), T.float32],
        hc_scale: T.Tensor[(3,), T.float32],
        hc_base: T.Tensor[(channels,), T.float32],
        norm_weight: T.Tensor[(dim,), T.bfloat16],
        out: T.Tensor[(n, dim), T.bfloat16],
        post: T.Tensor[(n, hc), T.float32],
        comb: T.Tensor[(n, hc, hc), T.float32],
    ):
        with T.Kernel(n, threads=96) as i:
            mixes_shared = T.alloc_shared(channels, 'float32')
            T.copy(mixes[i, :], mixes_shared)
            # The residual collapse depends on pre, not on the Sinkhorn result.
            # Run those independent branches on separate warp groups.
            if T.get_thread_binding() >= 64:
                cm = T.alloc_fragment((hc, hc), 'float32')
                for j in T.Parallel(hc):
                    post[i, j] = 2 * T.sigmoid(mixes_shared[j + hc] * hc_scale[1] + hc_base[j + hc])
                for j, k in T.Parallel(hc, hc):
                    cm[j, k] = mixes_shared[j * hc + k + hc * 2] * hc_scale[2] + hc_base[j * hc + k + hc * 2]
                row_max = T.alloc_fragment(hc, 'float32')
                row_sum = T.alloc_fragment(hc, 'float32')
                col_sum = T.alloc_fragment(hc, 'float32')
                T.reduce_max(cm, row_max, dim=1)
                for j, k in T.Parallel(hc, hc):
                    cm[j, k] = T.exp(cm[j, k] - row_max[j])
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc, hc):
                    cm[j, k] = cm[j, k] / row_sum[j] + eps
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc, hc):
                    cm[j, k] = cm[j, k] / (col_sum[k] + eps)
                for _ in T.serial(sinkhorn_iters - 1):
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(hc, hc):
                        cm[j, k] = cm[j, k] / (row_sum[j] + eps)
                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(hc, hc):
                        cm[j, k] = cm[j, k] / (col_sum[k] + eps)
                T.copy(cm, comb[i, :, :])

            else:
                pre_shared = T.alloc_shared(hc, 'float32')
                for j in T.Parallel(hc):
                    pre_shared[j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps
                reduced = T.alloc_fragment(dim, 'float32')
                rounded = T.alloc_fragment(dim, 'bfloat16')
                squares = T.alloc_fragment((2, dim // 2), 'float32')
                warp_sums = T.alloc_fragment(2, 'float32')
                T.annotate_layout({
                    reduced: T.Fragment((dim,), forward_thread_fn=lambda d: (d // 8) % 64,
                                        forward_index_fn=lambda d: (d // 512) * 8 + d % 8),
                    rounded: T.Fragment((dim,), forward_thread_fn=lambda d: (d // 8) % 64,
                                        forward_index_fn=lambda d: (d // 512) * 8 + d % 8),
                    squares: T.Fragment((2, dim // 2),
                                        forward_thread_fn=lambda w, d: w * 32 + (d // 8) % 32,
                                        forward_index_fn=lambda w, d: (d // 256) * 8 + d % 8),
                })
                sumsq = T.alloc_fragment(1, 'float32')
                T.clear(reduced)
                for h in T.serial(hc):
                    for d in T.Parallel(dim):
                        reduced[d] += pre_shared[h] * T.float32(x[i, h, d])
                # Match the BF16 intermediate consumed by standalone RMSNorm.
                T.copy(reduced, rounded)
                # TileLang reduces local elements in segment-first order.
                # Transpose the register indices so accumulation matches the
                # contiguous-element-first order of the original Triton kernel.
                for w, d in T.Parallel(2, dim // 2):
                    src = (d % 8) * 512 + w * 256 + ((d // 8) % 32) * 8 + d // 256
                    squares[w, d] = T.float32(rounded[src]) * T.float32(rounded[src])
                # Triton first reduces within each warp, then combines warps.
                T.reduce_sum(squares, warp_sums, dim=1)
                T.reduce_sum(warp_sums, sumsq, dim=0)
                for d in T.Parallel(dim):
                    out[i, d] = (T.float32(rounded[d]) * T.rsqrt(sumsq[0] * (1.0 / dim) + norm_eps)
                                 * T.float32(norm_weight[d]))

    return hc_pre_fused_kernel


def hc_pre_fused(x: torch.Tensor, mixes: torch.Tensor, hc_scale: torch.Tensor,
                 hc_base: torch.Tensor, norm_weight: torch.Tensor,
                 sinkhorn_iters: int, eps: float, norm_eps: float):
    """Fuse Sinkhorn, residual collapse and RMSNorm for small BF16 batches."""
    hc, dim = x.shape[-2:]
    out = x.new_empty((*x.shape[:-2], dim))
    post = mixes.new_empty((*x.shape[:-2], hc))
    comb = mixes.new_empty((*x.shape[:-2], hc, hc))
    _hc_pre_fused_kernel(dim, hc, sinkhorn_iters, eps, norm_eps)(
        x.view(-1, hc, dim), mixes.view(-1, (2 + hc) * hc), hc_scale, hc_base,
        norm_weight, out.view(-1, dim), post.view(-1, hc), comb.view(-1, hc, hc))
    return out, post, comb

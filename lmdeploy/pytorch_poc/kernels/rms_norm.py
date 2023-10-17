# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _sum_fn(x0, x1):
    return x0 + x1


@triton.jit
def rms_norm_kernel(X, WEIGHT, OUT, eps, BLOCK_N: tl.constexpr):
    prog_id = tl.program_id(0)

    w = tl.load(WEIGHT + tl.arange(0, BLOCK_N))
    x_off = prog_id * BLOCK_N + tl.arange(0, BLOCK_N)
    x = tl.load(X + x_off)
    xf = x.to(tl.float32)

    var = tl.reduce(xf * xf, 0, _sum_fn) * float(1 / BLOCK_N)
    sqrt = tl.sqrt(var + eps)
    out = xf / sqrt
    out = w * out.to(x.dtype)

    tl.store(OUT + x_off, out.to(x.dtype))


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6):
    """rms norm."""

    feat_size = weight.size(-1)
    seq_len = x.flatten(0, -2).size(0)

    BLOCK_N = feat_size

    out = torch.empty_like(x)

    grid = [
        seq_len,
    ]
    rms_norm_kernel[grid](x,
                          weight,
                          out,
                          eps,
                          BLOCK_N,
                          num_warps=4,
                          num_stages=2)

    return out

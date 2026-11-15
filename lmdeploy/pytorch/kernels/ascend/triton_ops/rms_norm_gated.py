# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Tri Dao.
#
# RMSNorm with gated SiLU activation (Triton kernel for Ascend NPU).
# Adapted from https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/fla/layernorm_guard.py
# Original source: https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py
# mypy: ignore-errors


import torch
import torch.nn as nn
import triton
import triton.language as tl

MAX_CORES = 65535


@triton.heuristics(
    {
        'HAS_BIAS': lambda args: args['B'] is not None,
        'HAS_Z': lambda args: args['Z'] is not None,
    }
)
@triton.jit
def _rms_norm_fwd_kernel(
    X,
    Y,
    W,
    B,
    Z,
    Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    N_CORES: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)

    BLOCK_ROWS = M if M < N_CORES else N_CORES
    n_iters = M // BLOCK_ROWS
    remain = M % BLOCK_ROWS
    if row < remain:
        n_iters = n_iters + 1

    for i in tl.range(n_iters):
        X_base = X + (i * BLOCK_ROWS * stride_x_row) + row * stride_x_row + group * N
        Y_base = Y + (i * BLOCK_ROWS * stride_y_row) + row * stride_y_row + group * N
        if HAS_Z:
            Z_base = Z + (i * BLOCK_ROWS * stride_z_row) + row * stride_z_row + group * N
        Rstd_base = Rstd + (i * BLOCK_ROWS) + group * M
        W_base = W + group * N
        if HAS_BIAS:
            B_base = B + group * N

        cols = tl.arange(0, BLOCK_N)
        x = tl.load(X_base + cols, mask=cols < N, other=0.0).to(tl.float32)
        if HAS_Z and not NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=cols < N).to(tl.float32)
            x *= z * tl.sigmoid(z)

        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        tl.store(Rstd_base + row, rstd)

        mask = cols < N
        w = tl.load(W_base + cols, mask=mask).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(B_base + cols, mask=mask).to(tl.float32)
        y = x * rstd * w + b if HAS_BIAS else x * rstd * w
        if HAS_Z and NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            y *= z * tl.sigmoid(z)
        tl.store(Y_base + cols, y, mask=mask)


def rmsnorm_fn(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True):
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    if z is not None:
        assert z.shape == x_shape_og
        z = z.reshape(-1, z.shape[-1])
        if z.stride(-1) != 1:
            z = z.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size

    out = torch.empty_like(x)
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    grid = (M if M < MAX_CORES else MAX_CORES, ngroups)

    with torch.npu.device(x.device.index):
        _rms_norm_fwd_kernel[grid](
            x,
            out,
            weight,
            bias,
            z,
            rstd,
            x.stride(0),
            out.stride(0),
            z.stride(0) if z is not None else 0,
            M,
            group_size,
            eps,
            BLOCK_N=BLOCK_N,
            NORM_BEFORE_GATE=norm_before_gate,
            N_CORES=MAX_CORES,
            num_warps=num_warps,
        )
    return out.reshape(x_shape_og)


class RMSNormGated(nn.Module):
    def __init__(
        self,
        hidden_size,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size, device=device, dtype=torch.bfloat16))
        self.register_parameter('bias', None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x, z=None):
        return rmsnorm_fn(
            x,
            self.weight,
            self.bias,
            z=z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
        )

# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py
# Copyright (c) 2024, Tri Dao.
# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This backward pass is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.
# mypy: ignore-errors

import torch
import triton
import triton.language as tl

MAX_CORES = 65535


@triton.heuristics({
    "HAS_BIAS": lambda args: args["B"] is not None,
    "HAS_Z": lambda args: args["Z"] is not None,
})
@triton.jit
def layer_norm_fwd_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X_base
    N,  # number of columns in X_base
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    N_CORES: tl.constexpr,
):
    # Map the program id to the row of X_base and Y_base it should compute.
    row = tl.program_id(0)
    group = tl.program_id(1)

    BLOCK_ROWS = M if M < N_CORES else N_CORES
    n_iters = M // BLOCK_ROWS
    remain = M % BLOCK_ROWS
    if row < remain:
        n_iters = n_iters + 1

    for i in tl.range(n_iters):
        X_base = X + (i * BLOCK_ROWS *
                      stride_x_row) + row * stride_x_row + group * N
        Y_base = Y + (i * BLOCK_ROWS *
                      stride_y_row) + row * stride_y_row + group * N
        if HAS_Z:
            Z_base = Z + (i * BLOCK_ROWS *
                          stride_z_row) + row * stride_z_row + group * N
        if not IS_RMS_NORM:
            Mean_base = Mean + (i * BLOCK_ROWS) + group * M
        Rstd_base = Rstd + (i * BLOCK_ROWS) + group * M
        W_base = W + group * N
        if HAS_BIAS:
            B_base = B + group * N
        # Compute mean and variance
        cols = tl.arange(0, BLOCK_N)
        x = tl.load(X_base + cols, mask=cols < N, other=0.).to(tl.float32)
        if HAS_Z and not NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=cols < N).to(tl.float32)
            x *= z * tl.sigmoid(z)
        if not IS_RMS_NORM:
            mean = tl.sum(x, axis=0) / N
            tl.store(Mean_base + row, mean)
            xbar = tl.where(cols < N, x - mean, 0.)
            var = tl.sum(xbar * xbar, axis=0) / N
        else:
            xbar = tl.where(cols < N, x, 0.)
            var = tl.sum(xbar * xbar, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        tl.store(Rstd_base + row, rstd)
        # Normalize and apply linear transformation
        mask = cols < N
        w = tl.load(W_base + cols, mask=mask).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(B_base + cols, mask=mask).to(tl.float32)
        x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        y = x_hat * w + b if HAS_BIAS else x_hat * w
        if HAS_Z and NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            y *= z * tl.sigmoid(z)
        # Write output
        tl.store(Y_base + cols, y, mask=mask)


def _layer_norm_fwd(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N, )
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N, )
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = (torch.empty((ngroups * M, ), dtype=torch.float32, device=x.device)
            if not is_rms_norm else None)
    rstd = torch.empty((ngroups * M, ), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    grid = (M if M < MAX_CORES else MAX_CORES, ngroups)
    with torch.npu.device(x.device.index):
        layer_norm_fwd_kernel[grid](
            x,
            out,
            weight,
            bias,
            z,
            mean,
            rstd,
            x.stride(0),
            out.stride(0),
            z.stride(0) if z is not None else 0,
            M,
            group_size,
            eps,
            BLOCK_N=BLOCK_N,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
            N_CORES=MAX_CORES,
            num_warps=num_warps,
        )
    return out, mean, rstd


class LayerNormFn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        z=None,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=False,
    ):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""

        x_shape_og = x.shape
        # reshape input data into 2D tensor
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
        y, mean, rstd = _layer_norm_fwd(
            x,
            weight,
            bias,
            eps,
            z=z,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
        )
        return y.reshape(x_shape_og)


def rmsnorm_fn(
    x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True
):
    return LayerNormFn.apply(
        x, weight, bias, z, eps, group_size, norm_before_gate, True
    )

    # def forward_native(
    #     self, x: torch.Tensor, z: torch.Tensor | None = None
    # ) -> torch.Tensor:
    #     """
    #     Native PyTorch implementation of RMS normalization with gating.

    #     Args:
    #         x: Input tensor
    #         z: Optional gating tensor

    #     Returns:
    #         Normalized (and optionally gated) tensor

    #     If z is not None:
    #         - norm_before_gate=True: out = norm(x) * silu(z)
    #         - norm_before_gate=False: out = norm(x * silu(z))
    #     """
    #     # Apply gating before normalization if needed
    #     if z is not None and not self.norm_before_gate:
    #         x = x * F.silu(z)

    #     # RMS Normalization
    #     if self.group_size is None:
    #         # Standard RMS norm across the last dimension
    #         variance = x.pow(2).mean(dim=-1, keepdim=True)
    #         x_normed = x * torch.rsqrt(variance + self.eps)
    #         out = x_normed * self.weight
    #     else:
    #         # Group RMS norm
    #         from einops import rearrange

    #         x_group = rearrange(x, "... (g d) -> ... g d", d=self.group_size)
    #         variance = x_group.pow(2).mean(dim=-1, keepdim=True)
    #         x_normed = x_group * torch.rsqrt(variance + self.eps)
    #         out = rearrange(x_normed, "... g d -> ... (g d)") * self.weight

    #     # Apply gating after normalization if needed
    #     if z is not None and self.norm_before_gate:
    #         out = out * F.silu(z)

    #     return out

    # def forward_cuda(
    #     self, x: torch.Tensor, z: torch.Tensor | None = None
    # ) -> torch.Tensor:
    #     from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn

    #     return rmsnorm_fn(
    #         x,
    #         self.weight,
    #         self.bias,
    #         z=z,
    #         eps=self.eps,
    #         group_size=self.group_size,
    #         norm_before_gate=self.norm_before_gate,
    #     )
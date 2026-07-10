# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl


def _get_block_d(dim: int) -> int:
    if dim <= 64:
        return triton.next_power_of_2(dim)
    if dim <= 128:
        return 128
    return 256


@triton.jit
def _hc_pre_reduce_kernel(
    x_ptr,
    pre_ptr,
    out_ptr,
    x_stride_n,
    x_stride_h,
    x_stride_d,
    pre_stride_n,
    pre_stride_h,
    out_stride_n,
    out_stride_d,
    dim: tl.constexpr,
    hc_mult: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_id = tl.program_id(0)
    d_tile = tl.program_id(1)
    offs_d = d_tile * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs_d < dim

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for hc_id in range(hc_mult):
        weight = tl.load(pre_ptr + row_id * pre_stride_n + hc_id * pre_stride_h).to(tl.float32)
        x = tl.load(
            x_ptr + row_id * x_stride_n + hc_id * x_stride_h + offs_d * x_stride_d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += weight * x

    tl.store(out_ptr + row_id * out_stride_n + offs_d * out_stride_d, acc, mask=mask)


@triton.jit
def _hc_post_expand_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    x_stride_n,
    x_stride_d,
    residual_stride_n,
    residual_stride_h,
    residual_stride_d,
    post_stride_n,
    post_stride_h,
    comb_stride_n,
    comb_stride_out_h,
    comb_stride_src_h,
    out_stride_n,
    out_stride_h,
    out_stride_d,
    dim: tl.constexpr,
    hc_mult: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_h = tl.program_id(0)
    row_id = row_h // hc_mult
    out_h = row_h - row_id * hc_mult
    d_tile = tl.program_id(1)
    offs_d = d_tile * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs_d < dim

    post = tl.load(post_ptr + row_id * post_stride_n + out_h * post_stride_h).to(tl.float32)
    x = tl.load(x_ptr + row_id * x_stride_n + offs_d * x_stride_d, mask=mask, other=0.0).to(tl.float32)
    acc = post * x

    for src_h in range(hc_mult):
        weight = tl.load(
            comb_ptr + row_id * comb_stride_n + out_h * comb_stride_out_h + src_h * comb_stride_src_h,
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + row_id * residual_stride_n + src_h * residual_stride_h + offs_d * residual_stride_d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += weight * residual

    tl.store(out_ptr + row_id * out_stride_n + out_h * out_stride_h + offs_d * out_stride_d, acc, mask=mask)


def hc_pre_reduce(
    x: torch.Tensor,
    pre: torch.Tensor,
    hc_mult: int,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Reduce DeepSeek-V4 HC states from ``[..., hc, dim]`` to ``[..., dim]``."""
    if out_dtype is None:
        out_dtype = x.dtype
    dim = x.size(-1)
    out_shape = (*x.shape[:-2], dim)
    out = torch.empty(out_shape, device=x.device, dtype=out_dtype)
    if x.numel() == 0:
        return out

    x = x.reshape(-1, hc_mult, dim)
    pre = pre.reshape(-1, hc_mult)
    out = out.reshape(-1, dim)
    n_rows = x.size(0)
    block_d = _get_block_d(dim)
    grid = (n_rows, triton.cdiv(dim, block_d))
    _hc_pre_reduce_kernel[grid](
        x,
        pre,
        out,
        *x.stride(),
        *pre.stride(),
        *out.stride(),
        dim,
        hc_mult,
        block_d,
        num_warps=4,
    )
    return out.reshape(out_shape)


def hc_post_expand(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    hc_mult: int,
) -> torch.Tensor:
    """Expand DeepSeek-V4 HC states from ``[..., dim]`` to ``[..., hc, dim]``."""
    dim = x.size(-1)
    out_shape = (*x.shape[:-1], hc_mult, dim)
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    if x.numel() == 0:
        return out

    x = x.reshape(-1, dim)
    residual = residual.reshape(-1, hc_mult, dim)
    post = post.reshape(-1, hc_mult)
    comb = comb.reshape(-1, hc_mult, hc_mult)
    out = out.reshape(-1, hc_mult, dim)
    n_rows = x.size(0)
    block_d = _get_block_d(dim)
    grid = (n_rows * hc_mult, triton.cdiv(dim, block_d))
    _hc_post_expand_kernel[grid](
        x,
        residual,
        post,
        comb,
        out,
        *x.stride(),
        *residual.stride(),
        *post.stride(),
        *comb.stride(),
        *out.stride(),
        dim,
        hc_mult,
        block_d,
        num_warps=4,
    )
    return out.reshape(out_shape)

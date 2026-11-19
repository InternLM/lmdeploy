# Copyright (c) OpenMMLab. All rights reserved.
"""Shared GLM forward boundaries for training and serving."""

import torch
import torch.nn.functional as F

from lmdeploy.pytorch import envs

ALIGN_FORWARD = envs.glm53_align_forward
ALIGN_ROUTER_PROJECTION = ALIGN_FORWARD and envs.glm53_align_router_projection


@torch.library.custom_op('lmdeploy::glm53_aligned_router_linear', mutates_args=())
def _router_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from lmdeploy.pytorch.kernels.cuda.glm53_alignment import router_linear

    rows = x.reshape(-1, 4096).contiguous()
    out = x.new_empty((*x.shape[:-1], 288), dtype=torch.float32)
    if rows.size(0):
        router_linear[((rows.size(0) + 31) // 32, 5)](
            rows, weight, out, rows.size(0), *weight.stride(), num_warps=4)
    return out


@_router_linear.register_fake
def _router_linear_fake(x, weight):
    return x.new_empty((*x.shape[:-1], 288), dtype=torch.float32)


def _router_linear_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _router_linear_backward(ctx, grad):
    x, weight = ctx.saved_tensors
    with torch.autocast(device_type=x.device.type, enabled=False):
        dy = grad.reshape(-1, 288).float()
        dx = ((dy @ weight.float()).reshape(x.shape).to(x.dtype)
              if ctx.needs_input_grad[0] else None)
        dw = ((dy.t() @ x.reshape(-1, 4096).float()).to(weight.dtype)
              if ctx.needs_input_grad[1] else None)
    return dx, dw


_router_linear.register_autograd(_router_linear_backward, setup_context=_router_linear_setup)


def aligned_router_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Keep FP32 router weights and logits; avoid a full FP32 input copy in forward."""
    if (x.ndim < 1 or x.size(-1) != 4096 or weight.shape != (288, 4096)
            or x.device != weight.device or x.dtype != torch.bfloat16
            or weight.dtype not in (torch.bfloat16, torch.float32)):
        raise ValueError('Aligned GLM router requires BF16 input and BF16/FP32 [288,4096] weights')
    with torch.autocast(device_type=x.device.type, enabled=False):
        if x.is_cuda:
            return _router_linear(x, weight)
        return F.linear(x.float(), weight.float())


@torch.library.custom_op('lmdeploy::glm53_aligned_linear', mutates_args=())
def _linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    from lmdeploy.pytorch.kernels.cuda.glm53_alignment import head_linear, linear, mhc_linear

    rows = x.to(weight.dtype).reshape(-1, x.size(-1)).contiguous()
    out = x.new_empty((*x.shape[:-1], weight.size(0)), dtype=weight.dtype)
    if rows.size(0) and weight.size(0):
        if weight.dtype == torch.float32 and weight.shape == (24, 16384):
            # The fixed 32x64 GEMM wastes rows/columns and leaves only one CTA
            # active on decode. This row-independent path also reduces FP32
            # accumulation error without an activation-sized FP64 workspace.
            mhc_linear[(rows.size(0) * 6,)](
                rows, weight, bias, out, 24, 16384, weight.stride(0), weight.stride(1),
                bias is not None, num_warps=8)
        elif (weight.dtype == torch.float32 and weight.size(1) == 4096 and weight.size(0) >= 4096
              and torch.cuda.get_device_capability(weight.device)[0] >= 8):
            # Only output tiling changes: both tiles use identical 32-feature
            # TF32x3 steps and even/odd accumulation order for every row.
            bm, bn = (16, 64) if rows.size(0) <= 16 else (64, 64)
            head_linear[((rows.size(0) + bm - 1) // bm, (weight.size(0) + bn - 1) // bn)](
                rows, weight, bias, out, rows.size(0), weight.size(0), rows.size(1),
                weight.stride(0), weight.stride(1), bias is not None, bm, bn, num_warps=4)
        else:
            linear[((rows.size(0) + 31) // 32, (weight.size(0) + 63) // 64)](
                rows, weight, bias, out, rows.size(0), weight.size(0), rows.size(1),
                weight.stride(0), weight.stride(1), bias is not None, num_warps=4)
    return out


@_linear.register_fake
def _linear_fake(x, weight, bias):
    return x.new_empty((*x.shape[:-1], weight.size(0)), dtype=weight.dtype)


def _linear_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1])
    ctx.has_bias = inputs[2] is not None


def _linear_backward(ctx, grad):
    x, weight = ctx.saved_tensors
    with torch.autocast(device_type=x.device.type, enabled=False):
        rows = x.reshape(-1, x.size(-1)).to(weight.dtype)
        dy = grad.reshape(-1, weight.size(0)).to(weight.dtype)
        dx = (dy @ weight).reshape(x.shape).to(x.dtype) if ctx.needs_input_grad[0] else None
        dw = dy.t() @ rows if ctx.needs_input_grad[1] else None
        db = dy.sum(0) if ctx.has_bias and ctx.needs_input_grad[2] else None
    return dx, dw, db


_linear.register_autograd(_linear_backward, setup_context=_linear_setup)


def aligned_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Fixed forward tiles with native GEMM gradients; no per-token launches."""
    if x.ndim == 0 or weight.ndim != 2 or x.size(-1) != weight.size(1) or x.device != weight.device:
        raise ValueError('Aligned projection requires matching feature dimensions and devices')
    if bias is not None and (bias.shape != (weight.size(0),) or bias.device != x.device or bias.dtype != weight.dtype):
        raise ValueError('Aligned projection bias must match the output dimension, weight dtype and device')
    with torch.autocast(device_type=x.device.type, enabled=False):
        if x.is_cuda and weight.dtype in (torch.float32, torch.bfloat16, torch.float16):
            return _linear(x, weight, bias)
        return F.linear(x.to(weight.dtype), weight, bias)


def kda_output_projection(x: torch.Tensor, projection) -> torch.Tensor:
    """Use a128-row native GEMM for small GLM BF16 output projections.

    Measured for the8192-to4096 TP1 projection on Hopper. Only the internal
    operand rows change; returned shape/dtype and model token metadata stay
    native. Other layouts, precisions, adapters, collectives and grad-enabled
    calls retain the original module. No FP32 activation copy is created.
    """
    weight = getattr(projection, 'weight', None)
    eligible = (
        x.ndim > 0 and x.is_cuda and x.dtype == torch.bfloat16
        and x.shape[-1] == 8192 and 0 < x.numel() // 8192 < 128
        and isinstance(weight, torch.Tensor) and weight.shape == (4096, 8192)
        and weight.dtype == x.dtype and weight.device == x.device
        and weight.is_contiguous() and getattr(projection, 'bias', None) is None
        and getattr(projection, 'tp', 1) == 1
        and not getattr(projection, 'all_reduce', False)
        and not getattr(projection, 'dp_gather', False)
        and not getattr(projection, 'lora_adapters', {})
        and not (torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad)))
    if not eligible:
        return projection(x)
    rows = x.numel() // 8192
    with torch.autocast(device_type='cuda', enabled=False):
        operand = x.new_zeros((128, 8192))
        operand[:rows].copy_(x.reshape(rows, 8192))
        output = F.linear(operand, weight)
    return output[:rows].reshape(*x.shape[:-1], 4096)


def kda_gate_projections(x, beta_proj, low_proj, forget_proj, output_proj, batched_proj):
    """Project BF16 KDA gates with fixed reductions and no FP32 activation copy.

    Keep original module dispatch for adapters, gathering and other precisions.
    These projections are local column shards, so ordinary TP needs no new
    collective. DP/TP gathering retains its original execution path.
    """
    from lmdeploy.pytorch.config import TPMode

    projections = (beta_proj, low_proj, forget_proj, output_proj)

    def supports_fixed_projection(projection):
        weight = getattr(projection, 'weight', None)
        return (
            isinstance(weight, torch.Tensor)
            and weight.ndim == 2
            and weight.dtype == x.dtype
            and weight.device == x.device
            and getattr(projection, 'bias', None) is None
            and not getattr(projection, 'lora_adapters', {})
            and not getattr(projection, 'dp_gather', False)
            and not getattr(projection, 'all_reduce', False)
            and not (getattr(projection, 'tp', 1) > 1
                     and getattr(projection, 'tp_mode', None) == TPMode.DP_TP)
        )

    eligible = x.is_cuda and x.dtype == torch.bfloat16 and all(
        supports_fixed_projection(projection) for projection in projections)
    if not eligible:
        beta = beta_proj(x)
        forget, output = batched_proj(low_proj(x))
        return beta, forget, output
    beta = aligned_linear(x, beta_proj.weight)
    low = aligned_linear(x, low_proj.weight)
    forget_low, output_low = low.split(forget_proj.in_features, dim=-1)
    forget = aligned_linear(forget_low, forget_proj.weight)
    output = aligned_linear(output_low, output_proj.weight)
    return beta, forget, output


def hc_pre(x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor,
           norm_weight: torch.Tensor, iters: int, eps: float, norm_eps: float):
    """Use the aligned HC path for inference-only GLM forward passes."""
    if x.numel() == 0:
        shape, h = x.shape[:-2], x.size(-2)
        return (x.new_empty((*shape, x.size(-1))), scale.new_empty((*shape, h)),
                scale.new_empty((*shape, h, h)))
    from lmdeploy.pytorch.kernels.cuda.dsv4.hc_prepost import hc_pre_reduce_norm
    from lmdeploy.pytorch.kernels.cuda.dsv4.hc_split_sinkhorn import hc_split_sinkhorn
    from lmdeploy.pytorch.nn.norm import rms_scale

    flat = x.flatten(-2).float()
    mixes = aligned_linear(flat, fn.float())
    mixes = rms_scale(mixes, flat, eps=norm_eps)
    pre, post, comb = hc_split_sinkhorn(mixes, scale.float(), base.float(), x.size(-2), iters, eps)
    y = hc_pre_reduce_norm(x, pre, x.size(-2), norm_weight, norm_eps, x.dtype)
    return y, post, comb

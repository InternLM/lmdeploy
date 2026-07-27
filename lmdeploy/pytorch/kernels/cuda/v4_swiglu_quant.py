# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
from torch import Tensor

from .activation import fast_expf
from .blocked_gemm_fp8 import fast_round_scale
from .utils import get_device_props


@triton.jit(do_not_specialize=['M'])
def _v4_swiglu_quant_fp8_kernel(
    gateup_ptr,
    out_ptr,
    scale_ptr,
    M,
    K: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    stride_gum,
    stride_gun: tl.constexpr,
    stride_om,
    stride_on: tl.constexpr,
    stride_sm,
    stride_sg,
    SWIGLU_LIMIT: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS_PER_CTA: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    group_id = tl.program_id(0) * NUM_GROUPS_PER_CTA
    m_id_start = tl.program_id(1)
    m_id_stride = tl.num_programs(1)

    GROUP_SIZE_CTA: tl.constexpr = GROUP_SIZE * NUM_GROUPS_PER_CTA
    offs_n = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE_CTA)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, GROUP_SIZE), GROUP_SIZE)
    offs_s = group_id + tl.arange(0, NUM_GROUPS_PER_CTA)
    rfp8_max = 1 / fp8_max

    if K % GROUP_SIZE_CTA == 0:
        mask_n = True
        mask_s = True
    else:
        mask_n = offs_n < K
        mask_s = offs_s < tl.cdiv(K, GROUP_SIZE)

    gate_ptrs = gateup_ptr + m_id_start * stride_gum + offs_n * stride_gun
    up_ptrs = gateup_ptr + m_id_start * stride_gum + (K + offs_n) * stride_gun
    out_ptrs = out_ptr + m_id_start * stride_om + offs_n * stride_on
    scale_ptrs = scale_ptr + m_id_start * stride_sm + offs_s * stride_sg

    for _ in tl.range(m_id_start, M, m_id_stride, num_stages=NUM_STAGES):
        gate = tl.load(gate_ptrs, mask=mask_n, other=0.0).to(tl.float32)
        up = tl.load(up_ptrs, mask=mask_n, other=0.0).to(tl.float32)

        if SWIGLU_LIMIT > 0:
            up = tl.minimum(tl.maximum(up, -SWIGLU_LIMIT), SWIGLU_LIMIT)
            gate = tl.minimum(gate, SWIGLU_LIMIT)

        act = gate / (1 + fast_expf(-gate)) * up
        act = act.to(gateup_ptr.dtype.element_ty).to(tl.float32)
        act = act.reshape(NUM_GROUPS_PER_CTA, GROUP_SIZE)

        amax = tl.max(tl.abs(act), axis=1)
        amax = tl.maximum(amax, 1e-6).to(tl.float32)
        if ROUND_SCALE == 1:
            scale = fast_round_scale(amax, rfp8_max)
            rscale = 1 / scale
        else:
            scale = amax * rfp8_max
            rscale = fp8_max / amax

        out = act * rscale[:, None]
        out = tl.clamp(out, fp8_min, fp8_max)
        out = out.to(out_ptr.dtype.element_ty)
        out = out.reshape(GROUP_SIZE_CTA)
        tl.store(out_ptrs, out, mask=mask_n)
        tl.store(scale_ptrs, scale, mask=mask_s)

        gate_ptrs += m_id_stride * stride_gum
        up_ptrs += m_id_stride * stride_gum
        out_ptrs += m_id_stride * stride_om
        scale_ptrs += m_id_stride * stride_sm


def v4_swiglu_and_quant_fp8(gate_up: Tensor,
                            swiglu_limit: float = 0.0,
                            group_size: int = 128,
                            dtype: torch.dtype = torch.float8_e4m3fn,
                            scale_fmt: str | None = 'ue8m0') -> tuple[Tensor, Tensor]:
    """Apply DeepSeek-V4 SwiGLU and directly quantize the result to FP8."""
    assert scale_fmt in (None, 'ue8m0')
    assert gate_up.dim() >= 2
    assert gate_up.stride(-1) == 1, 'last dimension must be contiguous'
    hidden = gate_up.size(-1) // 2
    assert gate_up.size(-1) == hidden * 2
    assert hidden % group_size == 0

    out = gate_up.new_empty(*gate_up.shape[:-1], hidden, dtype=dtype)
    scales = gate_up.new_empty(*gate_up.shape[:-1], hidden // group_size, dtype=torch.float32)
    if gate_up.numel() == 0:
        return out, scales

    gate_up_2d = gate_up.reshape(-1, hidden * 2)
    out_2d = out.reshape(-1, hidden)
    scales_2d = scales.reshape(-1, hidden // group_size)
    M = gate_up_2d.size(0)

    finfo = torch.finfo(dtype)
    num_warps = 4
    num_groups_per_cta = 4
    grid_size0 = triton.cdiv(hidden, group_size * num_groups_per_cta)
    props = get_device_props(gate_up.device.index)
    num_sm = props['multi_processor_count']
    warps_per_sm = props['warps_per_sm']
    blocks_per_sm = props['blocks_per_sm']
    max_ctas = num_sm * min(blocks_per_sm, warps_per_sm // num_warps)
    grid_size1 = min(M, max(1, max_ctas // grid_size0))
    assert grid_size0 < 65536 and grid_size1 < 65536

    num_stages = min(4, max(1, triton.cdiv(M, grid_size1)))
    _v4_swiglu_quant_fp8_kernel[(grid_size0, grid_size1)](
        gate_up_2d,
        out_2d,
        scales_2d,
        M,
        K=hidden,
        fp8_min=finfo.min,
        fp8_max=finfo.max,
        stride_gum=gate_up_2d.stride(0),
        stride_gun=gate_up_2d.stride(1),
        stride_om=out_2d.stride(0),
        stride_on=out_2d.stride(1),
        stride_sm=scales_2d.stride(0),
        stride_sg=scales_2d.stride(1),
        SWIGLU_LIMIT=swiglu_limit,
        ROUND_SCALE=1 if scale_fmt == 'ue8m0' else 0,
        GROUP_SIZE=group_size,
        NUM_GROUPS_PER_CTA=num_groups_per_cta,
        NUM_STAGES=num_stages,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, scales

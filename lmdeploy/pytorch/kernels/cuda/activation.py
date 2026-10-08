# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .blocked_gemm_fp8 import fast_round_scale
from .utils import get_device_props

fast_expf = tl.math.exp


@triton.jit
def _silu_and_mul_kernel(
    gateup_ptr,
    out_ptr,
    N: tl.constexpr,
    M,
    stride_gum: tl.constexpr,
    stride_gun: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
    PRECISE_MUL: tl.constexpr,
):
    """Silu and mul kernel."""
    n_block_id = tl.program_id(0)
    m_id_start = tl.program_id(1)
    m_id_stride = tl.num_programs(1)

    up_ptr = gateup_ptr + N * stride_gun
    offs_n = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if N % BLOCK_SIZE_N == 0:
        mask = None
    else:
        mask = offs_n < N

    gate_ptrs = gateup_ptr + m_id_start * stride_gum + offs_n * stride_gun
    up_ptrs = up_ptr + m_id_start * stride_gum + offs_n * stride_gun
    out_ptrs = out_ptr + m_id_start * stride_om + offs_n * stride_on

    for _ in tl.range(m_id_start, M, m_id_stride):
        gate = tl.load(gate_ptrs, mask=mask)
        up = tl.load(up_ptrs, mask=mask)
        if SWIGLU_LIMIT is not None:
            gate = tl.minimum(gate, SWIGLU_LIMIT)
            up = tl.maximum(tl.minimum(up, SWIGLU_LIMIT), -SWIGLU_LIMIT)
        # exp expect fp32
        gate = gate.to(tl.float32)

        if PRECISE_MUL:
            exp_neg_gate = libdevice.exp(-gate)
        else:
            exp_neg_gate = fast_expf(-gate)
        gate = gate / (1 + exp_neg_gate)
        if not PRECISE_MUL:
            gate = gate.to(gateup_ptr.dtype.element_ty)
        out = gate * up

        tl.store(out_ptrs, out, mask=mask)

        gate_ptrs += m_id_stride * stride_gum
        up_ptrs += m_id_stride * stride_gum
        out_ptrs += m_id_stride * stride_om


def silu_and_mul(gate_up: torch.Tensor,
                 out: torch.Tensor = None,
                 swiglu_limit: float | None = None,
                 precise_mul: bool = False):
    """Silu and mul."""
    assert gate_up.dim() == 2

    M = gate_up.size(0)
    N = gate_up.size(-1) // 2
    if out is None:
        out_shape = (M, N)
        out = gate_up.new_empty(out_shape)

    BLOCK_SIZE_N = triton.next_power_of_2(N)
    BLOCK_SIZE_N = min(BLOCK_SIZE_N, 512)
    num_warps = 4
    num_stages = 1

    props = get_device_props(gate_up.device.index)
    num_sm = props['multi_processor_count']
    warps_per_sm = props['warps_per_sm']
    grid_size0 = triton.cdiv(N, BLOCK_SIZE_N)
    grid_size1 = min(M, num_sm * warps_per_sm // num_warps)
    assert grid_size0 < 65536 and grid_size1 < 65536
    grid = (grid_size0, grid_size1)
    _silu_and_mul_kernel[grid](gate_up,
                               out,
                               N,
                               M,
                               stride_gum=gate_up.stride(0),
                               stride_gun=gate_up.stride(1),
                               stride_om=out.stride(0),
                               stride_on=out.stride(1),
                               BLOCK_SIZE_N=BLOCK_SIZE_N,
                               SWIGLU_LIMIT=swiglu_limit,
                               PRECISE_MUL=precise_mul,
                               num_warps=num_warps,
                               num_stages=num_stages)

    return out


@triton.jit
def _silu_and_mul_moe_ep_kernel(
    gateup_ptr,
    out_ptr,
    mask_ptr,
    scale_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    stride_gue: tl.constexpr,
    stride_gum: tl.constexpr,
    stride_gun: tl.constexpr,
    stride_oe: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    stride_m: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
    PRECISE_MUL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Silu and mul kernel."""
    n_block_id = tl.program_id(0)
    # Expert-capacity strides can exceed the signed 32-bit element range.
    e_id = tl.program_id(1).to(tl.int64)
    m_id_start = tl.program_id(2)
    m_id_stride = tl.num_programs(2)

    offs_n = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if N % BLOCK_SIZE_N == 0:
        mask = None
    else:
        mask = offs_n < N

    mask_m = tl.load(mask_ptr + e_id * stride_m)
    mask_m = tl.minimum(mask_m, M)
    if mask_m <= m_id_start:
        return
    gate_ptrs = gateup_ptr + e_id * stride_gue + m_id_start * stride_gum + offs_n * stride_gun
    up_ptrs = gate_ptrs + N * stride_gun
    out_ptrs = out_ptr + e_id * stride_oe + m_id_start * stride_om + offs_n * stride_on

    for m_id in tl.range(m_id_start, mask_m, m_id_stride):
        gate = tl.load(gate_ptrs, mask=mask)
        up = tl.load(up_ptrs, mask=mask)
        if SWIGLU_LIMIT is not None:
            gate = tl.minimum(gate, SWIGLU_LIMIT)
            up = tl.maximum(tl.minimum(up, SWIGLU_LIMIT), -SWIGLU_LIMIT)
        gate = gate.to(tl.float32)
        if PRECISE_MUL:
            exp_neg_gate = libdevice.exp(-gate)
        else:
            exp_neg_gate = fast_expf(-gate)
        gate = gate / (1 + exp_neg_gate)
        if not PRECISE_MUL:
            gate = gate.to(gateup_ptr.dtype.element_ty)
        out = gate * up

        if scale_ptr is not None:
            # Preserve the materialized activation's rounding before quantization.
            out = out.to(gateup_ptr.dtype.element_ty)
            groups: tl.constexpr = BLOCK_SIZE_N // GROUP_SIZE
            values = out.reshape(groups, GROUP_SIZE)
            amax = tl.max(tl.abs(values), axis=1)
            amax = tl.maximum(amax, 1e-6).to(tl.float32)
            if ROUND_SCALE:
                scale = fast_round_scale(amax, 1 / FP8_MAX)
                rscale = 1 / scale
            else:
                scale = amax * (1 / FP8_MAX)
                rscale = FP8_MAX / amax
            out = values.to(tl.float32) * rscale[:, None]
            out = tl.clamp(out, FP8_MIN, FP8_MAX).reshape(BLOCK_SIZE_N)
            group_offsets = n_block_id * groups + tl.arange(0, groups)
            scale_offsets = (e_id * M + m_id) * (N // GROUP_SIZE) + group_offsets
            tl.store(scale_ptr + scale_offsets, scale, group_offsets < N // GROUP_SIZE)

        tl.store(out_ptrs, out, mask=mask)

        gate_ptrs += m_id_stride * stride_gum
        up_ptrs += m_id_stride * stride_gum
        out_ptrs += m_id_stride * stride_om


def silu_and_mul_moe_ep(gate_up: torch.Tensor, mask_m: torch.Tensor, out: torch.Tensor = None,
                        swiglu_limit: float | None = None, precise_mul: bool = False,
                        output_scale: torch.Tensor = None, quant_group_size: int = 128,
                        scale_fmt: str | None = None):
    """Silu and mul for moe with expert parallelism."""
    # gate_up: [num_experts, batch_size, 2*hidden_size]
    assert gate_up.dim() == 3
    assert mask_m.dim() == 1
    assert mask_m.size(0) == gate_up.size(0)

    stride_m = mask_m.stride(0)
    assert gate_up.size(0) % stride_m == 0

    E = gate_up.size(0)
    M = gate_up.size(1)
    N = gate_up.size(-1) // 2
    if out is None:
        out_shape = (E, M, N)
        out = gate_up.new_empty(out_shape)

    BLOCK_SIZE_N = triton.next_power_of_2(N)
    BLOCK_SIZE_N = min(BLOCK_SIZE_N, 512)
    num_warps = 4
    num_stages = 1

    props = get_device_props(gate_up.device.index)
    num_sm = props['multi_processor_count']
    warps_per_sm = props['warps_per_sm']
    ctas_per_sm = warps_per_sm // num_warps
    ctas_per_device = num_sm * ctas_per_sm
    grid_size0 = triton.cdiv(N, BLOCK_SIZE_N)
    grid_size1 = min(M, triton.cdiv(ctas_per_device, grid_size0 * E))
    grid = (grid_size0, E, grid_size1)
    if output_scale is not None:
        assert out.is_contiguous() and output_scale.is_contiguous()
        assert out.shape == (E, M, N)
        assert output_scale.shape == (E, M, N // quant_group_size)
        assert N % quant_group_size == 0 and BLOCK_SIZE_N % quant_group_size == 0
        assert scale_fmt in (None, 'ue8m0')
        finfo = torch.finfo(out.dtype)
    _silu_and_mul_moe_ep_kernel[grid](gate_up,
                                      out,
                                      mask_m,
                                      output_scale,
                                      N,
                                      M,
                                      stride_gue=gate_up.stride(0),
                                      stride_gum=gate_up.stride(1),
                                      stride_gun=gate_up.stride(2),
                                      stride_oe=out.stride(0),
                                      stride_om=out.stride(1),
                                      stride_on=out.stride(2),
                                      stride_m=mask_m.stride(0),
                                      BLOCK_SIZE_N=BLOCK_SIZE_N,
                                      SWIGLU_LIMIT=swiglu_limit,
                                      PRECISE_MUL=precise_mul,
                                      GROUP_SIZE=quant_group_size,
                                      ROUND_SCALE=scale_fmt == 'ue8m0',
                                      FP8_MIN=finfo.min if output_scale is not None else 0,
                                      FP8_MAX=finfo.max if output_scale is not None else 0,
                                      num_warps=num_warps,
                                      num_stages=num_stages)

    return out


def silu_and_mul_masked_post_quant_fwd(input: torch.Tensor, output: torch.Tensor, output_scale: torch.Tensor,
                                       quant_group_size: int, masked_m: torch.Tensor, act_func=None, *,
                                       swiglu_limit: float | None = None, precise_mul: bool = False,
                                       scale_fmt: str | None = None):
    """Fuse activation and FP8 quantization for valid expert rows only.

    Invalid rows are left untouched; the following masked GEMM must ignore them.
    """
    assert input.is_contiguous()
    assert output.is_contiguous()
    assert input.dim() == 3
    assert input.shape[0] == masked_m.shape[0]
    assert input.shape[-1] % 2 == 0
    size_n = input.shape[-1] // 2
    assert size_n % quant_group_size == 0
    if act_func is not None:
        activated = act_func(input, masked_m=masked_m)
        from .blocked_gemm_fp8 import _quant_fp8_launcher
        _quant_fp8_launcher(activated.reshape(-1, size_n), quant_group_size,
                            output.reshape(-1, size_n),
                            output_scale.reshape(-1, size_n // quant_group_size), scale_fmt=scale_fmt)
        return
    silu_and_mul_moe_ep(input, masked_m, output, swiglu_limit=swiglu_limit,
                        precise_mul=precise_mul, output_scale=output_scale,
                        quant_group_size=quant_group_size, scale_fmt=scale_fmt)

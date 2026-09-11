# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl

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
        # exp expect fp32
        gate = gate.to(tl.float32)

        gate = gate / (1 + fast_expf(-gate))
        gate = gate.to(gateup_ptr.dtype.element_ty)
        out = gate * up

        tl.store(out_ptrs, out, mask=mask)

        gate_ptrs += m_id_stride * stride_gum
        up_ptrs += m_id_stride * stride_gum
        out_ptrs += m_id_stride * stride_om


def silu_and_mul(gate_up: torch.Tensor, out: torch.Tensor = None):
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
                               num_warps=num_warps,
                               num_stages=num_stages)

    return out


@triton.jit(do_not_specialize=['M'])
def _silu_and_mul_post_quant_kernel(
    gateup_ptr,
    out_ptr,
    scale_ptr,
    M,
    N: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    stride_gum,
    stride_gun: tl.constexpr,
    stride_om,
    stride_on: tl.constexpr,
    stride_sm,
    stride_sg,
    ROUND_SCALE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS_PER_CTA: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Apply SiLU-and-mul and directly emit block-quantized FP8 values."""
    group_id = tl.program_id(0) * NUM_GROUPS_PER_CTA
    m_id_start = tl.program_id(1)
    m_id_stride = tl.num_programs(1)

    group_size_cta: tl.constexpr = GROUP_SIZE * NUM_GROUPS_PER_CTA
    offs_n = group_id * GROUP_SIZE + tl.arange(0, group_size_cta)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, GROUP_SIZE), GROUP_SIZE)
    offs_s = group_id + tl.arange(0, NUM_GROUPS_PER_CTA)
    reciprocal_fp8_max = 1.0 / fp8_max

    if N % group_size_cta == 0:
        mask_n = True
        mask_s = True
    else:
        mask_n = offs_n < N
        mask_s = offs_s < tl.cdiv(N, GROUP_SIZE)

    gate_ptrs = gateup_ptr + m_id_start * stride_gum + offs_n * stride_gun
    up_ptrs = gateup_ptr + m_id_start * stride_gum + (N + offs_n) * stride_gun
    out_ptrs = out_ptr + m_id_start * stride_om + offs_n * stride_on
    scale_ptrs = scale_ptr + m_id_start * stride_sm + offs_s * stride_sg

    for _ in tl.range(m_id_start, M, m_id_stride, num_stages=NUM_STAGES):
        gate = tl.load(gate_ptrs, mask=mask_n, other=0.0).to(tl.float32)
        up = tl.load(up_ptrs, mask=mask_n, other=0.0)

        # Preserve the established two-kernel numerical boundary: the SiLU
        # result is rounded to the gate/up dtype before multiplication, and
        # the product is rounded again before its per-group amax is computed.
        gate = gate / (1 + fast_expf(-gate))
        gate = gate.to(gateup_ptr.dtype.element_ty)
        activated = (gate * up).to(gateup_ptr.dtype.element_ty).to(tl.float32)
        activated = activated.reshape(NUM_GROUPS_PER_CTA, GROUP_SIZE)

        amax = tl.max(tl.abs(activated), axis=1)
        amax = tl.maximum(amax, 1e-6).to(tl.float32)
        if ROUND_SCALE:
            scale = fast_round_scale(amax, reciprocal_fp8_max)
            reciprocal_scale = 1.0 / scale
        else:
            scale = amax * reciprocal_fp8_max
            reciprocal_scale = fp8_max / amax

        out = activated * reciprocal_scale[:, None]
        out = tl.clamp(out, fp8_min, fp8_max).to(out_ptr.dtype.element_ty)
        tl.store(out_ptrs, out.reshape(group_size_cta), mask=mask_n)
        tl.store(scale_ptrs, scale, mask=mask_s)

        gate_ptrs += m_id_stride * stride_gum
        up_ptrs += m_id_stride * stride_gum
        out_ptrs += m_id_stride * stride_om
        scale_ptrs += m_id_stride * stride_sm


def silu_and_mul_post_quant(gate_up: torch.Tensor,
                            group_size: int,
                            dtype: torch.dtype = torch.float8_e4m3fn,
                            scale_fmt: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse SiLU-and-mul with per-token-group FP8 quantization.

    The result matches ``quant_fp8(silu_and_mul(gate_up))`` while avoiding the
    materialized BF16/FP16 activation and its second HBM read.
    """
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
    m = gate_up_2d.size(0)

    finfo = torch.finfo(dtype)
    num_warps = 4
    num_groups_per_cta = 4
    grid_size0 = triton.cdiv(hidden, group_size * num_groups_per_cta)
    props = get_device_props(gate_up.device.index)
    max_ctas = props['multi_processor_count'] * min(props['blocks_per_sm'],
                                                    props['warps_per_sm'] // num_warps)
    grid_size1 = min(m, max(1, max_ctas // grid_size0))
    assert grid_size0 < 65536 and grid_size1 < 65536
    num_stages = min(4, max(1, triton.cdiv(m, grid_size1)))

    _silu_and_mul_post_quant_kernel[(grid_size0, grid_size1)](
        gate_up_2d,
        out_2d,
        scales_2d,
        m,
        N=hidden,
        fp8_min=finfo.min,
        fp8_max=finfo.max,
        stride_gum=gate_up_2d.stride(0),
        stride_gun=gate_up_2d.stride(1),
        stride_om=out_2d.stride(0),
        stride_on=out_2d.stride(1),
        stride_sm=scales_2d.stride(0),
        stride_sg=scales_2d.stride(1),
        ROUND_SCALE=scale_fmt == 'ue8m0',
        GROUP_SIZE=group_size,
        NUM_GROUPS_PER_CTA=num_groups_per_cta,
        NUM_STAGES=num_stages,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, scales


@triton.jit
def _silu_and_mul_moe_ep_kernel(
    gateup_ptr,
    out_ptr,
    mask_ptr,
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
):
    """Silu and mul kernel."""
    n_block_id = tl.program_id(0)
    e_id = tl.program_id(1)
    m_id_start = tl.program_id(2)
    m_id_stride = tl.num_programs(2)

    offs_n = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if N % BLOCK_SIZE_N == 0:
        mask = None
    else:
        mask = offs_n < N

    mask_m = tl.load(mask_ptr + e_id * stride_m)
    mask_m = tl.minimum(mask_m, M)
    if mask_m < m_id_start:
        return
    gate_ptrs = gateup_ptr + e_id * stride_gue + m_id_start * stride_gum + offs_n * stride_gun
    up_ptrs = gate_ptrs + N * stride_gun
    out_ptrs = out_ptr + e_id * stride_oe + m_id_start * stride_om + offs_n * stride_on

    for _ in tl.range(m_id_start, mask_m, m_id_stride):
        gate = tl.load(gate_ptrs, mask=mask)
        up = tl.load(up_ptrs, mask=mask)
        # exp expect fp32
        gate = gate.to(tl.float32)
        gate = gate / (1 + fast_expf(-gate))
        gate = gate.to(gateup_ptr.dtype.element_ty)
        out = gate * up

        tl.store(out_ptrs, out, mask=mask)

        gate_ptrs += m_id_stride * stride_gum
        up_ptrs += m_id_stride * stride_gum
        out_ptrs += m_id_stride * stride_om


def silu_and_mul_moe_ep(gate_up: torch.Tensor, mask_m: torch.Tensor, out: torch.Tensor = None):
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
    _silu_and_mul_moe_ep_kernel[grid](gate_up,
                                      out,
                                      mask_m,
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
                                      num_warps=num_warps,
                                      num_stages=num_stages)

    return out


def silu_and_mul_masked_post_quant_fwd(input: torch.Tensor, output: torch.Tensor, output_scale: torch.Tensor,
                                       quant_group_size: int, masked_m: torch.Tensor):
    """Apply masked MoE SiLU-and-mul, then quantize to the preallocated FP8
    output."""
    assert input.is_contiguous()
    assert output.is_contiguous()
    assert input.dim() == 3
    assert input.shape[0] == masked_m.shape[0]
    assert input.shape[-1] % 2 == 0
    size_n = input.shape[-1] // 2
    assert size_n % quant_group_size == 0
    activated = silu_and_mul_moe_ep(input, masked_m)
    from .blocked_gemm_fp8 import _quant_fp8_launcher
    _quant_fp8_launcher(activated.reshape(-1, size_n),
                        quant_group_size,
                        output.reshape(-1, size_n),
                        output_scale.reshape(-1, size_n // quant_group_size))

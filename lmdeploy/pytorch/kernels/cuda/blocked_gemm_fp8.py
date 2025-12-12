# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor

from lmdeploy.utils import get_logger

from .utils import get_device_props

logger = get_logger('lmdeploy')


@triton.jit
def fast_log2_ceil(x):
    bits_x = tl.cast(x, tl.uint32, bitcast=True)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    tmp = exp_x - 127 + tl.where(man_bits != 0, 1, 0)
    return tl.cast(tmp, tl.int32)


@triton.jit
def fast_pow2(x):
    bits_x = (x + 127) << 23
    return tl.cast(bits_x, tl.float32, bitcast=True)


@triton.jit
def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


@triton.jit(do_not_specialize=['M', 'M_out'])
def _quant_fp8_kernel(
    a_ptr,
    out_ptr,
    scale_ptr,
    M,
    M_out,
    K: tl.constexpr,
    num_groups_per_cta: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    stride_am,
    stride_ak: tl.constexpr,
    stride_om,
    stride_ok: tl.constexpr,
    stride_sm,
    stride_sg,
    ROUND_SCALE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Quant fp8 kernel."""
    group_id = tl.program_id(0) * num_groups_per_cta
    m_id_start = tl.program_id(1)
    m_id_stride = tl.num_programs(1)

    GROUP_SIZE_CTA: tl.constexpr = GROUP_SIZE * num_groups_per_cta
    g_offs = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE_CTA)
    g_offs = tl.max_contiguous(tl.multiple_of(g_offs, GROUP_SIZE), GROUP_SIZE)
    gs_offs = group_id + tl.arange(0, num_groups_per_cta)
    rfp8_max = 1 / fp8_max

    m_id = m_id_start
    a_ptrs = a_ptr + m_id * stride_am + g_offs * stride_ak
    o_ptrs = out_ptr + m_id * stride_om + g_offs * stride_ok
    s_ptr = scale_ptr + m_id * stride_sm + gs_offs * stride_sg
    if K % GROUP_SIZE_CTA == 0:
        mask_n = True
        mask_s = True
        mask_o = True
    else:
        mask_n = g_offs < K
        mask_o = g_offs < K
        mask_s = gs_offs < tl.cdiv(K, GROUP_SIZE)

    for m_id in tl.range(m_id_start, M_out, m_id_stride, num_stages=NUM_STAGES):
        a = tl.load(a_ptrs, mask=mask_n & (m_id < M), other=0)
        a = a.reshape(num_groups_per_cta, GROUP_SIZE)
        a_max = tl.max(tl.abs(a), axis=1)
        a_max = tl.maximum(a_max, 1e-6).to(tl.float32)
        if ROUND_SCALE == 1:
            scale = fast_round_scale(a_max, rfp8_max)
            rscale = 1 / scale
        else:
            scale = a_max * rfp8_max
            rscale = fp8_max / a_max  # triton does not support rcp
        out = a.to(tl.float32) * rscale[:, None]

        out = tl.clamp(out, fp8_min, fp8_max)
        out = out.to(out_ptr.dtype.element_ty)
        out = out.reshape(GROUP_SIZE * num_groups_per_cta)
        tl.store(o_ptrs, out, mask=mask_o)
        tl.store(s_ptr, scale, mask=mask_s)

        a_ptrs += m_id_stride * stride_am
        o_ptrs += m_id_stride * stride_om
        s_ptr += m_id_stride * stride_sm


def _quant_fp8_launcher(A: Tensor, group_size: int, out: Tensor, scales: Tensor, scale_fmt: Optional[str] = None):
    """Quant online."""
    assert scale_fmt in (None, 'ue8m0')
    round_scale = 1 if scale_fmt == 'ue8m0' else 0
    M, K = A.shape
    M_out = out.size(0)

    dtype = out.dtype
    finfo = torch.finfo(dtype)
    fmin = finfo.min
    fmax = finfo.max

    num_warps = 2
    # every cp/ldg instruct can load 128bit=16byte data
    # each warp can read 512 byte data
    elem_size = A.element_size()
    num_groups_per_warp = 512 // (group_size * elem_size)
    num_groups_per_cta = num_groups_per_warp * num_warps
    grid_size0 = triton.cdiv(K, group_size * num_groups_per_cta)
    props = get_device_props(A.device.index)
    num_sm = props['multi_processor_count']
    warps_per_sm = props['warps_per_sm']
    blocks_per_sm = props['blocks_per_sm']
    max_ctas = num_sm * min(blocks_per_sm, warps_per_sm // num_warps)
    grid_size1 = min(M_out, max_ctas // grid_size0)
    assert grid_size1 < 65536
    num_stages = min(4, max(1, triton.cdiv(M_out, grid_size1)))
    grid = (grid_size0, grid_size1)
    _quant_fp8_kernel[grid](
        A,
        out,
        scales,
        M,
        M_out,
        K,
        num_groups_per_cta=num_groups_per_cta,
        fp8_min=fmin,
        fp8_max=fmax,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_om=out.stride(0),
        stride_ok=out.stride(1),
        stride_sm=scales.stride(0),
        stride_sg=scales.stride(1),
        ROUND_SCALE=round_scale,
        GROUP_SIZE=group_size,
        NUM_STAGES=num_stages,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out, scales


def quant_fp8(A: Tensor,
              group_size: int,
              dtype: torch.dtype = torch.float8_e4m3fn,
              trans_scale: bool = False,
              scale_fmt: Optional[str] = None):
    """Quant fp8."""
    assert A.dim() == 2
    M, K = A.shape
    assert K % group_size == 0
    num_groups = K // group_size
    out = torch.empty_like(A, dtype=dtype)
    if trans_scale:
        scales = A.new_empty(num_groups, M, dtype=torch.float32).T
    else:
        scales = A.new_empty(M, num_groups, dtype=torch.float32)
    return _quant_fp8_launcher(A, group_size, out, scales, scale_fmt=scale_fmt)


def quant_fp8_tma(A: Tensor,
                  group_size: int,
                  dtype: torch.dtype = torch.float8_e4m3fn,
                  scale_fmt: Optional[str] = None):
    """Quant fp8 tma."""
    from lmdeploy.pytorch.third_party.deep_gemm import ceil_div, get_m_alignment_for_contiguous_layout
    assert A.dim() == 2
    M, K = A.shape
    assert K % group_size == 0
    num_groups = K // group_size
    alignment = get_m_alignment_for_contiguous_layout()
    aligned_M = ceil_div(M, alignment) * alignment
    out = A.new_empty(aligned_M, K, dtype=dtype)
    scales = A.new_empty(num_groups, aligned_M, dtype=torch.float32).T
    return _quant_fp8_launcher(A, group_size, out, scales, scale_fmt=scale_fmt)


def _gemm_fp8_tma_pre_hook(nargs):
    BLOCK_M = nargs['BLOCK_M']
    BLOCK_N = nargs['BLOCK_N']
    BLOCK_K = nargs['BLOCK_K']
    nargs['desc_a'].block_shape = (BLOCK_M, BLOCK_K)
    nargs['desc_b'].block_shape = (BLOCK_N, BLOCK_K)


@triton.autotune(configs=[
    triton.Config({
        'BLOCK_M': 128,
        'BLOCK_N': 128,
    }, num_stages=3, num_warps=8, pre_hook=_gemm_fp8_tma_pre_hook),
    triton.Config({
        'BLOCK_M': 128,
        'BLOCK_N': 64,
    }, num_stages=3, num_warps=4, pre_hook=_gemm_fp8_tma_pre_hook)
],
                 key=['N', 'K'])
@triton.jit
def _gemm_fp8_tma_kernel(
    desc_a,
    a_scale_ptr,
    desc_b,
    b_scale_ptr,
    C,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_ak: tl.constexpr,
    group_bk: tl.constexpr,
    group_bn: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_ask,
    stride_bsk: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_cm,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Gemm fp8 kernel."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M

    offs_bsn = pid_n * BLOCK_N // group_bn
    as_ptrs = a_scale_ptr + offs_am * stride_asm
    bs_ptrs = b_scale_ptr + offs_bsn * stride_bsn

    acc_scale = tl.load(as_ptrs) * tl.load(bs_ptrs)
    acc_ratio = 1 / acc_scale
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    off_k = 0
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # load scales
        k_start = (k + 1) * BLOCK_K
        offs_ksa = k_start // group_ak
        offs_ksb = k_start // group_bk
        a_scale = tl.load(as_ptrs + offs_ksa * stride_ask, mask=k_start < K, other=1.0)
        b_scale = tl.load(bs_ptrs + offs_ksb * stride_bsk, mask=k_start < K, other=1.0)

        # load ab
        a = desc_a.load([off_m, off_k])
        b = desc_b.load([off_n, off_k]).T

        # mma
        accumulator = tl.dot(a, b, acc=accumulator * acc_ratio[:, None])

        # update scales and ratio
        new_acc_scale = a_scale * b_scale
        acc_ratio = acc_scale / new_acc_scale
        acc_scale = new_acc_scale

        off_k += BLOCK_K
    c = accumulator * (acc_ratio * acc_scale)[:, None]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(configs=[
    triton.Config({
        'BLOCK_M': 64,
        'BLOCK_N': 128,
    }, num_stages=3, num_warps=4),
    triton.Config({
        'BLOCK_M': 128,
        'BLOCK_N': 64,
    }, num_stages=3, num_warps=4)
],
                 key=['N', 'K'])
@triton.jit
def _gemm_fp8_kernel(
    A,
    a_scale_ptr,
    B,
    b_scale_ptr,
    C,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_ak: tl.constexpr,
    group_bk: tl.constexpr,
    group_bn: tl.constexpr,
    stride_am,
    stride_ak: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_ask,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bsk: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_cm,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Gemm fp8 kernel."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    offs_bsn = pid_n * BLOCK_N // group_bn
    as_ptrs = a_scale_ptr + offs_am * stride_asm
    bs_ptrs = b_scale_ptr + offs_bsn * stride_bsn

    acc_scale = tl.load(as_ptrs) * tl.load(bs_ptrs)
    acc_ratio = 1 / acc_scale
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # load scales
        k_start = (k + 1) * BLOCK_K
        offs_ksa = k_start // group_ak
        offs_ksb = k_start // group_bk
        a_scale = tl.load(as_ptrs + offs_ksa * stride_ask, mask=k_start < K, other=1.0)
        b_scale = tl.load(bs_ptrs + offs_ksb * stride_bsk, mask=k_start < K, other=1.0)

        # load ab
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)

        # mma
        accumulator = tl.dot(a, b, acc=accumulator * acc_ratio[:, None])

        # update scales and ratio
        new_acc_scale = a_scale * b_scale
        acc_ratio = acc_scale / new_acc_scale
        acc_scale = new_acc_scale

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = accumulator * (acc_ratio * acc_scale)[:, None]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def blocked_gemm_fp8(A: Tensor,
                     A_scale: Tensor,
                     B: Tensor,
                     B_scale: torch.Tensor,
                     out_dtype: torch.dtype = torch.float16):
    """Gemm fp8."""

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )

    assert A.dim() == 2
    assert A_scale.dim() == 2
    assert B.dim() == 2
    assert B_scale.dim() == 2

    M, K = A.shape
    _, N = B.shape

    group_ak = triton.cdiv(K, A_scale.size(1))
    group_bk = triton.cdiv(K, B_scale.size(0))
    group_bn = triton.cdiv(N, B_scale.size(1))

    C = A.new_empty(M, N, dtype=out_dtype)

    BLOCK_K = max(group_ak, group_bk)

    from .utils import supports_tma

    run_tma = supports_tma()
    run_tma = run_tma and A.is_contiguous() and B.T.is_contiguous()

    # run_tma = False
    if run_tma:
        from .utils import TensorDescriptor

        dummy_block = (1, 1)
        desc_a = TensorDescriptor.from_tensor(A, block_shape=dummy_block)
        desc_b = TensorDescriptor.from_tensor(B.T, block_shape=dummy_block)

        def _grid_tma(META):
            """Grid tma."""
            BLOCK_M = META['BLOCK_M']
            BLOCK_N = META['BLOCK_N']
            return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

        _gemm_fp8_tma_kernel[_grid_tma](
            desc_a,
            A_scale,
            desc_b,
            B_scale,
            C,
            M=M,
            N=N,
            K=K,
            group_ak=group_ak,
            group_bk=group_bk,
            group_bn=group_bn,
            stride_asm=A_scale.stride(0),
            stride_ask=A_scale.stride(1),
            stride_bsk=B_scale.stride(0),
            stride_bsn=B_scale.stride(1),
            stride_cm=C.stride(0),
            stride_cn=C.stride(1),
            BLOCK_K=BLOCK_K,
            GROUP_M=8,
        )
    else:
        _gemm_fp8_kernel[grid](
            A,
            A_scale,
            B,
            B_scale,
            C,
            M=M,
            N=N,
            K=K,
            group_ak=group_ak,
            group_bk=group_bk,
            group_bn=group_bn,
            stride_am=A.stride(0),
            stride_ak=A.stride(1),
            stride_asm=A_scale.stride(0),
            stride_ask=A_scale.stride(1),
            stride_bk=B.stride(0),
            stride_bn=B.stride(1),
            stride_bsk=B_scale.stride(0),
            stride_bsn=B_scale.stride(1),
            stride_cm=C.stride(0),
            stride_cn=C.stride(1),
            BLOCK_K=BLOCK_K,
            GROUP_M=8,
        )

    return C


def deep_gemm_fp8(A: Tensor,
                  A_scale: Tensor,
                  B: Tensor,
                  B_scale: torch.Tensor,
                  out_dtype: torch.dtype = torch.bfloat16):
    """Deepgemm fp8."""
    from lmdeploy.pytorch.third_party.deep_gemm import fp8_gemm_nt
    M, _ = A.shape
    N, _ = B.shape
    assert out_dtype == torch.bfloat16, 'DeepGemm requires bf16 output.'
    C = A.new_empty(M, N, dtype=out_dtype)
    fp8_gemm_nt((A, A_scale), (B, B_scale), C, None)
    return C

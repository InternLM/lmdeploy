# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _quant_fp8_kernel(
    a_ptr,
    out_ptr,
    scale_ptr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    stride_am,
    stride_ak: tl.constexpr,
    stride_om,
    stride_ok: tl.constexpr,
    stride_sm,
    stride_sg: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """quant fp8 kernel."""
    group_id = tl.program_id(0)
    m_id = tl.program_id(1)

    g_offs = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

    a_ptrs = a_ptr + m_id * stride_am + g_offs * stride_ak
    o_ptrs = out_ptr + m_id * stride_om + g_offs * stride_ok
    s_ptr = scale_ptr + m_id * stride_sm + group_id * stride_sg

    rfp8_max = 1 / fp8_max

    a = tl.load(a_ptrs).to(tl.float32)
    scale = tl.max(tl.abs(a)) * rfp8_max
    out = a / scale

    out = tl.clamp(out, fp8_min, fp8_max)
    out = out.to(out_ptr.dtype.element_ty)

    tl.store(o_ptrs, out)
    tl.store(s_ptr, scale)


def quant_fp8(A: Tensor,
              group_size: int,
              dtype: torch.dtype = torch.float8_e4m3fn):
    """quant online."""
    assert A.dim() == 2
    M, K = A.shape
    assert K % group_size == 0
    num_groups = K // group_size

    finfo = torch.finfo(dtype)
    fmin = finfo.min
    fmax = finfo.max

    out = torch.empty_like(A, dtype=dtype)
    scales = A.new_empty(M, num_groups, dtype=torch.float32)
    grid = (num_groups, M)
    num_warps = 4
    num_stages = 1
    _quant_fp8_kernel[grid](
        A,
        out,
        scales,
        fp8_min=fmin,
        fp8_max=fmax,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_om=out.stride(0),
        stride_ok=out.stride(1),
        stride_sm=scales.stride(0),
        stride_sg=scales.stride(1),
        GROUP_SIZE=group_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out, scales


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
                 key=['N', 'K'],
                 warmup=5,
                 rep=10)
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
    stride_asm,
    stride_ask: tl.constexpr,
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
    """gemm fp8 kernel."""
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
        a_scale = tl.load(as_ptrs + offs_ksa * stride_ask,
                          mask=k_start < K,
                          other=1.0)
        b_scale = tl.load(bs_ptrs + offs_ksb * stride_bsk,
                          mask=k_start < K,
                          other=1.0)

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
    """gemm fp8."""

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) *
                triton.cdiv(N, META['BLOCK_N']), )

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

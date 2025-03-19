# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

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


def quant_fp8(A: Tensor, group_size: int, dtype: torch.dtype = torch.float8_e4m3fn):
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


# adapted from https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/jit_kernels/utils.py#L46
def get_tma_aligned_size(x: int, element_size: int) -> int:
    """Global memory address of TMA must be 16-byte aligned. Since we use
    column-major layout for the LHS scaling tensor, the M-axis of the LHS
    scaling tensor needs to be padded to a multiple of 16 bytes.

    Arguments:
        x: original M-axis shape of the LHS scaling tensor.
        element_size: element size of the LHS scaling tensor.

    Returns:
        M-axis shape of the LHS scaling tensor after padding.
    """
    tma_alignment_bytes = 16
    assert tma_alignment_bytes % element_size == 0
    alignment = tma_alignment_bytes // element_size
    return triton.cdiv(x, alignment) * alignment


@triton.jit
def _quant_fp8_tma_kernel(
    a_ptr,
    out_ptr,
    scale_ptr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    stride_am,
    stride_ak: tl.constexpr,
    stride_om,
    stride_ok: tl.constexpr,
    stride_sg,
    stride_sm: tl.constexpr,
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


def quant_fp8_tma(A: Tensor, group_size: int, dtype: torch.dtype = torch.float8_e4m3fn):
    """quant online."""
    assert A.dim() == 2
    M, K = A.shape
    assert K % group_size == 0
    num_groups = K // group_size

    finfo = torch.finfo(dtype)
    fmin = finfo.min
    fmax = finfo.max

    out = torch.empty_like(A, dtype=dtype)
    aligned_M = get_tma_aligned_size(M, torch.float32.itemsize)
    scales = A.new_empty(num_groups, aligned_M, dtype=torch.float32)
    grid = (num_groups, M)
    num_warps = 4
    num_stages = 1
    _quant_fp8_tma_kernel[grid](
        A,
        out,
        scales,
        fp8_min=fmin,
        fp8_max=fmax,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_om=out.stride(0),
        stride_ok=out.stride(1),
        stride_sg=scales.stride(0),
        stride_sm=scales.stride(1),
        GROUP_SIZE=group_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out, scales.transpose(0, 1)


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
    """gemm fp8."""

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


def get_col_major_tma_aligned_tensor(x: torch.Tensor) -> torch.Tensor:
    """Returns TMA-aligned transposed format of the input tensor.
    `torch.transpose` will be called if necessary. If the input tensor is
    already column-major layout and 16-byte aligned along the M axis (thus
    meets the requirement of LHS scaling tensor in DeepGEMM), this function
    will do nothing.

    Arguments:
        x: usually the LHS scaling tensor in GEMM.

    Returns:
        The LHS scaling tensor of TMA-aligned transposed format.
    """
    # NOTES: for the extreme performance, you may rewrite/fuse this function in CUDA
    assert x.dim() in (2, 3)
    remove_dim = False
    if x.dim() == 2:
        x, remove_dim = x.unsqueeze(0), True

    b, m, n = x.shape
    aligned_m = get_tma_aligned_size(m, x.element_size())

    # The last kernel gives a column-major TMA aligned layout
    # NOTE we modified the stride(0) == aligned_m from stride(0) == aligned_m * n
    if x.stride(0) == aligned_m and x.stride(1) == 1 and x.stride(2) == aligned_m:
        return x.squeeze(0) if remove_dim else x

    # Normal layout requires transposing
    aligned_x = torch.transpose(torch.empty((b, n, aligned_m), device=x.device, dtype=x.dtype), 1, 2)
    aligned_x[:, :m, :] = x
    aligned_x = aligned_x[:, :m, :]
    return aligned_x.squeeze(0) if remove_dim else aligned_x


def gemm_fp8_fp8_bf16_nt(lhs: Tuple[torch.Tensor, torch.Tensor], rhs: Tuple[torch.Tensor, torch.Tensor],
                         out: torch.Tensor) -> None:
    """Do a normal GEMM with FP8 inputs and BF16 output, with 1x128 LHS scaling
    and 128x128 RHS scaling. LHS, RHS, RHS scaling factors, and output tensors
    must be in contiguous format. RHS and RHS scaling factors are required to
    be transposed. The LHS scaling tensor requires TMA-aligned transposed
    format, if your input does not match the requirement, this function will do
    a transposing with a set of slow PyTorch operations.

    Arguments:
        lhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[m, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[m, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[n, k]`.
             the second element is an FP32 128x128 scaling tensor for RHS of shape `[⌈n / 128⌉, ⌈k / 128⌉]`.
        out: the BF16 output tensor of shape `[m, n]`, representing the result.
    """
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    m, k = lhs.shape
    n, k_ = rhs.shape
    m_, n_ = out.shape

    assert n % 64 == 0 and k % 128 == 0

    # Type and shape checks
    assert m == m_ and n == n_ and k == k_
    assert n > 0 and k > 0
    # NOTE This is modified to skip shape[0] check
    assert lhs_scales.shape[-1] == (k + 127) // 128
    assert rhs_scales.shape == ((n + 127) // 128, (k + 127) // 128)
    assert lhs.dtype == torch.float8_e4m3fn and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.float8_e4m3fn and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert lhs.is_contiguous() and rhs.is_contiguous() and out.is_contiguous()

    # LHS scales must be transposed for TMA load, but not for RHS scales
    # NOTES: `get_tma_aligned_lhs_scales` may launch a kernel if not processed by previous kernels
    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert rhs_scales.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    # Auto-tuning with compilation
    from deep_gemm.jit_kernels.gemm import get_best_configs, get_num_sms, includes, jit_tuner, template
    num_sms = get_num_sms()
    block_m, block_n, num_stages, num_tma_multicast, smem_size = get_best_configs(m, n, k, 1, num_sms)
    args = (lhs, lhs_scales, rhs, rhs_scales, out, m, torch.cuda.current_stream(), num_sms, smem_size)
    runtime = jit_tuner.compile_and_tune(name='gemm_fp8_fp8_bf16_nt',
                                         keys={
                                             'N': n,
                                             'K': k,
                                             'BLOCK_M': block_m,
                                             'BLOCK_N': block_n,
                                             'NUM_STAGES': num_stages,
                                             'NUM_TMA_MULTICAST': num_tma_multicast
                                         },
                                         space=(),
                                         includes=includes,
                                         arg_defs=(('lhs', torch.float8_e4m3fn), ('lhs_scales', torch.float),
                                                   ('rhs', torch.float8_e4m3fn), ('rhs_scales', torch.float),
                                                   ('out', torch.bfloat16), ('m', int), ('stream', torch.cuda.Stream),
                                                   ('num_sms', int), ('smem_size', int)),
                                         template=template,
                                         args=args)

    # Run the kernel
    runtime(*args)


def deep_gemm_fp8(A: Tensor,
                  A_scale: Tensor,
                  B: Tensor,
                  B_scale: torch.Tensor,
                  out_dtype: torch.dtype = torch.bfloat16):
    """deepgemm fp8."""
    M, K = A.shape
    N, _ = B.shape
    assert out_dtype == torch.bfloat16, 'DeepGemm requires bf16 output.'
    C = A.new_empty(M, N, dtype=out_dtype)
    gemm_fp8_fp8_bf16_nt((A, A_scale), (B, B_scale), C)
    return C

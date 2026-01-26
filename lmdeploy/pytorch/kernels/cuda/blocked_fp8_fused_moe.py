# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
from typing import Callable

import torch
import triton
import triton.language as tl

from .activation import silu_and_mul
from .blocked_gemm_fp8 import quant_fp8
from .fused_moe import _get_sorted_idx, _make_intermediate, _renormalize, moe_reduce


def get_cuda_autotune_config():
    return [
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
        }, num_stages=3, num_warps=4),
    ]


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['N', 'K', 'M_NP2'],
)
@triton.jit
def fused_moe_blocked_f8_kernel(
    A,
    A_scale,
    B,
    B_scale,
    bias,
    C,
    SortedIdx,
    ExpStart,
    ExpEnd,
    N: tl.constexpr,
    K: tl.constexpr,
    group_ak: tl.constexpr,
    group_bk: tl.constexpr,
    group_bn: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_asm,
    stride_ask: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bse: tl.constexpr,
    stride_bsk: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_bie: tl.constexpr,
    stride_bin: tl.constexpr,
    stride_cm,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    M_NP2: tl.constexpr,
    top_k: tl.constexpr,
    expert_offset: tl.constexpr,
    reindex_a: tl.constexpr,
    reindex_c: tl.constexpr,
):
    """Fused moe kernel."""
    exp_id = tl.program_id(1)
    pid = tl.program_id(0)

    exp_start = tl.load(ExpStart + exp_id + expert_offset)
    exp_end = tl.load(ExpEnd + exp_id + expert_offset)
    M = exp_end - exp_start
    if M <= 0:
        return

    num_pid_m = tl.cdiv(M_NP2, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if GROUP_SIZE_M == 1:
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m * BLOCK_SIZE_M >= M or pid_n * BLOCK_SIZE_N >= N:
        return

    offs_sid = exp_start + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_sid = offs_sid < exp_end
    sid = tl.load(SortedIdx + offs_sid, mask=mask_sid, other=0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if reindex_a:
        offs_am = sid // top_k
    else:
        offs_am = offs_sid
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    # deepseek has 160 experts, exp index would overflow int32
    exp_id = exp_id.to(tl.int64)
    exp_off = stride_be * exp_id
    b_ptrs = B + exp_off + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    offs_bsn = pid_n * BLOCK_SIZE_N // group_bn
    as_ptrs = A_scale + offs_am * stride_asm
    bs_ptrs = B_scale + stride_bse * exp_id + offs_bsn * stride_bsn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # initialize acc_ratio and acc_scale
    a_scale = tl.load(as_ptrs, mask=mask_sid, other=1.0)
    b_scale = tl.load(bs_ptrs)
    acc_scale0 = a_scale * b_scale

    k_start = BLOCK_SIZE_K
    offs_ksa = k_start // group_ak
    offs_ksb = k_start // group_bk
    a_scale = tl.load(as_ptrs + offs_ksa * stride_ask, mask=mask_sid & (k_start < K), other=1.0)
    b_scale = tl.load(bs_ptrs + offs_ksb * stride_bsk, mask=k_start < K, other=1.0)
    acc_scale1 = tl.maximum(a_scale * b_scale, 1e-12)
    acc_ratio = acc_scale0 / acc_scale1
    acc_scale = acc_scale1

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # load scales
        k_start = (k + 2) * BLOCK_SIZE_K
        offs_ksa = k_start // group_ak
        offs_ksb = k_start // group_bk
        a_scale = tl.load(as_ptrs + offs_ksa * stride_ask, mask=mask_sid & (k_start < K), other=1.0)
        b_scale = tl.load(bs_ptrs + offs_ksb * stride_bsk, mask=k_start < K, other=1.0)

        # load ab
        a = tl.load(a_ptrs, mask=mask_sid[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # mma
        accumulator = tl.dot(a, b, acc=accumulator)
        accumulator *= acc_ratio[:, None]

        # update scales and ratio
        new_acc_scale = tl.maximum(a_scale * b_scale, 1e-12)
        acc_ratio = acc_scale / new_acc_scale
        acc_scale = new_acc_scale

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator * (acc_ratio * acc_scale)[:, None]

    if bias is not None:
        bias_ptrs = bias + exp_id * stride_bie + offs_bn * stride_bin
        bias_val = tl.load(bias_ptrs).to(accumulator.dtype)
        c += bias_val[None]

    c = c.to(C.dtype.element_ty)

    if reindex_c:
        offs_cm = sid
    else:
        offs_cm = offs_sid
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_bn[None, :]
    tl.store(c_ptrs, c, mask=mask_sid[:, None])


def fused_moe_blocked_fp8_kernel_launcher(
    A: torch.Tensor,
    A_scale: torch.Tensor,
    B: torch.Tensor,
    B_scale: torch.Tensor,
    C: torch.Tensor,
    sorted_idx: torch.Tensor,
    exp_start: torch.Tensor,
    exp_end: torch.Tensor,
    bias: torch.Tensor = None,
    top_k: int = 1,
    num_tokens: int = None,
    expert_offset: int = 0,
    reindex_a: bool = True,
    reindex_c: bool = True,
):
    """Fused moe kernel launcher."""

    if num_tokens is None:
        num_tokens = A.size(0)
    M_NP2 = triton.next_power_of_2(num_tokens)
    M_NP2 = max(64, M_NP2)
    E, N, K = B.shape

    assert A.dim() == 2
    assert A_scale.dim() == 2
    assert B.dim() == 3
    assert B_scale.dim() == 3

    assert K % A_scale.size(1) == 0
    assert K % B_scale.size(2) == 0
    assert N % B_scale.size(1) == 0

    group_ak = K // A_scale.size(1)
    group_bk = K // B_scale.size(2)
    group_bn = N // B_scale.size(1)

    def _grid_fn(META):
        grid = (triton.cdiv(M_NP2, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), E)
        return grid

    A = A.flatten(0, -2)
    C = C.flatten(0, -2)
    enable_bias = bias is not None

    BLOCK_SIZE_K = group_bk
    GROUP_SIZE_M = 1
    grid = _grid_fn
    fused_moe_blocked_f8_kernel[grid](
        A,
        A_scale,
        B,
        B_scale,
        bias,
        C,
        sorted_idx,
        exp_start,
        exp_end,
        N=N,
        K=K,
        group_ak=group_ak,
        group_bk=group_bk,
        group_bn=group_bn,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_asm=A_scale.stride(0),
        stride_ask=A_scale.stride(1),
        stride_be=B.stride(0),
        stride_bn=B.stride(1),
        stride_bk=B.stride(2),
        stride_bse=B_scale.stride(0),
        stride_bsn=B_scale.stride(1),
        stride_bsk=B_scale.stride(2),
        stride_cm=C.stride(0),
        stride_cn=C.stride(1),
        stride_bie=bias.stride(0) if enable_bias else 0,
        stride_bin=bias.stride(1) if enable_bias else 0,
        top_k=top_k,
        expert_offset=expert_offset,
        reindex_a=reindex_a,
        reindex_c=reindex_c,
        M_NP2=M_NP2,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )


def fused_moe_blocked_fp8(input: torch.Tensor,
                          input_scale: torch.Tensor,
                          w1: torch.Tensor,
                          w1_scale: torch.Tensor,
                          w2: torch.Tensor,
                          w2_scale: torch.Tensor,
                          topk_weights: torch.Tensor,
                          topk_ids: torch.Tensor,
                          topk: int,
                          w1_bias: torch.Tensor = None,
                          w2_bias: torch.Tensor = None,
                          out_dtype: torch.dtype = torch.float16,
                          expert_offset: int = 0,
                          num_experts: int = None,
                          renormalize: bool = False,
                          act_func: Callable = None) -> torch.Tensor:
    """Fused moe."""
    device = input.device
    M = input.size(0)
    E, N, _ = w1.shape
    if num_experts is None:
        num_experts = E
    full_exp = num_experts == E
    group_size = input.size(-1) // input_scale.size(-1)

    topk_weights = _renormalize(topk_weights, renormalize)
    sorted_idx, exp_start, exp_end = _get_sorted_idx(topk_ids, num_experts)

    intermediate_cache1 = _make_intermediate((M, topk, N), dtype=out_dtype, device=device, zeros=not full_exp)
    # gate and up
    fused_moe_blocked_fp8_kernel_launcher(
        input,
        input_scale,
        w1,
        w1_scale,
        intermediate_cache1,
        sorted_idx=sorted_idx,
        exp_start=exp_start,
        exp_end=exp_end,
        bias=w1_bias,
        top_k=topk,
        num_tokens=M,
        expert_offset=expert_offset,
        reindex_a=True,
        reindex_c=False,
    )

    # activate
    intermediate_cache1 = intermediate_cache1.flatten(0, -2)
    if act_func is None:
        gate_cache = silu_and_mul(intermediate_cache1)
    else:
        gate_cache = act_func(intermediate_cache1)
    del intermediate_cache1
    gate_cache, gate_scale = quant_fp8(gate_cache, group_size, dtype=input.dtype)

    intermediate_cache2 = _make_intermediate((M, topk, w2.shape[1]), dtype=out_dtype, device=device, zeros=not full_exp)
    # down
    fused_moe_blocked_fp8_kernel_launcher(
        gate_cache,
        gate_scale,
        w2,
        w2_scale,
        intermediate_cache2,
        sorted_idx=sorted_idx,
        exp_start=exp_start,
        exp_end=exp_end,
        bias=w2_bias,
        top_k=1,
        num_tokens=M,
        expert_offset=expert_offset,
        reindex_a=False,
        reindex_c=True,
    )

    ret = moe_reduce(intermediate_cache2, topk_weights)
    return ret

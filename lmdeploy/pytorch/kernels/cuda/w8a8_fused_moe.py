# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
import torch
import triton
import triton.language as tl

from .activation import silu_and_mul
from .fused_moe import _get_sorted_idx, _make_intermediate, _renormalize, moe_reduce
from .w8a8_triton_kernels import per_token_quant_int8


def get_cuda_autotune_config():
    return [
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 1,
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 256,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 1,
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 1,
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 128,
            'GROUP_SIZE_M': 1,
        },
                      num_stages=3,
                      num_warps=8),
    ]


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['N', 'K', 'M_NP2'],
)
@triton.jit
def fused_moe_w8a8_kernel(
    A,
    A_scale,
    B,
    B_scale,
    C,
    SortedIdx,
    ExpStart,
    ExpEnd,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bse: tl.constexpr,
    stride_cm: tl.constexpr,
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
    ACCUMULATOR_DTYPE: tl.constexpr,
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
    as_ptrs = A_scale + offs_am
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    # deepseek has 160 experts, exp index would overflow int32
    exp_id = exp_id.to(tl.int64)
    exp_off = stride_be * exp_id
    b_ptrs = B + exp_off + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    bs_ptrs = B_scale + exp_id * stride_bse + offs_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUMULATOR_DTYPE)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=mask_sid[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator, out_dtype=ACCUMULATOR_DTYPE)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    ascale = tl.load(as_ptrs, mask=mask_sid)
    bscale = tl.load(bs_ptrs)
    c = accumulator.to(ascale.dtype)
    c = c * ascale[:, None] * bscale[None, :]

    c = c.to(C.dtype.element_ty)

    if reindex_c:
        offs_cm = sid
    else:
        offs_cm = offs_sid
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_bn[None, :]
    tl.store(c_ptrs, c, mask=mask_sid[:, None])


def fused_moe_w8a8_kernel_launcher(
    A: torch.Tensor,
    A_scale: torch.Tensor,
    B: torch.Tensor,
    B_scale: torch.Tensor,
    C: torch.Tensor,
    sorted_idx: torch.Tensor,
    exp_start: torch.Tensor,
    exp_end: torch.Tensor,
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

    assert A_scale.is_contiguous()
    assert B_scale.is_contiguous()
    accumulator_dtype = tl.float32 if A.is_floating_point() else tl.int32

    def _grid_fn(META):
        grid = (triton.cdiv(M_NP2, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), E)
        return grid

    A = A.flatten(0, -2)
    C = C.flatten(0, -2)

    grid = _grid_fn
    fused_moe_w8a8_kernel[grid](
        A,
        A_scale,
        B,
        B_scale,
        C,
        sorted_idx,
        exp_start,
        exp_end,
        N=N,
        K=K,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_be=B.stride(0),
        stride_bn=B.stride(1),
        stride_bk=B.stride(2),
        stride_bse=B_scale.stride(0),
        stride_cm=C.stride(0),
        stride_cn=C.stride(1),
        top_k=top_k,
        expert_offset=expert_offset,
        reindex_a=reindex_a,
        reindex_c=reindex_c,
        M_NP2=M_NP2,
        ACCUMULATOR_DTYPE=accumulator_dtype,
    )


def fused_moe_w8a8(input: torch.Tensor,
                   input_scale: torch.Tensor,
                   w1: torch.Tensor,
                   w1_scale: torch.Tensor,
                   w2: torch.Tensor,
                   w2_scale: torch.Tensor,
                   topk_weights: torch.Tensor,
                   topk_ids: torch.Tensor,
                   topk: int,
                   out_dtype: torch.dtype = torch.float16,
                   quant_dtype: torch.dtype = torch.int8,
                   expert_offset: int = 0,
                   num_experts: int = None,
                   renormalize: bool = False) -> torch.Tensor:
    """Fused moe."""
    device = input.device
    M = input.size(0)
    E, N, _ = w1.shape
    if num_experts is None:
        num_experts = E
    full_exp = num_experts == E

    topk_weights = _renormalize(topk_weights, renormalize)
    sorted_idx, exp_start, exp_end = _get_sorted_idx(topk_ids, num_experts)

    intermediate_cache1 = _make_intermediate((M, topk, N), dtype=out_dtype, device=device, zeros=not full_exp)
    # gate and up
    fused_moe_w8a8_kernel_launcher(
        input,
        input_scale,
        w1,
        w1_scale,
        intermediate_cache1,
        sorted_idx=sorted_idx,
        exp_start=exp_start,
        exp_end=exp_end,
        top_k=topk,
        num_tokens=M,
        expert_offset=expert_offset,
        reindex_a=True,
        reindex_c=False,
    )

    # activate
    unflat_size = intermediate_cache1.shape[:-1]
    intermediate_cache1 = intermediate_cache1.flatten(0, -2)
    gate_cache = silu_and_mul(intermediate_cache1)
    del intermediate_cache1
    gate_cache = gate_cache.unflatten(0, unflat_size)
    gate_cache, gate_scale = per_token_quant_int8(gate_cache, 1e-7, quant_dtype=quant_dtype)

    intermediate_cache2 = _make_intermediate((M, topk, w2.shape[1]), dtype=out_dtype, device=device, zeros=not full_exp)
    # down
    fused_moe_w8a8_kernel_launcher(
        gate_cache,
        gate_scale,
        w2,
        w2_scale,
        intermediate_cache2,
        sorted_idx=sorted_idx,
        exp_start=exp_start,
        exp_end=exp_end,
        top_k=1,
        num_tokens=M,
        expert_offset=expert_offset,
        reindex_a=False,
        reindex_c=True,
    )

    ret = moe_reduce(intermediate_cache2, topk_weights)
    return ret

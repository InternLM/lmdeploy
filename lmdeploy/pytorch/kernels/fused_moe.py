# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .triton_utils import get_kernel_meta, wrap_jit_func


def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 64,
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 32,
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32,
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
            },
            num_stages=5,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 32,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32,
            },
            num_stages=5,
            num_warps=2),
    ]


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['N', 'K'],
)
@wrap_jit_func(type_hint=dict(
    A=torch.Tensor,
    B=torch.Tensor,
    C=torch.Tensor,
    SortedIdx=torch.Tensor,
    ExpStart=torch.Tensor,
    ExpEnd=torch.Tensor,
    Weights=torch.Tensor,
    N=int,
    K=int,
    stride_am=int,
    stride_ak=int,
    stride_be=int,
    stride_bn=int,
    stride_bk=int,
    stride_cm=int,
    stride_cn=int,
    BLOCK_SIZE_M=torch.int32,
    BLOCK_SIZE_N=torch.int32,
    BLOCK_SIZE_K=torch.int32,
    GROUP_SIZE_M=torch.int32,
    ENABLE_WEIGHTS=bool,
    top_k=torch.int32,
    expert_offset=torch.int32,
    reindex_a=bool,
    reindex_c=bool,
))
@triton.jit
def fused_moe_kernel(
    A,
    B,
    C,
    SortedIdx,
    ExpStart,
    ExpEnd,
    Weights,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: int,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_cm: int,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ENABLE_WEIGHTS: tl.constexpr,
    top_k: tl.constexpr,
    expert_offset: tl.constexpr,
    reindex_a: tl.constexpr,
    reindex_c: tl.constexpr,
):
    """fused moe kernel."""
    exp_id = tl.program_id(0)
    pid = tl.program_id(1)

    exp_start = tl.load(ExpStart + exp_id + expert_offset)
    exp_end = tl.load(ExpEnd + exp_id + expert_offset)
    M = exp_end - exp_start
    if M <= 0:
        return

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m * BLOCK_SIZE_M >= M:
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

    # deepseek has 160 experts, exp index would overflow int32
    exp_off = tl.full((1, ), stride_be, dtype=tl.int64) * exp_id
    b_ptrs = B + exp_off + (offs_k[:, None] * stride_bk +
                            offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs,
                    mask=mask_sid[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if ENABLE_WEIGHTS:
        weight = tl.load(Weights + sid, mask=mask_sid)
        accumulator = accumulator * weight[:, None].to(accumulator.dtype)

    c = accumulator.to(A.dtype.element_ty)

    if reindex_c:
        offs_cm = sid
    else:
        offs_cm = exp_start + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = mask_sid[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def fused_moe_kernel_launcher(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    sorted_idx: torch.Tensor,
    exp_start: torch.Tensor,
    exp_end: torch.Tensor,
    weights: torch.Tensor,
    enable_weights: bool = False,
    top_k: int = 1,
    num_tokens: int = None,
    expert_offset: int = 0,
    reindex_a: bool = True,
    reindex_c: bool = True,
):
    """fused moe kernel launcher."""

    if num_tokens is None:
        num_tokens = A.size(0)
    E, N, K = B.shape

    def _grid_fn(META):
        grid = (
            E,
            triton.cdiv(num_tokens, META['BLOCK_SIZE_M']) *
            triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
        return grid

    GROUP_SIZE_M = 1
    A = A.flatten(0, -2)
    C = C.flatten(0, -2)

    grid = _grid_fn
    kernel_meta = get_kernel_meta(A)
    fused_moe_kernel[grid](
        A,
        B,
        C,
        sorted_idx,
        exp_start,
        exp_end,
        weights,
        N=N,
        K=K,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_be=B.stride(0),
        stride_bn=B.stride(1),
        stride_bk=B.stride(2),
        stride_cm=C.stride(0),
        stride_cn=C.stride(1),
        ENABLE_WEIGHTS=enable_weights,
        top_k=top_k,
        expert_offset=expert_offset,
        reindex_a=reindex_a,
        reindex_c=reindex_c,
        GROUP_SIZE_M=GROUP_SIZE_M,
        **kernel_meta,
    )


@wrap_jit_func(type_hint=dict(TopkIdx=torch.Tensor,
                              SortedIdx=torch.Tensor,
                              ExpStart=torch.Tensor,
                              ExpEnd=torch.Tensor,
                              len_sorted_idx=int,
                              num_experts=torch.int32,
                              BLOCK=torch.int32))
@triton.jit
def _start_end_kernel(TopkIdx, SortedIdx, ExpStart, ExpEnd,
                      len_sorted_idx: int, num_experts: tl.constexpr,
                      BLOCK: tl.constexpr):
    """start end kernel."""
    exp_id = tl.program_id(0)
    exp_start = -1
    cnt = 0

    s_off = tl.arange(0, BLOCK)

    # find start
    for sidx_start in range(0, len_sorted_idx, BLOCK):
        sidx_off = sidx_start + s_off
        sidx_mask = sidx_off < len_sorted_idx
        sidx = tl.load(SortedIdx + sidx_off, mask=sidx_mask, other=0)
        tidx = tl.load(TopkIdx + sidx, mask=sidx_mask, other=num_experts)
        tidx_mask = tidx == exp_id
        cnt += tl.sum(tidx_mask.to(tl.int32))
        if cnt > 0 and exp_start < 0:
            exp_start = sidx_start + tl.argmax(tidx_mask, axis=0)

    if exp_start < 0:
        exp_start *= 0
    exp_end = exp_start + cnt
    tl.store(ExpStart + exp_id, exp_start)
    tl.store(ExpEnd + exp_id, exp_end)


def get_start_end(topk_idx: torch.Tensor, sorted_idx: torch.Tensor,
                  num_experts: int):
    """get start and end.

    same process as:
    >>> exp_tok_cnt = F.one_hot(flatten_topk_ids, num_classes=E).sum(0)
    >>> exp_end = exp_tok_cnt.cumsum(0)
    >>> exp_start = exp_end - exp_tok_cnt
    """
    start_end = sorted_idx.new_empty(2, num_experts)
    exp_start = start_end[0, :]
    exp_end = start_end[1, :]

    BLOCK = 128
    kernel_meta = get_kernel_meta(topk_idx)
    _start_end_kernel[(num_experts, )](
        topk_idx,
        sorted_idx,
        exp_start,
        exp_end,
        len_sorted_idx=sorted_idx.numel(),
        num_experts=num_experts,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=1,
        **kernel_meta,
    )

    return exp_start, exp_end


def fused_moe(hidden_states: torch.Tensor,
              w1: torch.Tensor,
              w2: torch.Tensor,
              topk_weights: torch.Tensor,
              topk_ids: torch.Tensor,
              topk: int,
              expert_offset: int = 0,
              num_experts: int = None,
              renormalize: bool = False) -> torch.Tensor:
    """fused moe."""
    M = hidden_states.size(0)
    E, N, _ = w1.shape
    full_exp = False
    if num_experts is None:
        num_experts = E
    elif num_experts == E:
        full_exp = True

    def __get_sorted_idx(topk_ids: torch.Tensor):
        flatten_topk_ids = topk_ids.flatten()
        sorted_idx = flatten_topk_ids.argsort()

        exp_start, exp_end = get_start_end(flatten_topk_ids, sorted_idx,
                                           num_experts)
        return sorted_idx, exp_start, exp_end

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()

    sorted_idx, exp_start, exp_end = __get_sorted_idx(topk_ids)

    if full_exp:
        intermediate_cache1 = hidden_states.new_empty((M, topk, N))
    else:
        intermediate_cache1 = hidden_states.new_zeros((M, topk, N))
    # gate and up
    fused_moe_kernel_launcher(
        hidden_states,
        w1,
        intermediate_cache1,
        sorted_idx=sorted_idx,
        exp_start=exp_start,
        exp_end=exp_end,
        weights=topk_weights,
        enable_weights=False,
        top_k=topk,
        num_tokens=M,
        expert_offset=expert_offset,
        reindex_a=True,
        reindex_c=False,
    )

    # activate
    gate_cache, up_cache = intermediate_cache1.chunk(2, -1)
    gate_cache = F.silu(gate_cache, inplace=True) * up_cache

    if full_exp:
        intermediate_cache2 = hidden_states.new_empty((M, topk, w2.shape[1]))
    else:
        intermediate_cache2 = hidden_states.new_zeros((M, topk, w2.shape[1]))
    # down
    fused_moe_kernel_launcher(
        gate_cache,
        w2,
        intermediate_cache2,
        sorted_idx=sorted_idx,
        exp_start=exp_start,
        exp_end=exp_end,
        weights=topk_weights,
        enable_weights=True,
        top_k=1,
        num_tokens=M,
        expert_offset=expert_offset,
        reindex_a=False,
        reindex_c=True,
    )

    ret = intermediate_cache2.sum(dim=1)
    return ret

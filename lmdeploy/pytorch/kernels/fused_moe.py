# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_moe_kernel(
    A,
    B,
    C,
    SortedIdx,
    ExpStart,
    ExpEnd,
    Weights,
    N: int,
    K: int,
    stride_am: int,
    stride_ak: int,
    stride_be: int,
    stride_bn: int,
    stride_bk: int,
    stride_cm: int,
    stride_cn: int,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ENABLE_WEIGHTS: tl.constexpr,
    top_k: tl.constexpr,
):
    """fused moe kernel."""
    exp_id = tl.program_id(0)
    pid = tl.program_id(1)

    exp_start = tl.load(ExpStart + exp_id)
    exp_end = tl.load(ExpEnd + exp_id)
    M = exp_end - exp_start
    if M == 0:
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

    offs_am = sid // top_k
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + exp_id * stride_be + (offs_k[:, None] * stride_bk +
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

    offs_cm = sid
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
):
    """fused moe kernel launcher."""

    def _kernel_meta():
        from triton.runtime.jit import get_cuda_stream
        device = A.device
        device_idx = device.index
        device_type = device.type
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type=device_type, stream=stream)

    if num_tokens is None:
        num_tokens = A.size(0)
    E, N, K = B.shape
    A = A.flatten(0, -2)
    C = C.flatten(0, -2)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 1
    grid = (E, triton.cdiv(num_tokens, BLOCK_SIZE_M) *
            triton.cdiv(N, BLOCK_SIZE_N))
    kernel_meta = _kernel_meta()
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
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        ENABLE_WEIGHTS=enable_weights,
        top_k=top_k,
        num_warps=4,
        num_stages=1,
        **kernel_meta,
    )


def fused_moe(hidden_states: torch.Tensor,
              w1: torch.Tensor,
              w2: torch.Tensor,
              topk_weights: torch.Tensor,
              topk_ids: torch.Tensor,
              topk: int,
              renormalize: bool = False) -> torch.Tensor:
    """fused moe."""
    device = hidden_states.device
    M = hidden_states.size(0)
    E, N, _ = w1.shape

    def __get_sorted_idx(topk_ids: torch.Tensor):
        flatten_topk_ids = topk_ids.flatten()
        sorted_idx = flatten_topk_ids.argsort()
        exp_range = torch.arange(0, E, device=device)
        exp_tok_cnt = (flatten_topk_ids[None, :] == exp_range[:, None]).sum(1)
        return sorted_idx, exp_tok_cnt

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.contiguous()

    sorted_idx, exp_tok_cnt = __get_sorted_idx(topk_ids)
    exp_end = exp_tok_cnt.cumsum(0)
    exp_start = exp_end - exp_tok_cnt

    intermediate_cache1 = hidden_states.new_empty((M, topk, N))
    intermediate_cache2 = hidden_states.new_empty((M, topk, w2.shape[1]))
    # gate and up
    fused_moe_kernel_launcher(hidden_states,
                              w1,
                              intermediate_cache1,
                              sorted_idx=sorted_idx,
                              exp_start=exp_start,
                              exp_end=exp_end,
                              weights=topk_weights,
                              enable_weights=False,
                              top_k=topk,
                              num_tokens=hidden_states.size(0))

    # activate
    gate_cache, up_cache = intermediate_cache1.chunk(2, -1)
    F.silu(gate_cache, inplace=True).mul_(up_cache)

    # down
    fused_moe_kernel_launcher(gate_cache,
                              w2,
                              intermediate_cache2,
                              sorted_idx=sorted_idx,
                              exp_start=exp_start,
                              exp_end=exp_end,
                              weights=topk_weights,
                              enable_weights=True,
                              top_k=1,
                              num_tokens=hidden_states.size(0))

    ret = intermediate_cache2.sum(dim=1)
    return ret

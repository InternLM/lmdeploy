# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
from collections.abc import Callable

import torch
import triton
import triton.language as tl

from .activation import silu_and_mul


def get_cuda_autotune_config():
    return [
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 256,
            'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 1,
        },
                      num_stages=3,
                      num_warps=8),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 256,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 1,
        },
                      num_stages=4,
                      num_warps=4),
        # SM8
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
        # SM7-
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 1,
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 1,
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 1,
        },
                      num_stages=5,
                      num_warps=2),
    ]


def _config_prune_func(config: list, *args, **kwargs):
    """Fused moe config prune."""
    device_cap = torch.cuda.get_device_capability()
    num_sm9x = 2
    cum_num_sm8x = 5

    if device_cap[0] >= 9:
        return config[:num_sm9x]
    elif device_cap[0] >= 8:
        return config[num_sm9x:cum_num_sm8x]
    else:
        return config[cum_num_sm8x:]


@triton.jit
def _sorted_idx_phase1_kernel(
    ExpertIds,
    Counts,
    LocalPos,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Phase 1: sort within CTA, atomic-count per expert, store local position."""
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK_SIZE)
    offs = pid * BLOCK_SIZE + lane
    mask = offs < N

    # Pack (expert_id, local_lane) into one int32 for key-value sort
    expert_ids = tl.load(ExpertIds + offs, mask=mask, other=0).to(tl.int32)
    packed = tl.where(mask, expert_ids * BLOCK_SIZE + lane, 0x7FFFFFFF)

    # Sort groups same-expert threads for atomic coalescing
    sorted_packed = tl.sort(packed)
    sorted_expert = (sorted_packed // BLOCK_SIZE).to(tl.int64)
    sorted_local_idx = sorted_packed % BLOCK_SIZE
    sorted_valid = sorted_packed < 0x7FFFFFFF

    # Atomic count: Counts starts at 0, each thread adds 1, gets back local position
    local_pos = tl.atomic_add(Counts + sorted_expert, 1, mask=sorted_valid)

    # Store local_pos at original global index for phase 2
    orig = (pid * BLOCK_SIZE + sorted_local_idx).to(tl.int64)
    tl.store(LocalPos + orig, local_pos, mask=sorted_valid)


@triton.jit
def _sorted_idx_phase2_kernel(
    ExpertIds,
    LocalPos,
    ExpEnd,
    Counts,
    Out,
    ExpStart,
    N,
    num_experts,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Phase 2: scatter sorted_idx using cumsum result + compute exp_start."""
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK_SIZE)
    offs = pid * BLOCK_SIZE + lane
    mask = offs < N

    # Compute exp_start = exp_end - counts (only first block writes it)
    if pid == 0:
        e_offs = tl.arange(0, BLOCK_E)
        e_mask = e_offs < num_experts
        end_val = tl.load(ExpEnd + e_offs, mask=e_mask)
        cnt_val = tl.load(Counts + e_offs, mask=e_mask)
        tl.store(ExpStart + e_offs, end_val - cnt_val, mask=e_mask)

    # Scatter: sorted_idx[exp_start[e] + local_pos] = orig_idx
    expert_ids = tl.load(ExpertIds + offs, mask=mask, other=0).to(tl.int64)
    local_pos = tl.load(LocalPos + offs, mask=mask, other=0)
    end_val = tl.load(ExpEnd + expert_ids, mask=mask)
    cnt_val = tl.load(Counts + expert_ids, mask=mask)
    dst = end_val - cnt_val + local_pos

    tl.store(Out + dst, offs, mask=mask)


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['N', 'K', 'tune_hint'],
    prune_configs_by=dict(early_config_prune=_config_prune_func),
)
@triton.jit
def fused_moe_kernel(
    A,
    B,
    bias,
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
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_bie: tl.constexpr,
    stride_bin: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    M_NP2: tl.constexpr,
    tune_hint: tl.constexpr,
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
    exp_off = stride_be * exp_id.to(tl.int64)
    b_ptrs = B + exp_off + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=mask_sid[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if bias is not None:
        bias_ptrs = bias + exp_id * stride_bie + offs_bn * stride_bin
        bias_val = tl.load(bias_ptrs).to(accumulator.dtype)
        accumulator += bias_val[None]

    c = accumulator.to(A.dtype.element_ty)

    if reindex_c:
        offs_cm = sid
    else:
        offs_cm = offs_sid
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_bn[None, :]
    tl.store(c_ptrs, c, mask=mask_sid[:, None])


def fused_moe_kernel_launcher(
    A: torch.Tensor,
    B: torch.Tensor,
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
    tune_hint = min(2, triton.cdiv(M_NP2, 512))

    def _grid_fn(META):
        grid = (triton.cdiv(M_NP2, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), E)
        return grid

    A = A.flatten(0, -2)
    C = C.flatten(0, -2)
    enable_bias = bias is not None

    grid = _grid_fn
    fused_moe_kernel[grid](
        A,
        B,
        bias,
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
        stride_cm=C.stride(0),
        stride_cn=C.stride(1),
        stride_bie=bias.stride(0) if enable_bias else 0,
        stride_bin=bias.stride(1) if enable_bias else 0,
        tune_hint=tune_hint,
        top_k=top_k,
        expert_offset=expert_offset,
        reindex_a=reindex_a,
        reindex_c=reindex_c,
        M_NP2=M_NP2,
    )



def _get_sorted_idx_triton(topk_ids: torch.Tensor, num_experts: int):
    """Get sorted idx with 2-phase Triton kernels (4 kernel launches total)."""
    if topk_ids.dim() != 2:
        raise ValueError(f'topk_ids must be a 2D tensor, but got dim={topk_ids.dim()}')
    if topk_ids.size(1) > num_experts:
        raise ValueError(
            f'topk_ids.size(1) must be <= num_experts, but got topk={topk_ids.size(1)} '
            f'and num_experts={num_experts}')

    topk_ids = topk_ids.flatten()
    N = topk_ids.numel()

    BLOCK_SIZE = triton.next_power_of_2(min(num_experts, 256))
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    # Phase 1: sort + atomic histogram + store local positions
    # counts starts at 0; after phase1, counts[e] = number of tokens for expert e
    counts = torch.zeros(num_experts, dtype=topk_ids.dtype, device=topk_ids.device)
    local_pos = torch.empty(N, dtype=topk_ids.dtype, device=topk_ids.device)
    _sorted_idx_phase1_kernel[grid](topk_ids, counts, local_pos, N, BLOCK_SIZE=BLOCK_SIZE)

    # cumsum to get exp_end
    exp_end = torch.cumsum(counts, dim=0)

    # Phase 2: scatter sorted_idx + compute exp_start
    sorted_idx = torch.empty(N, dtype=topk_ids.dtype, device=topk_ids.device)
    exp_start = torch.empty(num_experts, dtype=topk_ids.dtype, device=topk_ids.device)
    BLOCK_E = triton.next_power_of_2(num_experts)
    _sorted_idx_phase2_kernel[grid](
        topk_ids, local_pos, exp_end, counts,
        sorted_idx, exp_start, N, num_experts,
        BLOCK_SIZE=BLOCK_SIZE, BLOCK_E=BLOCK_E,
    )

    return sorted_idx, exp_start, exp_end

_get_sorted_idx = _get_sorted_idx_triton


def _renormalize(topk_weights: torch.Tensor, renormalize: bool):
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()
    return topk_weights


def _make_intermediate(shape: tuple, dtype: torch.dtype, device: torch.device, zeros: bool):
    """Make intermediate."""
    if zeros:
        return torch.zeros(shape, dtype=dtype, device=device)
    else:
        return torch.empty(shape, dtype=dtype, device=device)


@triton.jit
def _moe_reduce_kernel(
    hidden_states_ptr,
    weights_ptr,
    out_ptr,
    stride_hm,
    stride_hk: tl.constexpr,
    stride_hn: tl.constexpr,
    stride_wm,
    stride_wk: tl.constexpr,
    stride_om,
    stride_on: tl.constexpr,
    fp32_acc: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_split = tl.cdiv(N, BLOCK_N)
    mid = pid // num_n_split
    nid = pid % num_n_split

    offs_k = tl.arange(0, BLOCK_K)
    offs_n = nid * BLOCK_N + tl.arange(0, BLOCK_N)
    weights_ptrs = weights_ptr + mid * stride_wm + offs_k * stride_wk
    h_ptrs = hidden_states_ptr + mid * stride_hm + offs_k[:, None] * stride_hk + offs_n[None, :] * stride_hn
    o_ptrs = out_ptr + mid * stride_om + offs_n * stride_on

    mask_k = offs_k < K
    mask_n = offs_n < N  # dummy load to get N
    mask_h = mask_k[:, None] & mask_n[None, :]

    h = tl.load(h_ptrs, mask=mask_h, other=0.0)
    w = tl.load(weights_ptrs, mask=mask_k, other=0.0)

    if fp32_acc:
        h = h.to(tl.float32)
        w = w.to(tl.float32)
    else:
        w = w.to(h.dtype)

    wh = h * w[:, None]
    o = wh.sum(axis=0)
    tl.store(o_ptrs, o, mask=mask_n)


def moe_reduce(hidden_states: torch.Tensor, topk_weights: torch.Tensor, fp32_acc: bool = False) -> torch.Tensor:
    """Moe reduce."""
    assert hidden_states.dim() == 3
    assert topk_weights.dim() == 2
    assert hidden_states.size(0) == topk_weights.size(0)
    assert hidden_states.size(1) == topk_weights.size(1)
    M, K, N = hidden_states.shape

    out = hidden_states.new_empty((M, N))

    BLOCK_K = triton.next_power_of_2(K)
    num_warps = 1
    BLOCK_N = triton.cdiv(num_warps * 512, hidden_states.element_size())
    grid = (M * triton.cdiv(N, BLOCK_N), )

    _moe_reduce_kernel[grid](
        hidden_states,
        topk_weights,
        out,
        hidden_states.stride(0),
        hidden_states.stride(1),
        hidden_states.stride(2),
        topk_weights.stride(0),
        topk_weights.stride(1),
        out.stride(0),
        out.stride(1),
        fp32_acc,
        K,
        N,
        BLOCK_K,
        BLOCK_N,
        num_warps=num_warps,
    )

    return out


def fused_moe(hidden_states: torch.Tensor,
              w1: torch.Tensor,
              w2: torch.Tensor,
              topk_weights: torch.Tensor,
              topk_ids: torch.Tensor,
              topk: int,
              w1_bias: torch.Tensor = None,
              w2_bias: torch.Tensor = None,
              expert_offset: int = 0,
              num_experts: int = None,
              renormalize: bool = False,
              act_func: Callable = None) -> torch.Tensor:
    """Fused moe."""
    M = hidden_states.size(0)
    E, N, _ = w1.shape
    if num_experts is None:
        num_experts = E
    full_exp = num_experts == E

    topk_weights = _renormalize(topk_weights, renormalize)
    sorted_idx, exp_start, exp_end = _get_sorted_idx(topk_ids, num_experts)

    intermediate_cache1 = _make_intermediate((M, topk, N),
                                             dtype=hidden_states.dtype,
                                             device=hidden_states.device,
                                             zeros=not full_exp)
    # gate and up
    fused_moe_kernel_launcher(
        hidden_states,
        w1,
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
    unflat_size = intermediate_cache1.shape[:-1]
    intermediate_cache1 = intermediate_cache1.flatten(0, -2)

    if act_func is None:
        gate_cache = silu_and_mul(intermediate_cache1)
    else:
        gate_cache = act_func(intermediate_cache1)
    gate_cache = gate_cache.unflatten(0, unflat_size)

    intermediate_cache2 = _make_intermediate((M, topk, w2.shape[1]),
                                             dtype=hidden_states.dtype,
                                             device=hidden_states.device,
                                             zeros=not full_exp)
    # down
    fused_moe_kernel_launcher(
        gate_cache,
        w2,
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

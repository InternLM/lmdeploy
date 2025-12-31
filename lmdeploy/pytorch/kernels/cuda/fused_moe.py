# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
from typing import Callable

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
    ]


def _config_prune_func(config: dict, *args, **kwargs):
    """Fused moe config prune."""
    device_cap = torch.cuda.get_device_capability()
    num_sm9x = 2

    if device_cap[0] >= 9:
        return config[:num_sm9x]
    else:
        return config[num_sm9x:]


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


@triton.jit
def _get_exp_mask_kernel(
    a_ptr,
    o_mask_ptr,
    o_k_ptr,
    stride_a_token: tl.constexpr,
    stride_a_exp: tl.constexpr,
    stride_o_exp,
    stride_o_token: tl.constexpr,
    topk: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_NA: tl.constexpr,
    BLOCK_NO: tl.constexpr,
):
    token_id = tl.program_id(0)

    offs_n = tl.arange(0, BLOCK_NA)
    mask_n = offs_n < topk
    a_ptrs = a_ptr + token_id * stride_a_token + offs_n * stride_a_exp
    a = tl.load(a_ptrs, mask=mask_n)

    # fill zeros
    offs_no = tl.arange(0, BLOCK_NO)
    mask_no = offs_no < num_experts
    o_ptrs = o_mask_ptr + token_id * stride_o_token + offs_no * stride_o_exp
    tl.store(o_ptrs, 0, mask=mask_no)

    # fill a
    o_ptrs = o_mask_ptr + token_id * stride_o_token + a * stride_o_exp
    tl.store(o_ptrs, 1, mask=mask_n)

    # fill kid
    ok_ptrs = o_k_ptr + token_id * stride_o_token + a * stride_o_exp
    tl.store(ok_ptrs, offs_n, mask=mask_n)


def _get_exp_mask(topk_ids: torch.Tensor, num_experts: int):
    """Get exp mask."""
    assert topk_ids.dim() == 2
    M, topk = topk_ids.shape
    assert topk <= num_experts

    out_mask = topk_ids.new_empty((num_experts, M))
    out_k = topk_ids.new_empty((num_experts, M))
    BLOCK_NA = triton.next_power_of_2(topk)
    BLOCK_NO = triton.next_power_of_2(num_experts)

    grid = (M, )
    _get_exp_mask_kernel[grid](
        topk_ids,
        out_mask,
        out_k,
        stride_a_token=topk_ids.stride(0),
        stride_a_exp=topk_ids.stride(1),
        stride_o_exp=out_mask.stride(0),
        stride_o_token=out_mask.stride(1),
        topk=topk,
        num_experts=num_experts,
        BLOCK_NA=BLOCK_NA,
        BLOCK_NO=BLOCK_NO,
        num_warps=1,
    )
    return out_mask, out_k


@triton.jit
def _get_start_end_kernel(
    exp_cum_ptr,
    exp_topk_ptr,
    exp_out_ptr,
    start_ptr,
    end_ptr,
    stride_cum_exp,
    stride_cum_token: tl.constexpr,
    stride_out: tl.constexpr,
    num_tokens,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Get start end kernel."""
    token_start = tl.program_id(0)

    offs_exp = tl.arange(0, BLOCK_N)
    off_cum = offs_exp * stride_cum_exp + token_start * stride_cum_token
    cum_ptrs = exp_cum_ptr + off_cum
    val_k_ptrs = exp_topk_ptr + off_cum

    mask_exp = offs_exp < num_experts

    # get prev and cur cum
    token_id = token_start
    prev_cum_mask = mask_exp
    if token_start == 0:
        prev_cum_mask = mask_exp & (tl.arange(0, BLOCK_N) > 0)
    prev_cum = tl.load(cum_ptrs - stride_cum_token, mask=prev_cum_mask, other=0)
    cur_cum = tl.load(cum_ptrs, mask=mask_exp)

    # store sorted idx
    mask_out = mask_exp & (cur_cum > prev_cum)
    val_k = tl.load(val_k_ptrs, mask=mask_exp)
    val = token_id * topk + val_k
    out_ptrs = exp_out_ptr + prev_cum * stride_out
    tl.store(out_ptrs, val, mask=mask_out)

    # fill start
    if token_id == 0:
        cur_start_ptrs = start_ptr + offs_exp
        tl.store(cur_start_ptrs, prev_cum, mask=mask_exp)

    # fill end
    if token_id == num_tokens - 1:
        cur_end_ptrs = end_ptr + offs_exp
        tl.store(cur_end_ptrs, cur_cum, mask=mask_exp)


def get_start_end(exp_cum: torch.Tensor, exp_topk: torch.Tensor, topk: int):
    """Get start end."""
    num_experts, num_tokens = exp_cum.shape

    start_end = exp_cum.new_empty(2, num_experts)
    exp_start = start_end[0, :]
    exp_end = start_end[1, :]

    out = exp_cum.new_empty((num_tokens * topk))

    num_warps = 1

    BLOCK_N = triton.next_power_of_2(num_experts)
    grid = (num_tokens, )

    _get_start_end_kernel[grid](
        exp_cum,
        exp_topk,
        out,
        exp_start,
        exp_end,
        stride_cum_exp=exp_cum.stride(0),
        stride_cum_token=exp_cum.stride(1),
        stride_out=out.stride(0),
        num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return out, exp_start, exp_end


def _get_sorted_idx(topk_ids: torch.Tensor, num_experts: int):
    """Get sorted idx."""
    assert topk_ids.dim() == 2
    _, topk = topk_ids.shape

    # get expert mask   (num_experts, num_tokens)
    exp_mask, exp_topk = _get_exp_mask(topk_ids, num_experts)
    # get cumsum   (num_experts, num_tokens)
    exp_cum = exp_mask.flatten().cumsum(0).view_as(exp_mask)

    # get sort idx and start/end
    sorted_idx, start, end = get_start_end(exp_cum, exp_topk, topk)

    return sorted_idx, start, end


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

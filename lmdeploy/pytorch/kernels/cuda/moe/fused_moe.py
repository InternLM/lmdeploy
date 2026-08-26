# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
from collections.abc import Callable

import torch
import triton
import triton.language as tl

from ..activation import silu_and_mul


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
    num_routes,
    BLOCK_R: tl.constexpr,
):
    """Phase 1: sort within CTA, atomic-count per expert, store local position."""
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK_R)
    route_offsets = pid * BLOCK_R + lane
    route_mask = route_offsets < num_routes

    # Pack (expert_id, local_lane) into one int32 for key-value sort
    expert_ids = tl.load(ExpertIds + route_offsets, mask=route_mask, other=0).to(tl.int32)
    packed = tl.where(route_mask, expert_ids * BLOCK_R + lane, 0x7FFFFFFF)

    # Sort groups same-expert threads for atomic coalescing
    sorted_packed = tl.sort(packed)
    sorted_expert = sorted_packed // BLOCK_R
    sorted_local_idx = sorted_packed % BLOCK_R
    sorted_valid = sorted_packed < 0x7FFFFFFF

    # Atomic count: Counts starts at 0, each thread adds 1, gets back local position
    local_pos = tl.atomic_add(Counts + sorted_expert,
                              1,
                              mask=sorted_valid,
                              sem='relaxed',
                              scope='gpu')

    # Store local_pos at the original route index for the scatter pass
    original_offset = pid * BLOCK_R + sorted_local_idx
    tl.store(LocalPos + original_offset, local_pos, mask=sorted_valid)


@triton.jit
def _route_prefix_kernel(
    Counts,
    ExpStart,
    ExpEnd,
    BlockEnd,
    num_experts,
    local_num_experts,
    expert_offset,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_LE: tl.constexpr,
    BUILD_BLOCKS: tl.constexpr,
):
    """Build global expert and optional local block prefixes in one CTA."""
    expert_offsets = tl.arange(0, BLOCK_E)
    expert_mask = expert_offsets < num_experts
    counts = tl.load(Counts + expert_offsets, mask=expert_mask, other=0)
    exp_end = tl.cumsum(counts, axis=0)
    tl.store(ExpStart + expert_offsets, exp_end - counts, mask=expert_mask)
    tl.store(ExpEnd + expert_offsets, exp_end, mask=expert_mask)

    if BUILD_BLOCKS:
        local_expert_offsets = tl.arange(0, BLOCK_LE)
        local_expert_mask = local_expert_offsets < local_num_experts
        local_counts = tl.load(Counts + expert_offset + local_expert_offsets,
                               mask=local_expert_mask,
                               other=0)
        block_counts = (local_counts + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        block_end = tl.cumsum(block_counts, axis=0)
        tl.store(BlockEnd + local_expert_offsets, block_end, mask=local_expert_mask)


@triton.jit
def _scatter_sorted_idx_kernel(
    ExpertIds,
    LocalPos,
    ExpStart,
    SortedIdx,
    num_routes,
    BLOCK_R: tl.constexpr,
):
    """Scatter route indices using global expert starts and local positions."""
    route_offsets = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    route_mask = route_offsets < num_routes
    expert_ids = tl.load(ExpertIds + route_offsets, mask=route_mask, other=0).to(tl.int32)
    local_pos = tl.load(LocalPos + route_offsets, mask=route_mask, other=0)
    exp_start = tl.load(ExpStart + expert_ids, mask=route_mask, other=0)
    tl.store(SortedIdx + exp_start + local_pos, route_offsets, mask=route_mask)


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
def _scatter_and_fill_block_meta_kernel(
    ExpertIds,
    LocalPos,
    Counts,
    ExpStart,
    BlockEnd,
    SortedIdx,
    BlockExpertIds,
    BlockOffsets,
    num_routes,
    num_route_blocks,
    local_num_experts,
    expert_offset,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Scatter routes and build local compact-block metadata."""
    pid = tl.program_id(0)
    if pid < num_route_blocks:
        route_offsets = pid * BLOCK_R + tl.arange(0, BLOCK_R)
        route_mask = route_offsets < num_routes
        expert_ids = tl.load(ExpertIds + route_offsets, mask=route_mask, other=0).to(tl.int32)
        local_pos = tl.load(LocalPos + route_offsets, mask=route_mask, other=0)
        exp_start = tl.load(ExpStart + expert_ids, mask=route_mask, other=0)
        tl.store(SortedIdx + exp_start + local_pos, route_offsets, mask=route_mask)

    if pid < local_num_experts:
        actual_expert = pid + expert_offset
        count = tl.load(Counts + actual_expert)
        n_blocks = tl.cdiv(count, BLOCK_SIZE_M)
        block_end = tl.load(BlockEnd + pid)
        block_base = block_end - n_blocks
        exp_start = tl.load(ExpStart + actual_expert)

        block_offsets = tl.arange(0, BLOCK_B)
        block_mask = block_offsets < n_blocks
        tl.store(BlockExpertIds + block_base + block_offsets, pid, mask=block_mask)
        tl.store(BlockOffsets + block_base + block_offsets,
                 exp_start + block_offsets * BLOCK_SIZE_M,
                 mask=block_mask)


@triton.jit
def _single_cta_route_prepare_kernel(
    ExpertIds,
    Cursors,
    SortedIdx,
    ExpStart,
    ExpEnd,
    BlockEnd,
    BlockExpertIds,
    BlockOffsets,
    num_routes,
    num_experts,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BUILD_BLOCKS: tl.constexpr,
):
    """Build sorted routes and optional block metadata in one CTA."""
    expert_offsets = tl.arange(0, BLOCK_E)
    expert_mask = expert_offsets < num_experts
    tl.store(Cursors + expert_offsets, 0, mask=expert_mask)
    tl.debug_barrier()

    # Keep each route's expert and local rank live across the prefix scan. This
    # removes both the histogram pass and the second ExpertIds load.
    route_offsets = tl.arange(0, BLOCK_R)
    route_mask = route_offsets < num_routes
    expert_ids = tl.load(ExpertIds + route_offsets,
                         mask=route_mask,
                         other=0).to(tl.int32)
    local_pos = tl.atomic_add(Cursors + expert_ids,
                              1,
                              mask=route_mask,
                              sem='relaxed',
                              scope='cta')
    tl.debug_barrier()

    counts = tl.load(Cursors + expert_offsets, mask=expert_mask, other=0)
    exp_end = tl.cumsum(counts, axis=0)
    exp_start = exp_end - counts
    tl.store(ExpStart + expert_offsets, exp_start, mask=expert_mask)
    tl.store(ExpEnd + expert_offsets, exp_end, mask=expert_mask)
    if BUILD_BLOCKS:
        block_counts = (counts + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        block_end = tl.cumsum(block_counts, axis=0)
        tl.store(BlockEnd + expert_offsets, block_end, mask=expert_mask)
    tl.debug_barrier()

    route_exp_start = tl.load(ExpStart + expert_ids, mask=route_mask, other=0)
    sorted_pos = route_exp_start + local_pos
    tl.store(SortedIdx + sorted_pos, route_offsets, mask=route_mask)

    if BUILD_BLOCKS:
        block_base = tl.where(
            expert_ids == 0,
            0,
            tl.load(BlockEnd + expert_ids - 1,
                    mask=route_mask & (expert_ids != 0),
                    other=0),
        )
        is_block_start = route_mask & (local_pos % BLOCK_SIZE_M == 0)
        block_id = block_base + local_pos // BLOCK_SIZE_M
        tl.store(BlockExpertIds + block_id,
                 expert_ids,
                 mask=is_block_start)
        tl.store(BlockOffsets + block_id,
                 sorted_pos,
                 mask=is_block_start)


@triton.jit
def fused_moe_compact_kernel(
    A,
    B,
    bias,
    C,
    SortedIdx,
    ExpEnd,
    BlockEnd,
    BlockExpertIds,
    BlockOffsets,
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
    top_k: tl.constexpr,
    expert_offset: tl.constexpr,
    num_local_experts: tl.constexpr,
    reindex_a: tl.constexpr,
    reindex_c: tl.constexpr,
):
    """Compact routed-block MoE GEMM."""
    block_id = tl.program_id(0)
    pid_n = tl.program_id(1)
    total_blocks = tl.load(BlockEnd + num_local_experts - 1)
    if block_id >= total_blocks:
        return

    local_exp = tl.load(BlockExpertIds + block_id)
    actual_exp = local_exp + expert_offset
    block_sorted_start = tl.load(BlockOffsets + block_id)
    exp_end = tl.load(ExpEnd + actual_exp)

    offs_sid = block_sorted_start + tl.arange(0, BLOCK_SIZE_M)
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

    exp_off = stride_be * local_exp.to(tl.int64)
    b_ptrs = B + exp_off + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=mask_sid[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if bias is not None:
        bias_ptrs = bias + local_exp * stride_bie + offs_bn * stride_bin
        bias_val = tl.load(bias_ptrs).to(accumulator.dtype)
        accumulator += bias_val[None, :]

    c = accumulator.to(A.dtype.element_ty)

    if reindex_c:
        offs_cm = sid
    else:
        offs_cm = offs_sid
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_bn[None, :]
    tl.store(c_ptrs, c, mask=mask_sid[:, None])


def fused_moe_compact_kernel_launcher(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    sorted_idx: torch.Tensor,
    exp_end: torch.Tensor,
    block_end: torch.Tensor,
    block_expert_ids: torch.Tensor,
    block_offsets: torch.Tensor,
    bias: torch.Tensor = None,
    top_k: int = 1,
    expert_offset: int = 0,
    reindex_a: bool = True,
    reindex_c: bool = True,
    block_m: int = 64,
    block_n: int = 256,
    block_k: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
):
    """Launch compact routed-block MoE kernel."""
    E, N, K = B.shape
    max_blocks = block_expert_ids.numel()

    A = A.flatten(0, -2)
    C = C.flatten(0, -2)
    enable_bias = bias is not None

    grid = (max_blocks, triton.cdiv(N, block_n))
    fused_moe_compact_kernel[grid](
        A,
        B,
        bias,
        C,
        sorted_idx,
        exp_end,
        block_end,
        block_expert_ids,
        block_offsets,
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
        top_k=top_k,
        expert_offset=expert_offset,
        num_local_experts=E,
        reindex_a=reindex_a,
        reindex_c=reindex_c,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _launch_parallel_route_rank(expert_ids: torch.Tensor, num_experts: int):
    """Count and rank routes independently across parallel CTAs."""
    num_routes = expert_ids.numel()
    block_r = triton.next_power_of_2(min(num_experts, 256))
    grid = (triton.cdiv(num_routes, block_r), )
    counts = torch.zeros(num_experts, dtype=torch.int32, device=expert_ids.device)
    local_pos = torch.empty(num_routes, dtype=torch.int32, device=expert_ids.device)
    _sorted_idx_phase1_kernel[grid](expert_ids, counts, local_pos, num_routes, BLOCK_R=block_r)
    return counts, local_pos, block_r, grid


def _get_sorted_idx_triton(topk_ids: torch.Tensor, num_experts: int):
    """Build sorted route indices with four CUDA graph nodes."""
    if topk_ids.dim() != 2:
        raise ValueError(f'topk_ids must be a 2D tensor, but got dim={topk_ids.dim()}')
    if topk_ids.size(1) > num_experts:
        raise ValueError(
            f'topk_ids.size(1) must be <= num_experts, but got topk={topk_ids.size(1)} '
            f'and num_experts={num_experts}')

    expert_ids = topk_ids.flatten()
    num_routes = expert_ids.numel()
    counts, local_pos, block_r, grid = _launch_parallel_route_rank(expert_ids, num_experts)

    exp_start = torch.empty(num_experts, dtype=torch.int32, device=expert_ids.device)
    exp_end = torch.empty(num_experts, dtype=torch.int32, device=expert_ids.device)
    _route_prefix_kernel[(1, )](
        counts,
        exp_start,
        exp_end,
        exp_start,
        num_experts,
        num_experts,
        0,
        BLOCK_SIZE_M=1,
        BLOCK_E=triton.next_power_of_2(num_experts),
        BLOCK_LE=1,
        BUILD_BLOCKS=False,
        num_warps=8,
    )

    sorted_idx = torch.empty(num_routes, dtype=torch.int32, device=expert_ids.device)
    _scatter_sorted_idx_kernel[grid](
        expert_ids,
        local_pos,
        exp_start,
        sorted_idx,
        num_routes,
        BLOCK_R=block_r,
    )

    return sorted_idx, exp_start, exp_end


def _get_sorted_idx_blocks_parallel(topk_ids: torch.Tensor,
                                    num_experts: int,
                                    local_num_experts: int,
                                    expert_offset: int,
                                    block_m: int):
    """Build sorted route indices and compact block metadata in four graph
    nodes."""
    if topk_ids.dim() != 2:
        raise ValueError(f'topk_ids must be a 2D tensor, but got dim={topk_ids.dim()}')
    if topk_ids.size(1) > num_experts:
        raise ValueError(
            f'topk_ids.size(1) must be <= num_experts, but got topk={topk_ids.size(1)} '
            f'and num_experts={num_experts}')

    expert_ids = topk_ids.flatten()
    num_routes = expert_ids.numel()
    counts, local_pos, block_r, route_grid = _launch_parallel_route_rank(expert_ids, num_experts)

    exp_start = torch.empty(num_experts, dtype=torch.int32, device=expert_ids.device)
    exp_end = torch.empty(num_experts, dtype=torch.int32, device=expert_ids.device)
    block_end = torch.empty(local_num_experts, dtype=torch.int32, device=expert_ids.device)
    _route_prefix_kernel[(1, )](
        counts,
        exp_start,
        exp_end,
        block_end,
        num_experts,
        local_num_experts,
        expert_offset,
        BLOCK_SIZE_M=block_m,
        BLOCK_E=triton.next_power_of_2(num_experts),
        BLOCK_LE=triton.next_power_of_2(local_num_experts),
        BUILD_BLOCKS=True,
        num_warps=8,
    )

    max_blocks = triton.cdiv(num_routes, block_m) + local_num_experts
    sorted_idx = torch.empty(num_routes, dtype=torch.int32, device=expert_ids.device)
    block_expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=expert_ids.device)
    block_offsets = torch.empty(max_blocks, dtype=torch.int32, device=expert_ids.device)
    block_b = triton.next_power_of_2(max(1, triton.cdiv(num_routes, block_m)))
    num_route_blocks = route_grid[0]
    grid = (max(num_route_blocks, local_num_experts), )
    _scatter_and_fill_block_meta_kernel[grid](
        expert_ids,
        local_pos,
        counts,
        exp_start,
        block_end,
        sorted_idx,
        block_expert_ids,
        block_offsets,
        num_routes,
        num_route_blocks,
        local_num_experts,
        expert_offset,
        BLOCK_SIZE_M=block_m,
        BLOCK_R=block_r,
        BLOCK_B=block_b,
    )

    return sorted_idx, exp_start, exp_end, block_end, block_expert_ids, block_offsets


def _should_use_single_cta_sorted_idx(num_routes: int, num_experts: int):
    """Use one CTA when it beats the parallel sorted-index preparation."""
    return num_routes <= 2048 and num_experts <= 2048


def _should_use_single_cta_sorted_idx_blocks(num_routes: int, num_experts: int):
    """Use one CTA when it beats the parallel compact-block preparation."""
    return num_routes <= 2048 and num_experts <= 2048


def _single_cta_route_prepare_num_warps(block_r: int):
    """Keep the retained route state below the register limit."""
    if block_r >= 8192:
        return 32
    if block_r >= 4096:
        return 16
    return 8


def _supports_single_cta_route_prepare(topk_ids: torch.Tensor):
    """Return whether the measured Hopper preparation is available."""
    return topk_ids.is_cuda and torch.cuda.get_device_capability(topk_ids.device)[0] == 9


def _get_sorted_idx_single_cta(topk_ids: torch.Tensor,
                               num_experts: int):
    """Build bounded-route sorted-index metadata in one CTA."""
    if topk_ids.dim() != 2:
        raise ValueError(f'topk_ids must be a 2D tensor, but got dim={topk_ids.dim()}')
    if topk_ids.size(1) > num_experts:
        raise ValueError(
            f'topk_ids.size(1) must be <= num_experts, but got topk={topk_ids.size(1)} '
            f'and num_experts={num_experts}')
    if num_experts > 2048:
        raise ValueError(f'single-CTA metadata supports at most 2048 experts, got {num_experts}')

    flatten_topk_ids = topk_ids.flatten()
    num_routes = flatten_topk_ids.numel()
    block_e = triton.next_power_of_2(num_experts)
    block_r = max(256, triton.next_power_of_2(num_routes))
    num_warps = _single_cta_route_prepare_num_warps(block_r)
    device = topk_ids.device
    cursors = torch.empty(num_experts, dtype=torch.int32, device=device)
    sorted_idx = torch.empty(num_routes, dtype=torch.int32, device=device)
    exp_start = torch.empty(num_experts, dtype=torch.int32, device=device)
    exp_end = torch.empty(num_experts, dtype=torch.int32, device=device)
    _single_cta_route_prepare_kernel[(1, )](
        flatten_topk_ids,
        cursors,
        sorted_idx,
        exp_start,
        exp_end,
        exp_start,
        exp_start,
        exp_start,
        num_routes,
        num_experts,
        BLOCK_SIZE_M=1,
        BLOCK_E=block_e,
        BLOCK_R=block_r,
        BUILD_BLOCKS=False,
        num_warps=num_warps,
    )
    return sorted_idx, exp_start, exp_end


def _get_sorted_idx_blocks_single_cta(topk_ids: torch.Tensor,
                                      num_experts: int,
                                      block_m: int):
    """Build bounded-route sorted-index and compact-block metadata in one
    CTA."""
    if topk_ids.dim() != 2:
        raise ValueError(f'topk_ids must be a 2D tensor, but got dim={topk_ids.dim()}')
    if topk_ids.size(1) > num_experts:
        raise ValueError(
            f'topk_ids.size(1) must be <= num_experts, but got topk={topk_ids.size(1)} '
            f'and num_experts={num_experts}')
    if num_experts > 2048:
        raise ValueError(f'single-CTA metadata supports at most 2048 experts, got {num_experts}')

    flatten_topk_ids = topk_ids.flatten()
    num_routes = flatten_topk_ids.numel()
    block_e = triton.next_power_of_2(num_experts)
    block_r = max(256, triton.next_power_of_2(num_routes))
    num_warps = _single_cta_route_prepare_num_warps(block_r)
    max_blocks = min(num_routes, triton.cdiv(num_routes, block_m) + num_experts)
    device = topk_ids.device
    cursors = torch.empty(num_experts, dtype=torch.int32, device=device)
    sorted_idx = torch.empty(num_routes, dtype=torch.int32, device=device)
    exp_start = torch.empty(num_experts, dtype=torch.int32, device=device)
    exp_end = torch.empty(num_experts, dtype=torch.int32, device=device)
    block_end = torch.empty(num_experts, dtype=torch.int32, device=device)
    block_expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=device)
    block_offsets = torch.empty(max_blocks, dtype=torch.int32, device=device)
    _single_cta_route_prepare_kernel[(1, )](
        flatten_topk_ids,
        cursors,
        sorted_idx,
        exp_start,
        exp_end,
        block_end,
        block_expert_ids,
        block_offsets,
        num_routes,
        num_experts,
        BLOCK_SIZE_M=block_m,
        BLOCK_E=block_e,
        BLOCK_R=block_r,
        BUILD_BLOCKS=True,
        num_warps=num_warps,
    )
    return sorted_idx, exp_start, exp_end, block_end, block_expert_ids, block_offsets


def _get_sorted_idx(topk_ids: torch.Tensor, num_experts: int):
    """Build route metadata with the best preparation for the shape."""
    if (_supports_single_cta_route_prepare(topk_ids)
            and _should_use_single_cta_sorted_idx(topk_ids.numel(), num_experts)):
        return _get_sorted_idx_single_cta(topk_ids, num_experts)
    return _get_sorted_idx_triton(topk_ids, num_experts)


def _get_sorted_idx_blocks(topk_ids: torch.Tensor,
                           num_experts: int,
                           local_num_experts: int,
                           expert_offset: int,
                           block_m: int):
    """Build compact block metadata with the best preparation for the shape."""
    if (_supports_single_cta_route_prepare(topk_ids) and local_num_experts == num_experts and expert_offset == 0
            and _should_use_single_cta_sorted_idx_blocks(topk_ids.numel(), num_experts)):
        return _get_sorted_idx_blocks_single_cta(topk_ids, num_experts, block_m)
    return _get_sorted_idx_blocks_parallel(topk_ids, num_experts, local_num_experts, expert_offset, block_m)


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


def _compact_moe_config(num_tokens: int, num_experts: int, local_ffn_size: int):
    """Choose a compact MoE config for the profiled TP bf16 shapes."""
    if num_experts >= 1024 and num_tokens >= 8192:
        return dict(block_m=64, block_n=512, block_k=64, num_warps=8, num_stages=3)
    if num_tokens <= 1024:
        return dict(block_m=64, block_n=256, block_k=32, num_warps=4, num_stages=4)
    return dict(block_m=64, block_n=512, block_k=64, num_warps=8, num_stages=3)


def _supports_compact_moe(hidden_states: torch.Tensor,
                          w1: torch.Tensor,
                          w2: torch.Tensor,
                          topk_ids: torch.Tensor,
                          num_experts: int,
                          expert_offset: int = 0):
    """Return whether this MoE call can use compact routed-block kernels."""
    if not hidden_states.is_cuda:
        return False
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        return False
    if w1.dtype != hidden_states.dtype or w2.dtype != hidden_states.dtype:
        return False
    if topk_ids.dim() != 2 or topk_ids.numel() == 0:
        return False
    if topk_ids.size(1) > num_experts:
        return False
    if w1.size(0) != w2.size(0):
        return False
    if expert_offset < 0:
        return False
    if w1.size(0) > num_experts:
        return False
    if expert_offset + w1.size(0) > num_experts:
        return False
    if torch.cuda.get_device_capability(hidden_states.device)[0] < 9:
        return False
    return True


def _should_use_compact_moe(hidden_states: torch.Tensor,
                            w1: torch.Tensor,
                            w2: torch.Tensor,
                            topk_ids: torch.Tensor,
                            num_experts: int,
                            expert_offset: int = 0):
    """Return whether both MoE projections should use compact scheduling."""
    if not _supports_compact_moe(hidden_states, w1, w2, topk_ids, num_experts, expert_offset):
        return False

    local_experts = w1.size(0)
    if local_experts >= 1024:
        return True

    avg_routes_per_expert = topk_ids.numel() / num_experts
    return avg_routes_per_expert >= 32


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
    use_compact = _should_use_compact_moe(hidden_states, w1, w2, topk_ids, num_experts, expert_offset)
    if use_compact:
        compact_cfg = _compact_moe_config(M, num_experts, w2.shape[2])
        sorted_idx, _, exp_end, block_end, block_expert_ids, block_offsets = _get_sorted_idx_blocks(
            topk_ids,
            num_experts,
            E,
            expert_offset,
            compact_cfg['block_m'],
        )
    else:
        sorted_idx, exp_start, exp_end = _get_sorted_idx(topk_ids, num_experts)

    intermediate_cache1 = _make_intermediate((M, topk, N),
                                             dtype=hidden_states.dtype,
                                             device=hidden_states.device,
                                             zeros=not full_exp)
    # gate and up
    if use_compact:
        fused_moe_compact_kernel_launcher(
            hidden_states,
            w1,
            intermediate_cache1,
            sorted_idx=sorted_idx,
            exp_end=exp_end,
            block_end=block_end,
            block_expert_ids=block_expert_ids,
            block_offsets=block_offsets,
            bias=w1_bias,
            top_k=topk,
            expert_offset=expert_offset,
            reindex_a=True,
            reindex_c=False,
            **compact_cfg,
        )
    else:
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
    if use_compact:
        fused_moe_compact_kernel_launcher(
            gate_cache,
            w2,
            intermediate_cache2,
            sorted_idx=sorted_idx,
            exp_end=exp_end,
            block_end=block_end,
            block_expert_ids=block_expert_ids,
            block_offsets=block_offsets,
            bias=w2_bias,
            top_k=1,
            expert_offset=expert_offset,
            reindex_a=False,
            reindex_c=True,
            **compact_cfg,
        )
    else:
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

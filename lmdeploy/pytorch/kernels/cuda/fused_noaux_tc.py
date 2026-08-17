# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl


@triton.jit
def _kimi_noaux_routing_kernel(
    logits_ptr,
    bias_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    batch_size,
    routed_scaling_factor,
    logits_stride_0,
    logits_stride_1,
    bias_stride_0,
    weights_stride_0,
    weights_stride_1,
    ids_stride_0,
    ids_stride_1,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused no-aux routing for Kimi K2.6's single expert group."""
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    expert_idx = tl.arange(0, BLOCK_SIZE)
    expert_mask = expert_idx < NUM_EXPERTS
    logits = tl.load(
        logits_ptr + pid * logits_stride_0 + expert_idx * logits_stride_1,
        mask=expert_mask,
        other=0.0,
    )
    bias = tl.load(bias_ptr + expert_idx * bias_stride_0, mask=expert_mask, other=0.0)
    scores = tl.sigmoid(logits)
    scores_for_choice = tl.where(expert_mask, scores + bias, -float('inf'))

    topk_slot = tl.arange(0, TOP_K)
    selected_weights = tl.zeros((TOP_K, ), dtype=tl.float32)
    selected_ids = tl.zeros((TOP_K, ), dtype=tl.int32)
    for k in range(TOP_K):
        expert_id = tl.argmax(scores_for_choice, axis=0)
        weight = tl.sum(tl.where(expert_idx == expert_id, scores, 0.0), axis=0)
        selected_weights = tl.where(topk_slot == k, weight, selected_weights)
        selected_ids = tl.where(topk_slot == k, expert_id, selected_ids)
        scores_for_choice = tl.where(expert_idx == expert_id, -float('inf'), scores_for_choice)

    denominator = tl.sum(selected_weights, axis=0) + 1e-20
    selected_weights = selected_weights / denominator * routed_scaling_factor
    weights_offset = pid * weights_stride_0 + topk_slot * weights_stride_1
    ids_offset = pid * ids_stride_0 + topk_slot * ids_stride_1
    tl.store(topk_weights_ptr + weights_offset, selected_weights)
    tl.store(topk_ids_ptr + ids_offset, selected_ids.to(tl.int64))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=['num_experts', 'n_group', 'top_k'],
)
@triton.jit
def _fused_noaux_tc_kernel(
    logits_ptr,
    bias_ptr,
    topk_weight_ptr,
    topk_idx_ptr,
    batch_size,
    num_experts: tl.constexpr,
    n_group: tl.constexpr,
    group_size: tl.constexpr,
    topk_group: tl.constexpr,
    top_k: tl.constexpr,
    renormalize: tl.constexpr,
    routed_scaling_factor,
    logits_stride_0,
    logits_stride_1,
    bias_stride_0,
    BLOCK_SIZE: tl.constexpr,
    TOPK_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return
    idx = tl.arange(0, BLOCK_SIZE)
    mask = idx < num_experts
    logits = tl.load(logits_ptr + pid * logits_stride_0 + idx * logits_stride_1, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + idx * bias_stride_0, mask=mask, other=0.0)
    scores = tl.sigmoid(logits)
    scores_for_choice = scores + bias

    if n_group == 1:
        candidates = tl.where(mask, scores_for_choice, -float('inf'))
    else:
        grouped_scores = tl.reshape(scores_for_choice, (n_group, group_size))
        group_max = tl.max(grouped_scores, axis=1)
        group_argmax = tl.argmax(grouped_scores, axis=1)
        group_idx = tl.arange(0, group_size)
        grouped_scores = tl.where(group_idx[None, :] == group_argmax[:, None], -float('inf'), grouped_scores)
        group_scores = group_max + tl.max(grouped_scores, axis=1)

        selected_groups = tl.zeros((BLOCK_SIZE, ), dtype=tl.int1)
        for _ in range(topk_group):
            selected_group = tl.argmax(group_scores, axis=0)
            group_start = selected_group * group_size
            selected_groups |= (idx >= group_start) & (idx < group_start + group_size) & mask
            group_ids = tl.arange(0, n_group)
            group_scores = tl.where(group_ids == selected_group, -float('inf'), group_scores)
        # Keep the existing zero-masking behavior for routing compatibility.
        candidates = tl.where(selected_groups, scores_for_choice, 0.0)

    output_idx = tl.arange(0, TOPK_BLOCK_SIZE)
    topk_weights = tl.zeros((TOPK_BLOCK_SIZE, ), dtype=tl.float32)
    topk_indices = tl.zeros((TOPK_BLOCK_SIZE, ), dtype=tl.int32)
    for rank in range(top_k):
        expert_idx = tl.argmax(candidates, axis=0)
        weight = tl.sum(tl.where(idx == expert_idx, scores, 0.0), axis=0)
        topk_weights += tl.where(output_idx == rank, weight, 0.0)
        topk_indices += tl.where(output_idx == rank, expert_idx, 0)
        candidates = tl.where(idx == expert_idx, -float('inf'), candidates)

    if renormalize:
        topk_weights /= tl.sum(topk_weights, axis=0) + 1e-20
    topk_weights *= routed_scaling_factor
    output_offsets = pid * top_k + output_idx
    output_mask = output_idx < top_k
    tl.store(topk_weight_ptr + output_offsets, topk_weights, mask=output_mask)
    tl.store(topk_idx_ptr + output_offsets, topk_indices, mask=output_mask)


def fused_noaux_tc_routing(
    logits: torch.Tensor,
    bias: torch.Tensor,
    num_experts: int = 256,
    n_group: int = 8,
    topk_group: int = 4,
    top_k: int = 8,
    renormalize: bool = True,
    routed_scaling_factor: float = 2.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = logits.shape[0]
    group_size = num_experts // n_group
    assert num_experts % n_group == 0, 'num_experts must be divisible by n_group'
    use_kimi_kernel = (
        num_experts == 384
        and n_group == 1
        and topk_group == 1
        and top_k == 8
        and renormalize
        and routed_scaling_factor == 2.827
        and logits.dtype == torch.float32
        and bias.dtype == torch.float32
    )
    logits = logits.float().contiguous()
    bias = bias.float().contiguous()
    if use_kimi_kernel:
        topk_weights = torch.empty(batch_size, top_k, device=logits.device, dtype=torch.float32)
        topk_ids = torch.empty(batch_size, top_k, device=logits.device, dtype=torch.int64)
        _kimi_noaux_routing_kernel[(batch_size, )](
            logits,
            bias,
            topk_weights,
            topk_ids,
            batch_size,
            routed_scaling_factor,
            logits_stride_0=logits.stride(0),
            logits_stride_1=logits.stride(1),
            bias_stride_0=bias.stride(0),
            weights_stride_0=topk_weights.stride(0),
            weights_stride_1=topk_weights.stride(1),
            ids_stride_0=topk_ids.stride(0),
            ids_stride_1=topk_ids.stride(1),
            NUM_EXPERTS=384,
            TOP_K=8,
            BLOCK_SIZE=512,
            num_warps=1,
            num_stages=1,
        )
        return topk_weights, topk_ids

    topk_weight = torch.empty(batch_size, top_k, device=logits.device, dtype=torch.float32)
    topk_idx = torch.empty(batch_size, top_k, device=logits.device, dtype=torch.int64)
    block_size = num_experts
    assert block_size % 32 == 0, 'num_experts must be a multiple of 32 for optimal performance'
    grid = (batch_size, )
    _fused_noaux_tc_kernel[grid](
        logits,
        bias,
        topk_weight,
        topk_idx,
        batch_size,
        num_experts=num_experts,
        n_group=n_group,
        group_size=group_size,
        topk_group=topk_group,
        top_k=top_k,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        logits_stride_0=logits.stride(0),
        logits_stride_1=logits.stride(1),
        bias_stride_0=bias.stride(0),
        BLOCK_SIZE=block_size,
        TOPK_BLOCK_SIZE=triton.next_power_of_2(top_k),
    )
    return topk_weight, topk_idx

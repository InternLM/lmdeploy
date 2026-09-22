# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl


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
    logits = logits.float().contiguous()
    bias = bias.float().contiguous()
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

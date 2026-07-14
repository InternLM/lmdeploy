# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=1, num_stages=3),
        triton.Config({}, num_warps=1, num_stages=4),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=4),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=4),
    ],
    key=['num_experts', 'n_group'],
)
@triton.jit
def _noaux_routing_kernel(
    logits_ptr,
    bias_ptr,
    scores_ptr,
    tmp_scores_ptr,
    batch_size,
    num_experts: tl.constexpr,
    n_group: tl.constexpr,
    group_size: tl.constexpr,
    topk_group: tl.constexpr,
    # The following arguments are not used inside the kernel but kept for signature compatibility
    renormalize: tl.constexpr,
    routed_scaling_factor,
    logits_stride_0,
    logits_stride_1,
    bias_stride_0,
    scores_stride_0,
    scores_stride_1,
    tmp_scores_stride_0,
    tmp_scores_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return
    idx = tl.arange(0, BLOCK_SIZE)
    mask = idx < num_experts  # always true if BLOCK_SIZE == num_experts, but kept for safety
    # 1. Load logits and bias
    logits = tl.load(logits_ptr + pid * logits_stride_0 + idx * logits_stride_1, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + idx * bias_stride_0, mask=mask, other=0.0)
    # 2. Compute scores (sigmoid) and bias‑adjusted scores
    scores = tl.sigmoid(logits)  # original scores
    scores_fc = scores + bias  # bias‑adjusted scores
    # 3. Compute group scores: sum of top‑2 scores_fc per group
    # Reshape to (n_group, group_size) – requires BLOCK_SIZE == num_experts
    scores_fc_2d = tl.reshape(scores_fc, (n_group, group_size))
    # Max and argmax per group
    max_val = tl.max(scores_fc_2d, axis=1)
    max_idx = tl.argmax(scores_fc_2d, axis=1)  # index within group (0..group_size-1)
    # Second max per group: mask out the max element
    col_range = tl.arange(0, group_size)
    mask_max = col_range[None, :] == max_idx[:, None]
    scores_fc_masked = tl.where(mask_max, -float('inf'), scores_fc_2d)
    second_max = tl.max(scores_fc_masked, axis=1)
    group_scores = max_val + second_max
    # 4. Select top‑k groups and build selected_mask
    selected_mask = tl.zeros((BLOCK_SIZE, ), dtype=tl.int1)
    group_scores_copy = group_scores
    for _ in range(topk_group):
        max_idx_g = tl.argmax(group_scores_copy, axis=0)  # group index
        # mark experts in this group
        group_start = max_idx_g * group_size
        group_end = group_start + group_size
        group_mask = (idx >= group_start) & (idx < group_end) & mask
        selected_mask = selected_mask | group_mask
        # remove this group
        g_idx = tl.arange(0, n_group)
        g_mask = g_idx == max_idx_g
        group_scores_copy = tl.where(g_mask, -float('inf'), group_scores_copy)
    # 5. Build masked scores (tmp_scores) – experts in selected groups keep scores_fc, others 0
    tmp_scores = tl.where(selected_mask, scores_fc, 0.0)
    # 6. Store outputs
    off_scores = pid * scores_stride_0 + idx * scores_stride_1
    tl.store(scores_ptr + off_scores, scores, mask=mask)
    off_tmp = pid * tmp_scores_stride_0 + idx * tmp_scores_stride_1
    tl.store(tmp_scores_ptr + off_tmp, tmp_scores, mask=mask)


# ---------------------------------------------------------------------------
# Wrappers and Benchmarking Logic (Kept exactly as requested)
# ---------------------------------------------------------------------------


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
    # Convert to float32 and ensure contiguous
    logits = logits.float().contiguous()
    bias = bias.float().contiguous()
    # Output tensors from the kernel
    scores = torch.empty(batch_size, num_experts, device=logits.device, dtype=torch.float32)
    tmp_scores = torch.empty(batch_size, num_experts, device=logits.device, dtype=torch.float32)
    # Block size: exactly num_experts (must be multiple of 32 for good performance)
    BLOCK_SIZE = num_experts
    # Ensure BLOCK_SIZE is at least 32 and a multiple of 32? Not strictly required but good.
    # If not multiple of 32, we could round up, but then reshape would break. So we assume it is.
    # For safety, we assert:
    assert BLOCK_SIZE % 32 == 0, 'num_experts must be a multiple of 32 for optimal performance'
    # Kernel launch
    grid = (batch_size, )
    _noaux_routing_kernel[grid](
        logits,
        bias,
        scores,
        tmp_scores,
        batch_size,
        num_experts=num_experts,
        n_group=n_group,
        group_size=group_size,
        topk_group=topk_group,
        renormalize=int(renormalize),  # not used inside kernel
        routed_scaling_factor=routed_scaling_factor,
        logits_stride_0=logits.stride(0),
        logits_stride_1=logits.stride(1),
        bias_stride_0=bias.stride(0),
        scores_stride_0=scores.stride(0),
        scores_stride_1=scores.stride(1),
        tmp_scores_stride_0=tmp_scores.stride(0),
        tmp_scores_stride_1=tmp_scores.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Final expert selection using PyTorch's topk (guarantees exact match)
    _, topk_idx = torch.topk(tmp_scores, k=top_k, dim=-1, sorted=False)
    topk_weight = scores.gather(1, topk_idx)
    if renormalize:
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weight = topk_weight * routed_scaling_factor
    return topk_weight, topk_idx

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl


@triton.jit
def _single_group_topk_router_kernel(
    scores_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    scores_stride_0,
    weights_stride_0,
    ids_stride_0,
    batch_size,
    routed_scaling_factor,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= batch_size:
        return

    expert_offsets = tl.arange(0, BLOCK_SIZE)
    expert_mask = expert_offsets < NUM_EXPERTS
    scores = tl.load(
        scores_ptr + row * scores_stride_0 + expert_offsets,
        mask=expert_mask,
        other=0.0,
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + expert_offsets,
        mask=expert_mask,
        other=0.0,
    ).to(tl.float32)
    choice_scores = tl.where(
        expert_mask,
        scores + bias,
        -float('inf'),
    )

    output_offsets = tl.arange(0, TOP_K)
    selected_ids = tl.zeros((TOP_K,), dtype=tl.int32)
    selected_weights = tl.zeros((TOP_K,), dtype=tl.float32)

    for index in tl.static_range(0, TOP_K):
        selected_id = tl.argmax(choice_scores, axis=0)
        selected_weight = tl.sum(
            tl.where(
                expert_offsets == selected_id,
                scores,
                0.0,
            ),
            axis=0,
        )
        selected_ids = tl.where(
            output_offsets == index,
            selected_id,
            selected_ids,
        )
        selected_weights = tl.where(
            output_offsets == index,
            selected_weight,
            selected_weights,
        )
        choice_scores = tl.where(
            expert_offsets == selected_id,
            -float('inf'),
            choice_scores,
        )

    denominator = (
        tl.sum(selected_weights, axis=0) + 1e-20
    )
    selected_weights = (
        selected_weights
        / denominator
        * routed_scaling_factor
    )
    tl.store(
        weights_ptr
        + row * weights_stride_0
        + output_offsets,
        selected_weights,
    )
    tl.store(
        ids_ptr
        + row * ids_stride_0
        + output_offsets,
        selected_ids.to(tl.int64),
    )


def fused_single_group_topk_router(
    scores: torch.Tensor,
    bias: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse single-group top-k routing after score computation."""
    assert num_experts == 192
    assert top_k == 8
    assert scores.dtype == torch.float32
    assert bias.dtype == torch.float32
    assert scores.shape[-1] == num_experts
    assert bias.numel() == num_experts

    scores = scores.contiguous()
    bias = bias.contiguous()
    batch_size = scores.shape[0]
    weights = torch.empty(
        batch_size,
        top_k,
        dtype=torch.float32,
        device=scores.device,
    )
    ids = torch.empty(
        batch_size,
        top_k,
        dtype=torch.int64,
        device=scores.device,
    )
    _single_group_topk_router_kernel[(batch_size,)](
        scores,
        bias,
        weights,
        ids,
        scores.stride(0),
        weights.stride(0),
        ids.stride(0),
        batch_size,
        routed_scaling_factor,
        NUM_EXPERTS=num_experts,
        TOP_K=top_k,
        BLOCK_SIZE=256,
        num_warps=8,
        num_stages=1,
    )
    return weights, ids

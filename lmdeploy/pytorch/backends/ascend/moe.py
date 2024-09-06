# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.pytorch.kernels.ascend import moe_gating_topk_softmax

from ..moe import (FusedMoEBuilder, FusedMoEImpl, SoftmaxTopKBuilder,
                   SoftmaxTopKImpl)


class AscendSoftmaxTopKImpl(SoftmaxTopKImpl):
    """ascend softmax topk implementation."""

    def __init__(self, top_k: int, dim: int = -1):
        self.top_k = top_k
        self.dim = dim

    def forward(self, x: torch.Tensor):
        routing_weights, selected_experts = moe_gating_topk_softmax(
            x, self.top_k)
        return routing_weights.to(torch.float32), selected_experts.to(
            torch.int64)


class AscendSoftmaxTopKBuilder(SoftmaxTopKBuilder):
    """ascend softmax topk implementation builder."""

    @staticmethod
    def build(top_k: int, dim: int = -1):
        """build."""
        return AscendSoftmaxTopKImpl(top_k, dim)


class AscendFusedMoEImpl(FusedMoEImpl):
    """ascend fused moe implementation."""

    def __init__(self, top_k: int, renormalize: bool = False):
        self.top_k = top_k
        self.renormalize = renormalize

    def forward(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor, gate_up_weights: torch.Tensor,
                down_weights: torch.Tensor):
        """forward."""
        seq_length = hidden_states.size(0)
        moe_output = torch.zeros_like(hidden_states)

        for i in range(seq_length):
            current_hidden_state = hidden_states[i]

            # faster than remove the for loop
            for j in range(self.top_k):
                expert_id = topk_ids[i][j]
                weight = topk_weights[i][j]

                up_weight = gate_up_weights[expert_id]
                up_proj = torch.matmul(up_weight, current_hidden_state)

                gate_cache, up_cache = up_proj.chunk(2, -1)
                gate_cache = torch.nn.functional.silu(gate_cache,
                                                      inplace=True) * up_cache

                down_weight = down_weights[expert_id]
                down_proj = torch.matmul(down_weight, gate_cache)

                moe_output[i] += weight * down_proj

        return moe_output


class AscendFusedMoEBuilder(FusedMoEBuilder):
    """ascend fused moe builder."""

    @staticmethod
    def build(top_k: int, renormalize: bool = False):
        """build from mlp."""
        return AscendFusedMoEImpl(top_k=top_k, renormalize=renormalize)

# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.pytorch.kernels.cuda import fused_moe

from ..moe import FusedMoEBuilder, FusedMoEImpl


class TritonFusedMoEImpl(FusedMoEImpl):
    """triton fused moe implementation."""

    def __init__(self, top_k: int, renormalize: bool = False):
        self.top_k = top_k
        self.renormalize = renormalize

    def update_weights(self, gate_up_weights: torch.Tensor,
                       down_weights: torch.Tensor):
        gate_up_weights = gate_up_weights.transpose(1,
                                                    2).contiguous().transpose(
                                                        1, 2)
        down_weights = down_weights.transpose(1,
                                              2).contiguous().transpose(1, 2)
        return gate_up_weights, down_weights

    def forward(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor, gate_up_weights: torch.Tensor,
                down_weights: torch.Tensor):
        """forward."""
        return fused_moe(hidden_states,
                         gate_up_weights,
                         down_weights,
                         topk_weights=topk_weights,
                         topk_ids=topk_ids,
                         topk=self.top_k,
                         renormalize=self.renormalize)


class TritonFusedMoEBuilder(FusedMoEBuilder):
    """triton fused moe builder."""

    @staticmethod
    def build(top_k: int, renormalize: bool = False):
        """build from mlp."""
        return TritonFusedMoEImpl(top_k=top_k, renormalize=renormalize)

# Copyright (c) OpenMMLab. All rights reserved.
import os
from dataclasses import dataclass
from typing import Callable, List

import torch

from lmdeploy.pytorch.kernels.dlinfer import MoeType, fused_moe, moe_gating_topk_softmax
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager

from ..moe import FusedMoEBuilder, FusedMoEImpl, SoftmaxTopKBuilder, SoftmaxTopKImpl


@dataclass
class MOEMetadata:
    max_tokens_across_dp: int = 1
    pad_size: int = 0
    dp_size: int = 1
    tp_size: int = 1
    ep_size: int = 1
    tp_rank: int = 0
    ep_rank: int = 0
    tp_group: torch.distributed.ProcessGroup = None
    ep_group: torch.distributed.ProcessGroup = None
    moe_type: MoeType = MoeType.UNDEFINED
    x_active_mask: torch.Tensor = None
    moe_group_name: str = None


class DlinferSoftmaxTopKImpl(SoftmaxTopKImpl):
    """Dlinfer softmax topk implementation."""

    def __init__(self, top_k: int, dim: int = -1, n_groups: int = -1):
        self.top_k = top_k
        self.dim = dim
        if n_groups != -1:
            raise NotImplementedError('Group router not supported')

    def forward(self, x: torch.Tensor):
        moe_metadata = get_step_ctx_manager().current_context().moe_metadata
        routing_weights, selected_experts = moe_gating_topk_softmax(x, self.top_k, moe_metadata.max_tokens_across_dp,
                                                                    moe_metadata.pad_size, moe_metadata.tp_size,
                                                                    moe_metadata.ep_size, moe_metadata.tp_rank)
        return routing_weights, selected_experts


class DlinferSoftmaxTopKBuilder(SoftmaxTopKBuilder):
    """Dlinfer softmax topk implementation builder."""

    @staticmethod
    def build(top_k: int, dim: int = -1, n_groups: int = -1):
        """build."""
        return DlinferSoftmaxTopKImpl(top_k, dim, n_groups)


class DlinferFusedMoEImpl(FusedMoEImpl):
    """Dlinfer fused moe implementation."""

    def __init__(self,
                 top_k: int,
                 num_experts: int,
                 renormalize: bool = False,
                 ep_size: int = 1,
                 ep_group: torch.distributed.ProcessGroup = None):
        self.top_k = top_k
        self.num_experts = num_experts
        self.renormalize = renormalize
        self.ep_size = ep_size
        self.ep_group = ep_group
        self.expert_ids_per_ep_rank = torch.tensor(
            [i % (self.num_experts // self.ep_size) for i in range(num_experts)],
            dtype=torch.int32,
            device=torch.npu.current_device(),
        )

    def update_weights(self, gate_up_weights: torch.Tensor, down_weights: torch.Tensor):
        """Update weights."""
        device_type = gate_up_weights.device.type
        if device_type in ['npu']:
            if os.getenv('DLINFER_RESET_MOE_UPDATE_WEIGHTS', '0') == '1':
                return gate_up_weights, down_weights
            return gate_up_weights.transpose(-1, -2).contiguous(), down_weights.transpose(-1, -2).contiguous()
        return gate_up_weights, down_weights

    def ep_expert_list(self, world_size: int, rank: int):
        """Experts list of current rank."""
        num_experts = self.num_experts
        expert_per_rank = (num_experts + world_size - 1) // world_size
        first_expert = rank * expert_per_rank
        last_expert = min(first_expert + expert_per_rank, num_experts)
        return list(range(first_expert, last_expert))

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                down_weights: torch.Tensor,
                gate_up_bias: torch.Tensor = None,
                down_bias: torch.Tensor = None,
                expert_list: List[int] = None,
                act_func: Callable = None):
        """forward."""
        assert gate_up_bias is None
        assert down_bias is None
        moe_metadata = get_step_ctx_manager().current_context().moe_metadata

        return fused_moe(hidden_states, gate_up_weights, down_weights, topk_weights, topk_ids, self.top_k,
                         self.renormalize, moe_metadata.pad_size, moe_metadata.tp_size, moe_metadata.ep_size,
                         moe_metadata.tp_rank, moe_metadata.ep_rank, moe_metadata.tp_group, moe_metadata.ep_group,
                         moe_metadata.moe_type, moe_metadata.x_active_mask, moe_metadata.moe_group_name,
                         self.expert_ids_per_ep_rank)


class DlinferFusedMoEBuilder(FusedMoEBuilder):
    """Dlinfer fused moe builder."""

    @staticmethod
    def build(top_k: int,
              num_experts: int,
              renormalize: bool = False,
              hidden_dim: int = 1,
              ep_size: int = 1,
              ep_group: torch.distributed.ProcessGroup = None,
              layer_idx: int = 0,
              out_dtype: torch.dtype = torch.bfloat16):
        """Build from mlp."""
        return DlinferFusedMoEImpl(top_k=top_k,
                                   num_experts=num_experts,
                                   renormalize=renormalize,
                                   ep_size=ep_size,
                                   ep_group=ep_group)

# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Callable

import torch

from lmdeploy.pytorch.kernels.dlinfer import (
    DlinferMoECommType,  # noqa: F401
    DlinferMoeMetadata,  # noqa: F401
    fused_moe,
    moe_gating_topk_softmax,
)
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager

from ..moe import FusedMoEBuildSpec, FusedMoEImpl, SoftmaxTopKImpl


class DlinferSoftmaxTopKImpl(SoftmaxTopKImpl):
    """Dlinfer softmax topk implementation."""

    def __init__(self, top_k: int, dim: int = -1, n_groups: int = -1):
        self.top_k = top_k
        self.dim = dim
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor):
        step_context = get_step_ctx_manager().current_context()
        moe_metadata = getattr(step_context, 'moe_metadata', None)
        if moe_metadata is not None:
            moe_metadata.router_n_groups = self.n_groups
        routing_weights, selected_experts = moe_gating_topk_softmax(x, self.top_k, moe_metadata)
        return routing_weights, selected_experts
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
        self.expert_ids_per_ep_rank = None
        self.chunked_moe_layout = None
        if self.ep_size > 1:
            self.expert_ids_per_ep_rank = torch.tensor(
                [i % (self.num_experts // self.ep_size) for i in range(num_experts)],
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )

    def ep_expert_list(self, world_size: int, rank: int):
        """Experts list of current rank."""
        num_experts = self.num_experts
        expert_per_rank = (num_experts + world_size - 1) // world_size
        first_expert = rank * expert_per_rank
        last_expert = min(first_expert + expert_per_rank, num_experts)
        return list(range(first_expert, last_expert))

    def build_expert_storage(self, num_experts: int):
        """Build backend-specific physical expert storage."""
        from dlinfer.vendor import vendor_name

        if vendor_name == 'ascend':
            from dlinfer.vendor.ascend.moe import build_chunked_moe_storage_layout, chunked_moe_storage_expert_id
            storage_num_experts, self.chunked_moe_layout = build_chunked_moe_storage_layout(num_experts)
            if self.chunked_moe_layout is not None:

                def expert_id_mapper(expert_id: int):
                    return chunked_moe_storage_expert_id(expert_id, self.chunked_moe_layout)

                return storage_num_experts, expert_id_mapper
        return num_experts, None

    def update_weights(self, gate_up_weights: torch.Tensor, down_weights: torch.Tensor):
        """Update backend-specific chunked weight layout."""
        from dlinfer.vendor import vendor_name

        if vendor_name == 'ascend':
            if self.chunked_moe_layout is not None and self.chunked_moe_layout.packed:
                from dlinfer.vendor.ascend.moe import zero_chunked_moe_weight_padding
                zero_chunked_moe_weight_padding(gate_up_weights, down_weights, self.chunked_moe_layout)

        return gate_up_weights, down_weights

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                down_weights: torch.Tensor,
                gate_up_bias: torch.Tensor = None,
                down_bias: torch.Tensor = None,
                expert_list: list[int] = None,
                act_func: Callable = None):
        """forward."""
        assert gate_up_bias is None
        assert down_bias is None

        step_context = get_step_ctx_manager().current_context()
        moe_metadata = getattr(step_context, 'moe_metadata', None)
        if moe_metadata is not None:
            moe_metadata.expert_ids_per_ep_rank = self.expert_ids_per_ep_rank
        return fused_moe(hidden_states, gate_up_weights, down_weights, topk_weights, topk_ids, self.top_k,
                         self.renormalize, moe_metadata, self.chunked_moe_layout)


def _build_fused_moe(spec: FusedMoEBuildSpec) -> FusedMoEImpl:
    """Build a DLINFER fused MoE implementation."""
    return DlinferFusedMoEImpl(
        top_k=spec.top_k,
        num_experts=spec.num_experts,
        renormalize=spec.renormalize,
        ep_size=spec.ep_size,
        ep_group=spec.ep_group,
    )

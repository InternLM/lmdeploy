# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Callable

import torch

from lmdeploy.pytorch.kernels.dlinfer import (
    DlinferMoECommType,  # noqa: F401
    DlinferMoeMetadata,  # noqa: F401
    fused_moe,
    fused_moe_w8a8,
    moe_gating_topk_softmax,
)
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager

from ..moe import (
    FusedMoEBuilder,
    FusedMoEImpl,
    FusedMoEW8A8Builder,
    FusedMoEW8A8Impl,
    SoftmaxTopKBuilder,
    SoftmaxTopKImpl,
)


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
              out_dtype: torch.dtype = torch.bfloat16,
              num_max_dispatch_tokens_per_rank: int = 128):
        """Build from mlp."""
        return DlinferFusedMoEImpl(top_k=top_k,
                                   num_experts=num_experts,
                                   renormalize=renormalize,
                                   ep_size=ep_size,
                                   ep_group=ep_group)


class DlinferFusedMoEW8A8Impl(DlinferFusedMoEImpl, FusedMoEW8A8Impl):
    """Ascend non-fused dynamic W8A8 MoE implementation."""

    def __init__(self,
                 top_k: int,
                 num_experts: int,
                 renormalize: bool = False,
                 ep_size: int = 1,
                 ep_group: torch.distributed.ProcessGroup = None,
                 out_dtype: torch.dtype = torch.bfloat16,
                 quant_dtype: torch.dtype = torch.int8):
        DlinferFusedMoEImpl.__init__(self, top_k, num_experts, renormalize, ep_size, ep_group)
        if quant_dtype != torch.int8:
            raise ValueError(
                f'Ascend W8A8 MoE requires torch.int8, got {quant_dtype}')
        self.out_dtype = out_dtype
        self.quant_dtype = quant_dtype

    def update_weights(self, gate_up_weights: torch.Tensor, down_weights: torch.Tensor, gate_up_scale: torch.Tensor,
                       down_scale: torch.Tensor):
        """Keep OI weights and satisfy Ascend quant-GMM scale dtype rules."""
        scale_dtype = torch.bfloat16 if self.out_dtype == torch.bfloat16 else torch.float32
        gate_up_scale = gate_up_scale.to(scale_dtype)
        down_scale = down_scale.to(scale_dtype)
        return gate_up_weights, down_weights, gate_up_scale, down_scale

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                gate_up_scale: torch.Tensor,
                down_weights: torch.Tensor,
                down_scale: torch.Tensor,
                expert_list: list[int] = None):
        """Forward through dlinfer's public-op W8A8 fallback."""
        step_context = get_step_ctx_manager().current_context()
        moe_metadata = getattr(step_context, 'moe_metadata', None)
        if moe_metadata is None:
            raise RuntimeError('Dlinfer W8A8 MoE requires moe_metadata in the current step context.')
        moe_metadata.expert_ids_per_ep_rank = self.expert_ids_per_ep_rank
        return fused_moe_w8a8(hidden_states, gate_up_weights, gate_up_scale, down_weights, down_scale, topk_weights,
                              topk_ids, self.top_k, self.renormalize, moe_metadata)


class DlinferFusedMoEW8A8Builder(FusedMoEW8A8Builder):
    """Builder for the Ascend public-op W8A8 MoE fallback."""

    @staticmethod
    def build(top_k: int,
              num_experts: int,
              renormalize: bool = False,
              out_dtype: torch.dtype = torch.bfloat16,
              quant_dtype: torch.dtype = torch.int8,
              hidden_dim: int = 1,
              ep_size: int = 1,
              ep_group: torch.distributed.ProcessGroup = None,
              layer_idx: int = 0):
        return DlinferFusedMoEW8A8Impl(top_k=top_k,
                                       num_experts=num_experts,
                                       renormalize=renormalize,
                                       ep_size=ep_size,
                                       ep_group=ep_group,
                                       out_dtype=out_dtype,
                                       quant_dtype=quant_dtype)

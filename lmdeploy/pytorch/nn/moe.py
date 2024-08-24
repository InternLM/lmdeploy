# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from torch import nn

from ..backends import LayerType, get_backend


def _merge_mlp(gates: List[torch.Tensor], ups: List[torch.Tensor],
               downs: List[torch.Tensor]):
    """merge experts."""
    num_experts = len(gates)

    def __get_meta():
        gate = gates[0]
        down = downs[0]
        ffn_dim = gate.weight.size(0)
        hidden_dim = down.weight.size(0)
        dtype = gate.weight.dtype
        device = gate.weight.device
        return ffn_dim, hidden_dim, dtype, device

    def __copy_assign_param(param, weight):
        """copy assign."""
        weight.copy_(param.data)
        param.data = weight

    ffn_dim, hidden_dim, dtype, device = __get_meta()

    gate_up_weights = torch.empty(num_experts,
                                  ffn_dim * 2,
                                  hidden_dim,
                                  device=device,
                                  dtype=dtype)
    down_weights = torch.empty(num_experts,
                               hidden_dim,
                               ffn_dim,
                               device=device,
                               dtype=dtype)
    for exp_id in range(num_experts):
        gate = gates[exp_id]
        up = ups[exp_id]
        down = downs[exp_id]
        __copy_assign_param(gate.weight, gate_up_weights[exp_id, :ffn_dim])
        __copy_assign_param(up.weight, gate_up_weights[exp_id, ffn_dim:])
        __copy_assign_param(down.weight, down_weights[exp_id])

    torch.cuda.empty_cache()
    return gate_up_weights, down_weights


class SoftmaxTopK(nn.Module):
    """softmax topk."""

    def __init__(self, top_k: int, dim: int = -1):
        super().__init__()
        self.top_k = top_k
        impl_builder = get_backend().get_layer_impl_builder(
            LayerType.SoftmaxTopK)
        self.impl = impl_builder.build(top_k, dim)

    def forward(self, x: torch.Tensor):
        """forward."""
        return self.impl.forward(x)


class FusedMoE(nn.Module):
    """fused moe."""

    def __init__(
        self,
        gate_up_weights: torch.Tensor,
        down_weights: torch.Tensor,
        top_k: int,
        renormalize: bool = False,
    ):
        super().__init__()
        impl_builder = get_backend().get_layer_impl_builder(LayerType.FusedMoE)
        self.impl = impl_builder.build(top_k, renormalize)
        gate_up_weights, down_weights = self.impl.update_weights(
            gate_up_weights, down_weights)
        gate_up_weights = torch.nn.Parameter(gate_up_weights,
                                             requires_grad=False)
        down_weights = torch.nn.Parameter(down_weights, requires_grad=False)
        self.register_parameter('gate_up_weights', gate_up_weights)
        self.register_parameter('down_weights', down_weights)

    def forward(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor):
        return self.impl.forward(hidden_states, topk_weights, topk_ids,
                                 self.gate_up_weights, self.down_weights)


def build_moe_from_mlp(
    gates: List[nn.Linear],
    ups: List[nn.Linear],
    downs: List[nn.Linear],
    top_k: int,
    renormalize: bool = False,
):
    """build moe from mlp."""

    gate_up_weights, down_weights = _merge_mlp(gates, ups, downs)
    return FusedMoE(gate_up_weights,
                    down_weights,
                    top_k=top_k,
                    renormalize=renormalize)

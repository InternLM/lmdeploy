# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from ..backends import OpType, get_backend
from .utils import get_world_rank


class SoftmaxTopK(nn.Module):
    """softmax topk."""

    def __init__(self, top_k: int, dim: int = -1):
        super().__init__()
        self.top_k = top_k
        impl_builder = get_backend().get_layer_impl_builder(OpType.SoftmaxTopK)
        self.impl = impl_builder.build(top_k, dim)

    def forward(self, x: torch.Tensor):
        """forward."""
        return self.impl.forward(x)


class FusedMoE(nn.Module):
    """fused moe."""

    def __init__(self,
                 hidden_dim: int,
                 ffn_dim: int,
                 num_experts: int,
                 top_k: int,
                 renormalize: bool = False,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 all_reduce: bool = True):
        super().__init__()
        if device is None:
            device = torch.device('cpu')
        if dtype is None:
            dtype = torch.float16
        hidden_dim, ffn_dim = self._update_args(hidden_dim, ffn_dim)

        impl_builder = get_backend().get_layer_impl_builder(OpType.FusedMoE)
        self.impl = impl_builder.build(top_k, renormalize)

        gate_up_weights, down_weights = self.create_weights(hidden_dim,
                                                            ffn_dim,
                                                            num_experts,
                                                            dtype=dtype,
                                                            device=device)
        gate_up_weights = torch.nn.Parameter(gate_up_weights,
                                             requires_grad=False)
        down_weights = torch.nn.Parameter(down_weights, requires_grad=False)
        gate_up_weights.weight_loader = self.weight_loader
        down_weights.weight_loader = self.weight_loader
        gate_up_weights._weight_type = 'gate_up_weights'
        down_weights._weight_type = 'down_weights'
        self.register_parameter('gate_up_weights', gate_up_weights)
        self.register_parameter('down_weights', down_weights)

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.dtype = dtype
        self.device = device
        world_size, _ = get_world_rank()
        if world_size == 1:
            all_reduce = False
        self.all_reduce = all_reduce

    def _update_args(self, hidden_dim: int, ffn_dim: int):
        """update args."""
        world_size, _ = get_world_rank()
        assert ffn_dim % world_size == 0
        ffn_dim = ffn_dim // world_size
        return hidden_dim, ffn_dim

    def create_weights(self, hidden_dim: int, ffn_dim: int, num_experts: int,
                       dtype: torch.dtype, device: torch.device):
        """create weights."""
        gate_up_weights = torch.empty((num_experts, ffn_dim * 2, hidden_dim),
                                      dtype=dtype,
                                      device=device)
        down_weights = torch.empty((num_experts, hidden_dim, ffn_dim),
                                   dtype=dtype,
                                   device=device)
        return gate_up_weights, down_weights

    def update_weights(self):
        """update weights."""
        gate_up_weights, down_weights = self.impl.update_weights(
            self.gate_up_weights, self.down_weights)
        gate_up_weights = torch.nn.Parameter(gate_up_weights,
                                             requires_grad=False)
        down_weights = torch.nn.Parameter(down_weights, requires_grad=False)
        gate_up_weights.weight_loader = self.weight_loader
        down_weights.weight_loader = self.weight_loader
        gate_up_weights._weight_type = 'gate_up_weights'
        down_weights._weight_type = 'down_weights'
        self.register_parameter('gate_up_weights', gate_up_weights)
        self.register_parameter('down_weights', down_weights)

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, expert_id: int,
                      shard_id: str):
        """weight loader."""
        world_size, rank = get_world_rank()
        if shard_id == 'gate':
            param_data = param.data[expert_id, :self.ffn_dim]
            weight = loaded_weight.chunk(world_size, dim=0)[rank]
        elif shard_id == 'up':
            param_data = param.data[expert_id, self.ffn_dim:]
            weight = loaded_weight.chunk(world_size, dim=0)[rank]
        elif shard_id == 'down':
            param_data = param.data[expert_id]
            weight = loaded_weight.chunk(world_size, dim=1)[rank]
        else:
            raise RuntimeError(f'Unknown shard_id: {shard_id}')
        param_data.copy_(weight)

    def forward(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor):
        ret = self.impl.forward(hidden_states, topk_weights, topk_ids,
                                self.gate_up_weights, self.down_weights)
        if self.all_reduce:
            dist.all_reduce(ret)
        return ret

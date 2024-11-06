# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from lmdeploy.pytorch.distributed import get_world_rank

from ..backends import OpType, get_backend


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
                 all_reduce: bool = True,
                 enable_ep: bool = False):
        super().__init__()
        if device is None:
            device = torch.device('cpu')
        if dtype is None:
            dtype = torch.float16

        impl_builder = get_backend().get_layer_impl_builder(OpType.FusedMoE)
        self.impl = impl_builder.build(top_k, num_experts, renormalize)

        self.expert_list = None
        self.expert_map = None
        enable_ep = enable_ep and self.impl.support_ep()
        if enable_ep:
            world_size, rank = get_world_rank()
            expert_list = self.impl.ep_expert_list(world_size, rank)
            self.expert_list = expert_list
            self.expert_map = dict(
                (eid, idx) for idx, eid in enumerate(expert_list))
            num_experts = len(expert_list)
            gate_up_weights, down_weights = self.create_weights(hidden_dim,
                                                                ffn_dim,
                                                                num_experts,
                                                                dtype=dtype,
                                                                device=device)
        else:
            hidden_dim, ffn_dim = self._update_args(hidden_dim, ffn_dim)
            gate_up_weights, down_weights = self.create_weights(hidden_dim,
                                                                ffn_dim,
                                                                num_experts,
                                                                dtype=dtype,
                                                                device=device)
        gate_up_weights = torch.nn.Parameter(gate_up_weights,
                                             requires_grad=False)
        down_weights = torch.nn.Parameter(down_weights, requires_grad=False)
        gate_up_weights._weight_type = 'gate_up_weights'
        down_weights._weight_type = 'down_weights'
        self.register_parameter('gate_up_weights', gate_up_weights)
        self.register_parameter('down_weights', down_weights)

        if enable_ep:
            gate_up_weights.weight_loader = self.weight_loader_ep
            down_weights.weight_loader = self.weight_loader_ep
        else:
            gate_up_weights.weight_loader = self.weight_loader_tp
            down_weights.weight_loader = self.weight_loader_tp

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
        gateup_loader = self.gate_up_weights.weight_loader
        down_loader = self.down_weights.weight_loader
        gate_up_weights, down_weights = self.impl.update_weights(
            self.gate_up_weights, self.down_weights)
        gate_up_weights = torch.nn.Parameter(gate_up_weights,
                                             requires_grad=False)
        down_weights = torch.nn.Parameter(down_weights, requires_grad=False)
        gate_up_weights.weight_loader = gateup_loader
        down_weights.weight_loader = down_loader
        gate_up_weights._weight_type = 'gate_up_weights'
        down_weights._weight_type = 'down_weights'
        self.register_parameter('gate_up_weights', gate_up_weights)
        self.register_parameter('down_weights', down_weights)

    def weight_loader_tp(self, param: torch.nn.Parameter,
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

    def weight_loader_ep(self, param: torch.nn.Parameter,
                         loaded_weight: torch.Tensor, expert_id: int,
                         shard_id: str):
        """weight loader."""
        expert_list = self.expert_list
        if expert_id not in expert_list:
            return

        expert_map = self.expert_map
        param_id = expert_map[expert_id]
        if shard_id == 'gate':
            param_data = param.data[param_id, :self.ffn_dim]
        elif shard_id == 'up':
            param_data = param.data[param_id, self.ffn_dim:]
        elif shard_id == 'down':
            param_data = param.data[param_id]
        else:
            raise RuntimeError(f'Unknown shard_id: {shard_id}')
        param_data.copy_(loaded_weight)

    def forward(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor):
        ret = self.impl.forward(hidden_states, topk_weights, topk_ids,
                                self.gate_up_weights, self.down_weights,
                                self.expert_list)
        if self.all_reduce:
            dist.all_reduce(ret)
        return ret

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import torch

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.distributed import get_tp_world_rank

from .base import FusedMoEBase, MoeType, moe_gather_inputs, moe_reduce, update_dims
from .default import LinearWeights


class LinearWeightsW8A8(LinearWeights):
    """Fused moe linear w8a8 weights."""

    def __init__(self,
                 num_experts: int,
                 in_features: int,
                 out_features: int,
                 weight_type: str,
                 device: torch.device,
                 expert_list: List[int] = None,
                 quant_dtype: torch.dtype = torch.int8):
        super().__init__(
            num_experts=num_experts,
            in_features=in_features,
            out_features=out_features,
            weight_type=weight_type,
            dtype=quant_dtype,
            device=device,
            expert_list=expert_list,
        )
        scale = torch.empty((num_experts, out_features, 1), dtype=torch.float32, device=device)
        scale = torch.nn.Parameter(scale, requires_grad=False)
        self.register_parameter('scale', scale)

        if self.ep:
            self.scale.weight_loader = self.weight_loader_ep
        else:
            self.scale.weight_loader = self.weight_loader_scale_tp

    def update_weight(self, weight: torch.Tensor, scale: torch.Tensor):
        """Update weight."""
        super().update_weight(weight=weight)
        weight_loader = self.scale.weight_loader
        scale = torch.nn.Parameter(scale, requires_grad=False)
        scale.weight_loader = weight_loader
        self.register_parameter('scale', scale)

    def weight_loader_scale_tp(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int,
                               shard_id: str):
        """Weight loader scale tp."""
        world_size, rank = get_tp_world_rank('moe')
        if shard_id == 'gate':
            param_data = param.data[expert_id, :self.half_out]
            weight = loaded_weight.chunk(world_size, dim=0)[rank]
        elif shard_id == 'up':
            param_data = param.data[expert_id, self.half_out:]
            weight = loaded_weight.chunk(world_size, dim=0)[rank]
        elif shard_id == 'down':
            param_data = param.data[expert_id]
            weight = loaded_weight
        else:
            raise RuntimeError(f'Unknown shard_id: {shard_id}')
        weight = weight.to(param.dtype)
        param_data.copy_(weight)


class FusedMoEW8A8(FusedMoEBase):
    """Fused moe w8a8."""

    def __init__(self,
                 hidden_dim: int,
                 ffn_dim: int,
                 num_experts: int,
                 top_k: int,
                 renormalize: bool = False,
                 dtype: Optional[torch.dtype] = None,
                 quant_dtype: Optional[torch.dtype] = torch.int8,
                 device: Optional[torch.device] = None,
                 all_reduce: bool = True):

        device = device or torch.device('cpu')
        dtype = dtype or torch.float16
        # init distributed tp arguments
        self.init_dist_args(all_reduce)

        # check ep
        if self.ep > 1:
            raise RuntimeError('FusedMoEW8A8 does not support EP mode now.')

        super().__init__(
            tp=self.tp,
            tp_mode=self.tp_mode,
            do_renormalize=renormalize,
        )

        # create implementation
        impl_builder = get_backend().get_layer_impl_builder(OpType.FusedMoEW8A8)
        self.impl = impl_builder.build(top_k, num_experts, renormalize, dtype, quant_dtype=quant_dtype)

        # create weights
        hidden_dim, ffn_dim = update_dims(hidden_dim, ffn_dim)
        expert_list = None
        self.expert_list = expert_list
        self.gate_up = LinearWeightsW8A8(num_experts,
                                         hidden_dim,
                                         ffn_dim * 2,
                                         weight_type='gate_up',
                                         device=device,
                                         expert_list=expert_list,
                                         quant_dtype=quant_dtype)
        self.down = LinearWeightsW8A8(num_experts,
                                      ffn_dim,
                                      hidden_dim,
                                      weight_type='down',
                                      device=device,
                                      expert_list=expert_list,
                                      quant_dtype=quant_dtype)

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.dtype = dtype
        self.device = device
        self.all_reduce = all_reduce

    def update_weights(self):
        """Update weights."""
        (gate_up_weights, down_weights, gate_up_scale,
         down_scale) = self.impl.update_weights(self.gate_up.weight, self.down.weight, self.gate_up.scale,
                                                self.down.scale)
        self.gate_up.update_weight(gate_up_weights, gate_up_scale)
        self.down.update_weight(down_weights, down_scale)

    def dispatch(self, state: Dict):
        """dispatch."""
        moe_type = state['moe_type']
        if moe_type == MoeType.Default:
            hidden_states, topk_weights, topk_idx = moe_gather_inputs(state['hidden_states'],
                                                                      state['topk_weights'],
                                                                      state['topk_idx'],
                                                                      group=self.gather_group)
            recv_state = {
                'hidden_states': hidden_states,
                'topk_idx': topk_idx,
                'topk_weights': topk_weights,
                'moe_type': moe_type
            }
        else:
            raise NotImplementedError(f'Not supported moe type: {moe_type}')
        return recv_state

    def gemm(self, state: Dict):
        """gemm."""
        hidden_states = state['hidden_states']
        topk_weights = state['topk_weights']
        topk_ids = state['topk_idx']

        ret = self.impl.forward(hidden_states, topk_weights, topk_ids, self.gate_up.weight, self.gate_up.scale,
                                self.down.weight, self.down.scale, self.expert_list)
        return dict(hidden_states=ret, moe_type=state['moe_type'])

    def combine(self, state: Dict):
        """combine."""
        moe_type = state['moe_type']
        if moe_type == MoeType.Default:
            if self.all_reduce:
                state['hidden_states'] = moe_reduce(state['hidden_states'],
                                                    rank=self.tp_rank,
                                                    tp_mode=self.tp_mode,
                                                    group=self.tp_group)
            out_state = {'hidden_states': state['hidden_states'], 'moe_type': moe_type}
        else:
            raise NotImplementedError(f'Not supported moe type: {moe_type}')
        return out_state

    def wait(self, state: Dict):
        """wait."""
        raise NotImplementedError

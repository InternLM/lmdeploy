# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional

import torch
import torch.distributed as dist
from torch import nn

from lmdeploy.pytorch.distributed import get_world_rank

from ..backends import OpType, get_backend
from .utils import div_up


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


def create_mlp_weights(hidden_dim: int, ffn_dim: int, num_experts: int, dtype: torch.dtype, device: torch.device):
    """create weights."""
    gate_up_weights = torch.empty((num_experts, ffn_dim * 2, hidden_dim), dtype=dtype, device=device)
    down_weights = torch.empty((num_experts, hidden_dim, ffn_dim), dtype=dtype, device=device)
    return gate_up_weights, down_weights


def _update_args(hidden_dim: int, ffn_dim: int):
    """update args."""
    world_size, _ = get_world_rank()
    assert ffn_dim % world_size == 0
    ffn_dim = ffn_dim // world_size
    return hidden_dim, ffn_dim


class LinearWeights(nn.Module):
    """fused moe linear weights."""

    def __init__(self,
                 num_experts: int,
                 in_features: int,
                 out_features: int,
                 weight_type: str,
                 dtype: torch.dtype,
                 device: torch.device,
                 expert_list: List[int] = None,
                 ep: bool = False):
        super().__init__()
        weight = torch.empty((num_experts, out_features, in_features), dtype=dtype, device=device)
        weight = torch.nn.Parameter(weight, requires_grad=False)
        self.register_parameter('weight', weight)
        self.ep = ep
        self.expert_list = expert_list
        self.weight_type = weight_type
        self.half_out = out_features // 2

        if self.ep:
            self.expert_map = dict((eid, idx) for idx, eid in enumerate(expert_list))
            self.weight.weight_loader = self.weight_loader_ep
        else:
            self.weight.weight_loader = self.weight_loader_tp

    def update_weight(self, weight: torch.Tensor):
        """update weight."""
        weight_loader = self.weight.weight_loader
        weight = torch.nn.Parameter(weight, requires_grad=False)
        weight.weight_loader = weight_loader
        self.register_parameter('weight', weight)

    def weight_loader_tp(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int, shard_id: str):
        """weight loader."""
        world_size, rank = get_world_rank()
        if shard_id == 'gate':
            param_data = param.data[expert_id, :self.half_out]
            weight = loaded_weight.chunk(world_size, dim=0)[rank]
        elif shard_id == 'up':
            param_data = param.data[expert_id, self.half_out:]
            weight = loaded_weight.chunk(world_size, dim=0)[rank]
        elif shard_id == 'down':
            param_data = param.data[expert_id]
            # weight is not contiguous, chunk and copy in cpu is slow
            weight = loaded_weight.to(param_data.device)
            weight = weight.chunk(world_size, dim=1)[rank]
        else:
            raise RuntimeError(f'Unknown shard_id: {shard_id}')
        param_data.copy_(weight)

    def weight_loader_ep(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int, shard_id: str):
        """weight loader."""
        expert_list = self.expert_list
        if expert_id not in expert_list:
            return

        expert_map = self.expert_map
        param_id = expert_map[expert_id]
        if shard_id == 'gate':
            param_data = param.data[param_id, :self.half_out]
        elif shard_id == 'up':
            param_data = param.data[param_id, self.half_out:]
        elif shard_id == 'down':
            param_data = param.data[param_id]
        else:
            raise RuntimeError(f'Unknown shard_id: {shard_id}')
        param_data.copy_(loaded_weight)


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

        enable_ep = enable_ep and self.impl.support_ep()
        if enable_ep:
            world_size, rank = get_world_rank()
            expert_list = self.impl.ep_expert_list(world_size, rank)
            num_experts = len(expert_list)
        else:
            hidden_dim, ffn_dim = _update_args(hidden_dim, ffn_dim)
            expert_list = None
        self.expert_list = expert_list
        self.gate_up = LinearWeights(num_experts,
                                     hidden_dim,
                                     ffn_dim * 2,
                                     weight_type='gate_up',
                                     dtype=dtype,
                                     device=device,
                                     expert_list=expert_list,
                                     ep=enable_ep)
        self.down = LinearWeights(
            num_experts,
            ffn_dim,
            hidden_dim,
            weight_type='down',
            dtype=dtype,
            device=device,
            expert_list=expert_list,
            ep=enable_ep,
        )

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.dtype = dtype
        self.device = device
        world_size, _ = get_world_rank()
        if world_size == 1:
            all_reduce = False
        self.all_reduce = all_reduce

    def update_weights(self):
        """update weights."""
        gate_up_weights, down_weights = self.impl.update_weights(self.gate_up.weight, self.down.weight)
        self.gate_up.update_weight(gate_up_weights)
        self.down.update_weight(down_weights)

    def forward(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.LongTensor):
        ret = self.impl.forward(hidden_states, topk_weights, topk_ids, self.gate_up.weight, self.down.weight,
                                self.expert_list)
        if self.all_reduce:
            dist.all_reduce(ret)
        return ret


class LinearWeightsW8A8(LinearWeights):
    """fused moe linear w8a8 weights."""

    def __init__(self,
                 num_experts: int,
                 in_features: int,
                 out_features: int,
                 weight_type: str,
                 device: torch.device,
                 expert_list: List[int] = None,
                 ep: bool = False,
                 quant_dtype: torch.dtype = torch.int8):
        super().__init__(
            num_experts=num_experts,
            in_features=in_features,
            out_features=out_features,
            weight_type=weight_type,
            dtype=quant_dtype,
            device=device,
            expert_list=expert_list,
            ep=ep,
        )
        scale = torch.empty((num_experts, out_features, 1), dtype=torch.float32, device=device)
        scale = torch.nn.Parameter(scale, requires_grad=False)
        self.register_parameter('scale', scale)

        if self.ep:
            self.scale.weight_loader = self.weight_loader_ep
        else:
            self.scale.weight_loader = self.weight_loader_scale_tp

    def update_weight(self, weight: torch.Tensor, scale: torch.Tensor):
        """update weight."""
        super().update_weight(weight=weight)
        weight_loader = self.scale.weight_loader
        scale = torch.nn.Parameter(scale, requires_grad=False)
        scale.weight_loader = weight_loader
        self.register_parameter('scale', scale)

    def weight_loader_scale_tp(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int,
                               shard_id: str):
        """weight loader scale tp."""
        world_size, rank = get_world_rank()
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
        param_data.copy_(weight)


class FusedMoEW8A8(nn.Module):
    """fused moe w8a8."""

    def __init__(self,
                 hidden_dim: int,
                 ffn_dim: int,
                 num_experts: int,
                 top_k: int,
                 renormalize: bool = False,
                 dtype: Optional[torch.dtype] = None,
                 quant_dtype: Optional[torch.dtype] = torch.int8,
                 device: Optional[torch.device] = None,
                 all_reduce: bool = True,
                 enable_ep: bool = False):
        super().__init__()

        if device is None:
            device = torch.device('cpu')
        dtype = torch.float16 if dtype is None else dtype

        impl_builder = get_backend().get_layer_impl_builder(OpType.FusedMoEW8A8)
        self.impl = impl_builder.build(top_k, num_experts, renormalize, dtype, quant_dtype=quant_dtype)

        enable_ep = enable_ep and self.impl.support_ep()
        if enable_ep:
            world_size, rank = get_world_rank()
            expert_list = self.impl.ep_expert_list(world_size, rank)
            num_experts = len(expert_list)
        else:
            hidden_dim, ffn_dim = _update_args(hidden_dim, ffn_dim)
            expert_list = None
        self.expert_list = expert_list

        self.gate_up = LinearWeightsW8A8(num_experts,
                                         hidden_dim,
                                         ffn_dim * 2,
                                         weight_type='gate_up',
                                         device=device,
                                         expert_list=expert_list,
                                         ep=enable_ep,
                                         quant_dtype=quant_dtype)
        self.down = LinearWeightsW8A8(num_experts,
                                      ffn_dim,
                                      hidden_dim,
                                      weight_type='down',
                                      device=device,
                                      expert_list=expert_list,
                                      ep=enable_ep,
                                      quant_dtype=quant_dtype)

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.dtype = dtype
        self.device = device
        world_size, _ = get_world_rank()
        if world_size == 1:
            all_reduce = False
        self.all_reduce = all_reduce

    def update_weights(self):
        """update weights."""
        (gate_up_weights, down_weights, gate_up_scale,
         down_scale) = self.impl.update_weights(self.gate_up.weight, self.down.weight, self.gate_up.scale,
                                                self.down.scale)
        self.gate_up.update_weight(gate_up_weights, gate_up_scale)
        self.down.update_weight(down_weights, down_scale)

    def forward(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.LongTensor):
        ret = self.impl.forward(hidden_states, topk_weights, topk_ids, self.gate_up.weight, self.gate_up.scale,
                                self.down.weight, self.down.scale, self.expert_list)
        if self.all_reduce:
            dist.all_reduce(ret)
        return ret


class LinearWeightsBlockedF8(LinearWeights):
    """fused moe linear blocked fp8 weights."""

    def __init__(self,
                 num_experts: int,
                 in_features: int,
                 out_features: int,
                 weight_type: str,
                 block_size: int,
                 dtype: torch.dtype,
                 device: torch.device,
                 expert_list: List[int] = None,
                 ep: bool = False):
        super().__init__(
            num_experts=num_experts,
            in_features=in_features,
            out_features=out_features,
            weight_type=weight_type,
            dtype=dtype,
            device=device,
            expert_list=expert_list,
            ep=ep,
        )
        self.block_size = block_size
        scale = torch.empty((num_experts, div_up(out_features, block_size), div_up(in_features, block_size)),
                            dtype=torch.float32,
                            device=device)
        scale = torch.nn.Parameter(scale, requires_grad=False)
        self.register_parameter('scale', scale)

        if self.ep:
            self.scale.weight_loader = self.weight_loader_ep
        else:
            self.scale.weight_loader = self.weight_loader_scale_tp

    def update_weight(self, weight: torch.Tensor, scale: torch.Tensor):
        """update weight."""
        super().update_weight(weight=weight)
        weight_loader = self.scale.weight_loader
        scale = torch.nn.Parameter(scale, requires_grad=False)
        scale.weight_loader = weight_loader
        self.register_parameter('scale', scale)

    def weight_loader_scale_tp(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int,
                               shard_id: str):
        """weight loader scale tp."""
        world_size, rank = get_world_rank()
        block_size = self.block_size
        half_out = self.half_out // block_size
        if shard_id == 'gate':
            param_data = param.data[expert_id, :half_out]
            weight = loaded_weight.chunk(world_size, dim=0)[rank]
        elif shard_id == 'up':
            param_data = param.data[expert_id, half_out:]
            weight = loaded_weight.chunk(world_size, dim=0)[rank]
        elif shard_id == 'down':
            param_data = param.data[expert_id]
            loaded_weight = loaded_weight.to(param_data.device)
            weight = loaded_weight.chunk(world_size, dim=1)[rank]
        else:
            raise RuntimeError(f'Unknown shard_id: {shard_id}')
        param_data.copy_(weight)


class FusedMoEBlockedF8(nn.Module):
    """fused moe blocked f8."""

    def __init__(self,
                 hidden_dim: int,
                 ffn_dim: int,
                 num_experts: int,
                 top_k: int,
                 renormalize: bool = False,
                 fp8_dtype: torch.dtype = torch.float8_e4m3fn,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 all_reduce: bool = True,
                 enable_ep: bool = False):
        super().__init__()
        if device is None:
            device = torch.device('cpu')
        dtype = torch.float16 if dtype is None else dtype
        self.block_size = 128
        impl_builder = get_backend().get_layer_impl_builder(OpType.FusedMoEBlockedF8)
        self.impl = impl_builder.build(top_k, num_experts, renormalize, block_size=self.block_size, out_dtype=dtype)

        enable_ep = enable_ep and self.impl.support_ep()
        if enable_ep:
            world_size, rank = get_world_rank()
            expert_list = self.impl.ep_expert_list(world_size, rank)
            num_experts = len(expert_list)
        else:
            hidden_dim, ffn_dim = _update_args(hidden_dim, ffn_dim)
            expert_list = None
        self.expert_list = expert_list

        self.gate_up = LinearWeightsBlockedF8(num_experts,
                                              hidden_dim,
                                              ffn_dim * 2,
                                              weight_type='gate_up',
                                              block_size=self.block_size,
                                              dtype=fp8_dtype,
                                              device=device,
                                              expert_list=expert_list,
                                              ep=enable_ep)
        self.down = LinearWeightsBlockedF8(
            num_experts,
            ffn_dim,
            hidden_dim,
            weight_type='down',
            block_size=self.block_size,
            dtype=fp8_dtype,
            device=device,
            expert_list=expert_list,
            ep=enable_ep,
        )

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.dtype = dtype
        self.device = device
        world_size, _ = get_world_rank()
        if world_size == 1:
            all_reduce = False
        self.all_reduce = all_reduce

    def update_weights(self):
        """update weights."""
        (gate_up_weights, down_weights, gate_up_scale,
         down_scale) = self.impl.update_weights(self.gate_up.weight, self.down.weight, self.gate_up.scale,
                                                self.down.scale)
        self.gate_up.update_weight(gate_up_weights, gate_up_scale)
        self.down.update_weight(down_weights, down_scale)

    def forward(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.LongTensor):
        ret = self.impl.forward(hidden_states, topk_weights, topk_ids, self.gate_up.weight, self.gate_up.scale,
                                self.down.weight, self.down.scale, self.expert_list)
        if self.all_reduce:
            dist.all_reduce(ret)
        return ret


def build_fused_moe(
    hidden_dim: int,
    ffn_dim: int,
    num_experts: int,
    top_k: int,
    renormalize: bool = False,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    all_reduce: bool = True,
    enable_ep: bool = False,
    quant_config: Any = None,
):
    """fused moe builder."""

    if quant_config is None:
        return FusedMoE(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            renormalize=renormalize,
            dtype=dtype,
            device=device,
            all_reduce=all_reduce,
            enable_ep=enable_ep,
        )

    quant_method = quant_config['quant_method']
    if quant_method == 'smooth_quant':
        quant_dtype = eval('torch.' + quant_config.get('quant_dtype', 'int8'))
        return FusedMoEW8A8(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            renormalize=renormalize,
            dtype=dtype,
            quant_dtype=quant_dtype,
            device=device,
            all_reduce=all_reduce,
            enable_ep=enable_ep,
        )
    elif quant_method == 'fp8':
        fmt = quant_config.get('fmt', 'e4m3')
        if fmt == 'e4m3':
            fp8_dtype = torch.float8_e4m3fn
        elif fmt == 'e5m2':
            fp8_dtype = torch.float8_e5m2
        else:
            raise TypeError(f'Unsupported fp8 fmt: {fmt}')
        return FusedMoEBlockedF8(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            renormalize=renormalize,
            fp8_dtype=fp8_dtype,
            dtype=dtype,
            device=device,
            all_reduce=all_reduce,
            enable_ep=enable_ep,
        )
    else:
        raise RuntimeError(f'Unsupported quant method: {quant_method}')

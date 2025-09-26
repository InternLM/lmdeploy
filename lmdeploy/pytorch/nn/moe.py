# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.config import TPMode
from lmdeploy.pytorch.distributed import get_dist_manager, get_ep_world_rank, get_tp_world_rank
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager

from ..backends import OpType, get_backend
from .quant_utils import quant_blocked_fp8
from .utils import div_up


class MoeType(Enum):
    """Batch ecex type."""
    Default = auto()
    DSSyncDecode = auto()
    DSAsyncDecode = auto()
    DSSyncPrefill = auto()
    DSAsyncPrefill = auto()


class SoftmaxTopK(nn.Module):
    """Softmax topk."""

    def __init__(self, top_k: int, dim: int = -1):
        super().__init__()
        self.top_k = top_k
        impl_builder = get_backend().get_layer_impl_builder(OpType.SoftmaxTopK)
        self.impl = impl_builder.build(top_k, dim)

    def forward(self, x: torch.Tensor):
        """forward."""
        return self.impl.forward(x)


def create_mlp_weights(hidden_dim: int, ffn_dim: int, num_experts: int, dtype: torch.dtype, device: torch.device):
    """Create weights."""
    gate_up_weights = torch.empty((num_experts, ffn_dim * 2, hidden_dim), dtype=dtype, device=device)
    down_weights = torch.empty((num_experts, hidden_dim, ffn_dim), dtype=dtype, device=device)
    return gate_up_weights, down_weights


def _update_args(hidden_dim: int, ffn_dim: int):
    """Update args."""
    world_size, _ = get_tp_world_rank('moe')
    assert ffn_dim % world_size == 0
    ffn_dim = ffn_dim // world_size
    return hidden_dim, ffn_dim


def _split_size(size: int, world_size: int, align: int):
    size = size // align
    assert size >= world_size
    base = size // world_size
    remain = size % world_size
    split_size = [base + 1] * remain + [base] * (world_size - remain)
    split_size = [s * align for s in split_size]
    return split_size


class MoEForwardDPTP:

    def __init__(self, gemm_func: Callable, max_tokens_per_round: int = 8192):
        """MoE forward dp tp."""
        self.gemm_func = gemm_func
        self.dist_ctx = get_dist_manager().current_context()
        self.dist_config = self.dist_ctx.dist_config
        self.tp = self.dist_config.moe_tp
        self.attn_tp = self.dist_config.attn_tp

        tp_group = self.dist_ctx.moe_tp_group
        self.rank = tp_group.rank
        self.gather_rank = self.rank // self.attn_tp
        self.gather_group = tp_group.gpu_gather_group
        self.tp_group = tp_group.gpu_group
        self.max_tokens_per_round = max_tokens_per_round * self.attn_tp // self.tp // 2

    def all_gather(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                   tp_sizes: List[int]):
        """All gather."""
        hidden_states, _ = dist.gather_by_tp_sizes(hidden_states, tp_sizes, group=self.gather_group, async_op=True)
        topk_weights, _ = dist.gather_by_tp_sizes(topk_weights, tp_sizes, group=self.gather_group, async_op=True)
        topk_ids, handle = dist.gather_by_tp_sizes(topk_ids, tp_sizes, group=self.gather_group, async_op=True)
        return hidden_states, topk_weights, topk_ids, handle

    def reduce_scatter(self, hidden_states: torch.Tensor, out_states: torch.Tensor, tp_sizes: List[int]):
        """Reduce scatter."""
        hidden_states_list = list(hidden_states.split(tp_sizes, -2))
        hidden_states_list[self.gather_rank] = out_states
        hidden_states_list = [item for item in hidden_states_list for _ in range(self.attn_tp)]
        handle = dist.reduce_scatter(out_states, hidden_states_list, group=self.tp_group, async_op=True)
        return out_states, handle

    def _gemm_and_reduce_scatter(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                                 output_states: torch.Tensor, tp_sizes: List[int], handle: dist.Work):
        """Gemm and reduce scatter."""
        handle.wait()
        cur_out = self.gemm_func(hidden_states, topk_weights, topk_ids)
        cur_out_states = cur_out.split(tp_sizes, dim=0)[self.gather_rank]
        output_states.copy_(cur_out_states)
        return self.reduce_scatter(cur_out, output_states, tp_sizes)

    def forward(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor):
        """forward."""

        def __slice_tensor(tensor: torch.Tensor, slice_size: int):
            """Slice tensor."""
            cur_tensor = tensor[:slice_size]
            tensor = tensor[slice_size:]
            return cur_tensor, tensor

        def __slice_and_gather():
            """Slice and gather."""
            nonlocal hidden_states, topk_weights, topk_ids, tp_sizes, output_states
            cur_tp_sizes = tp_sizes.minimum(max_tokens_per_round)
            tp_sizes -= cur_tp_sizes
            cur_tp_sizes = cur_tp_sizes.tolist()

            slice_size = cur_tp_sizes[self.gather_rank]
            cur_hidden_states, hidden_states = __slice_tensor(hidden_states, slice_size)
            cur_topk_weights, topk_weights = __slice_tensor(topk_weights, slice_size)
            cur_topk_ids, topk_ids = __slice_tensor(topk_ids, slice_size)
            cur_output, output_states = __slice_tensor(output_states, slice_size)

            # all gather
            cur_hidden_states, cur_topk_weights, cur_topk_ids, handle = self.all_gather(
                cur_hidden_states, cur_topk_weights, cur_topk_ids, cur_tp_sizes)
            return dict(hidden_states=cur_hidden_states,
                        topk_weights=cur_topk_weights,
                        topk_ids=cur_topk_ids,
                        output_states=cur_output,
                        handle=handle,
                        tp_sizes=cur_tp_sizes)

        step_ctx = get_step_ctx_manager().current_context()
        tp_sizes = step_ctx.dp_meta.moe_tp_sizes
        tp_sizes = torch.tensor(tp_sizes)
        max_tokens_per_round = tp_sizes.new_tensor(self.max_tokens_per_round)

        output_states = torch.empty_like(hidden_states)
        return_states = output_states

        # pre
        cur_inputs = __slice_and_gather()

        # main loop
        while tp_sizes.sum() > 0:
            next_inputs = __slice_and_gather()
            self._gemm_and_reduce_scatter(**cur_inputs)
            cur_inputs = next_inputs

        # post
        _, handle = self._gemm_and_reduce_scatter(**cur_inputs)
        handle.wait()
        return return_states


class LinearWeights(nn.Module):
    """Fused moe linear weights."""

    def __init__(self,
                 num_experts: int,
                 in_features: int,
                 out_features: int,
                 weight_type: str,
                 dtype: torch.dtype,
                 device: torch.device,
                 bias: bool = False,
                 expert_list: List[int] = None,
                 ep: bool = False):
        super().__init__()
        weight = torch.empty((num_experts, out_features, in_features), dtype=dtype, device=device)
        weight = torch.nn.Parameter(weight, requires_grad=False)
        self.register_parameter('weight', weight)

        if bias:
            bias = torch.empty((num_experts, out_features), dtype=dtype, device=device)
            bias = torch.nn.Parameter(bias, requires_grad=False)
            self.register_parameter('bias', bias)
        else:
            self.bias = None

        self.ep = ep
        self.expert_list = expert_list
        self.weight_type = weight_type
        self.half_out = out_features // 2

        self.setup_weight_loader()

    def setup_weight_loader(self):
        """Setup weight loader."""
        if self.ep:
            self.expert_map = defaultdict(list)
            for idx, eid in enumerate(self.expert_list):
                self.expert_map[eid].append(idx)
            self.weight.weight_loader = self.weight_loader_ep
        else:
            self.weight.weight_loader = self.weight_loader_tp

        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader_ep if self.ep else self.weight_loader_tp

    def update_weight(self, weight: torch.Tensor):
        """Update weight."""
        weight_loader = self.weight.weight_loader
        weight = torch.nn.Parameter(weight, requires_grad=False)
        weight.weight_loader = weight_loader
        self.register_parameter('weight', weight)

    def weight_loader_tp(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int, shard_id: str):
        """Weight loader."""
        world_size, rank = get_tp_world_rank('moe')
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
            if weight.dim() > 1:
                weight = weight.chunk(world_size, dim=1)[rank]
            elif weight.dim() == 1 and rank != 0:
                # bias with rank>0 should be 0
                weight = torch.zeros_like(weight)
        else:
            raise RuntimeError(f'Unknown shard_id: {shard_id}')
        param_data.copy_(weight)

    def weight_loader_ep(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int, shard_id: str):
        """Weight loader."""
        expert_list = self.expert_list
        if expert_id not in expert_list:
            return

        expert_map = self.expert_map
        param_ids = expert_map[expert_id]
        for param_id in param_ids:
            if shard_id == 'gate':
                param_data = param.data[param_id, :self.half_out]
            elif shard_id == 'up':
                param_data = param.data[param_id, self.half_out:]
            elif shard_id == 'down':
                param_data = param.data[param_id]
            else:
                raise RuntimeError(f'Unknown shard_id: {shard_id}')
            param_data.copy_(loaded_weight)


def _moe_gather_inputs(hidden_states, topk_weights, topk_ids, group: Optional[dist.ProcessGroup] = None):
    dist_config = get_dist_manager().current_config()
    tp = dist_config.moe_tp
    if tp == 1:
        return hidden_states, topk_weights, topk_ids

    tp_mode = dist_config.moe_tp_mode
    if tp_mode == TPMode.DEFAULT:
        return hidden_states, topk_weights, topk_ids
    elif tp_mode == TPMode.DP_TP:
        step_ctx = get_step_ctx_manager().current_context()
        dp_meta = step_ctx.dp_meta
        tp_sizes = dp_meta.moe_tp_sizes
        hidden_states = dist.gather_by_tp_sizes(hidden_states, tp_sizes, group=group)
        topk_weights = dist.gather_by_tp_sizes(topk_weights, tp_sizes, group=group)
        topk_ids = dist.gather_by_tp_sizes(topk_ids, tp_sizes, group=group)
    else:
        raise RuntimeError('Not supported.')

    return hidden_states, topk_weights, topk_ids


def _moe_reduce(ret, rank: int, tp_mode: TPMode, group: Optional[dist.ProcessGroup] = None):
    dist_config = get_dist_manager().current_config()
    if dist_config.moe_tp == 1:
        return ret

    if tp_mode == TPMode.DEFAULT:
        dist.all_reduce(ret, group=group)
        return ret
    elif tp_mode == TPMode.DP_TP:
        step_ctx = get_step_ctx_manager().current_context()
        dp_meta = step_ctx.dp_meta
        tp_size = dp_meta.moe_tp_sizes
        ret = dist.reduce_scatter_by_tp_sizes(ret, rank, tp_size, group=group)
        return ret
    else:
        raise RuntimeError('Not supported.')


class FusedMoE(nn.Module):
    """Fused moe."""

    def __init__(self,
                 hidden_dim: int,
                 ffn_dim: int,
                 num_experts: int,
                 top_k: int,
                 bias: bool = False,
                 renormalize: bool = False,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 all_reduce: bool = True,
                 enable_ep: bool = False,
                 act_func: Callable = None):
        super().__init__()
        if device is None:
            device = torch.device('cpu')
        if dtype is None:
            dtype = torch.float16
        self.init_tp_args(all_reduce, enable_ep)

        impl_builder = get_backend().get_layer_impl_builder(OpType.FusedMoE)
        self.impl = impl_builder.build(top_k, num_experts, renormalize)

        enable_ep = enable_ep and self.impl.support_ep()
        if enable_ep:
            world_size, rank = get_tp_world_rank('moe')
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
                                     bias=bias,
                                     expert_list=expert_list,
                                     ep=enable_ep)
        self.down = LinearWeights(
            num_experts,
            ffn_dim,
            hidden_dim,
            weight_type='down',
            dtype=dtype,
            device=device,
            bias=bias,
            expert_list=expert_list,
            ep=enable_ep,
        )

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.dtype = dtype
        self.device = device
        self.enable_ep = enable_ep
        self.act_func = act_func

    def init_tp_args(self, all_reduce: bool, enable_ep: bool):
        """Init tp args."""
        tp, tp_rank = get_tp_world_rank('moe')
        dist_ctx = get_dist_manager().current_context()
        dist_cfg = dist_ctx.dist_config
        _, tp_mode = dist_cfg.get_tp_by_layer('moe')
        tp = 1 if enable_ep else tp
        tp_rank = 0 if enable_ep else tp_rank
        all_reduce = all_reduce if tp > 1 else False
        all_reduce = False if enable_ep else all_reduce

        self.tp = tp
        self.tp_rank = tp_rank
        self.tp_mode = tp_mode
        self.all_reduce = all_reduce
        self.tp_group = dist_ctx.moe_tp_group.gpu_group
        self.gather_group = dist_ctx.moe_tp_group.gpu_gather_group

        if self.tp > 1 and self.tp_mode == TPMode.DP_TP:

            def __gemm_func(hidden_states, topk_weights, topk_ids):
                return self.gemm(
                    dict(
                        hidden_states=hidden_states,
                        topk_weights=topk_weights,
                        topk_idx=topk_ids,
                        moe_type=MoeType.Default,
                    ))['hidden_states']

            self.forward_dptp = MoEForwardDPTP(__gemm_func)

    def update_weights(self):
        """Update weights."""
        gate_up_weights, down_weights = self.impl.update_weights(self.gate_up.weight, self.down.weight)
        self.gate_up.update_weight(gate_up_weights)
        self.down.update_weight(down_weights)

    def gemm(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Gemm."""
        hidden_states = inputs['hidden_states']
        topk_weights = inputs['topk_weights']
        topk_ids = inputs['topk_idx']

        ret = self.impl.forward(hidden_states,
                                topk_weights,
                                topk_ids,
                                self.gate_up.weight,
                                self.down.weight,
                                self.gate_up.bias,
                                self.down.bias,
                                self.expert_list,
                                act_func=self.act_func)
        return dict(hidden_states=ret)

    def forward_default(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.LongTensor):
        hidden_states, topk_weights, topk_ids = _moe_gather_inputs(hidden_states,
                                                                   topk_weights,
                                                                   topk_ids,
                                                                   group=self.gather_group)

        ret = self.impl.forward(hidden_states,
                                topk_weights,
                                topk_ids,
                                self.gate_up.weight,
                                self.down.weight,
                                self.gate_up.bias,
                                self.down.bias,
                                self.expert_list,
                                act_func=self.act_func)
        if self.all_reduce:
            ret = _moe_reduce(ret, rank=self.tp_rank, tp_mode=self.tp_mode, group=self.tp_group)
        return ret

    def forward(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.LongTensor):
        """forward."""
        if self.tp > 1 and self.tp_mode == TPMode.DP_TP:
            return self.forward_dptp.forward(hidden_states, topk_weights, topk_ids)
        return self.forward_default(hidden_states, topk_weights, topk_ids)


class LinearWeightsW8A8(LinearWeights):
    """Fused moe linear w8a8 weights."""

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


class FusedMoEW8A8(nn.Module):
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
            world_size, rank = get_tp_world_rank('moe')
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
        world_size, _ = get_tp_world_rank('moe')
        if world_size == 1:
            all_reduce = False
        self.all_reduce = all_reduce

    def update_weights(self):
        """Update weights."""
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
    """Fused moe linear blocked fp8 weights."""

    def __init__(self,
                 num_experts: int,
                 in_features: int,
                 out_features: int,
                 weight_type: str,
                 block_size: int,
                 dtype: torch.dtype,
                 device: torch.device,
                 bias: bool = False,
                 expert_list: List[int] = None,
                 ep: bool = False):
        super().__init__(
            num_experts=num_experts,
            in_features=in_features,
            out_features=out_features,
            weight_type=weight_type,
            dtype=dtype,
            device=device,
            bias=bias,
            expert_list=expert_list,
            ep=ep,
        )
        self.block_size = block_size
        weight_scale_inv = torch.empty((num_experts, div_up(out_features, block_size), div_up(in_features, block_size)),
                                       dtype=torch.float32,
                                       device=device)
        weight_scale_inv = torch.nn.Parameter(weight_scale_inv, requires_grad=False)
        self.register_parameter('weight_scale_inv', weight_scale_inv)
        self.weight._base_weight_loader = self.weight_loader_tp
        self.weight.weight_loader = self.weight_loader_with_quant

        if self.ep:
            self.weight_scale_inv.weight_loader = self.weight_loader_scale_ep
        else:
            self.weight_scale_inv.weight_loader = self.weight_loader_scale_tp

    def update_weight(self, weight: torch.Tensor, weight_scale_inv: torch.Tensor):
        """Update weight."""
        super().update_weight(weight=weight)
        weight_loader = self.weight_scale_inv.weight_loader
        weight_scale_inv = torch.nn.Parameter(weight_scale_inv, requires_grad=False)
        weight_scale_inv.weight_loader = weight_loader
        self.register_parameter('weight_scale_inv', weight_scale_inv)

    def weight_loader_scale_ep(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int,
                               shard_id: str):
        expert_list = self.expert_list
        if expert_id not in expert_list:
            return
        expert_ids = self.expert_map[expert_id]
        for expert_id in expert_ids:
            self.weight_loader_scale_tp(param, loaded_weight, expert_id, shard_id)

    def _chunk_weight_tp(self, weight: torch.Tensor, dim: int, world_size: int, rank: int, align: int):
        """Chunk with align."""
        split_size = _split_size(weight.size(dim), world_size, align)
        return weight.split(split_size, dim=dim)[rank]

    def weight_loader_tp(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int, shard_id: str):
        """Weight loader."""
        world_size, rank = get_tp_world_rank('moe')
        if shard_id == 'gate':
            param_data = param.data[expert_id, :self.half_out]
            weight = self._chunk_weight_tp(loaded_weight,
                                           dim=0,
                                           world_size=world_size,
                                           rank=rank,
                                           align=self.block_size)
        elif shard_id == 'up':
            param_data = param.data[expert_id, self.half_out:]
            weight = self._chunk_weight_tp(loaded_weight,
                                           dim=0,
                                           world_size=world_size,
                                           rank=rank,
                                           align=self.block_size)
        elif shard_id == 'down':
            param_data = param.data[expert_id]
            # weight is not contiguous, chunk and copy in cpu is slow
            weight = loaded_weight.to(param_data.device)
            if weight.dim() > 1:
                weight = self._chunk_weight_tp(weight, dim=1, world_size=world_size, rank=rank, align=self.block_size)
            elif weight.dim() == 1 and rank != 0:
                # bias with rank>0 should be 0
                weight = torch.zeros_like(weight)
        else:
            raise RuntimeError(f'Unknown shard_id: {shard_id}')
        param_data.copy_(weight)

    def weight_loader_scale_tp(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int,
                               shard_id: str):
        """Weight loader scale tp."""
        world_size, rank = get_tp_world_rank('moe')
        block_size = self.block_size
        half_out = self.half_out // block_size
        if shard_id == 'gate':
            param_data = param.data[expert_id, :half_out]
            weight = self._chunk_weight_tp(loaded_weight, dim=0, world_size=world_size, rank=rank, align=1)
        elif shard_id == 'up':
            param_data = param.data[expert_id, half_out:]
            weight = self._chunk_weight_tp(loaded_weight, dim=0, world_size=world_size, rank=rank, align=1)
        elif shard_id == 'down':
            param_data = param.data[expert_id]
            loaded_weight = loaded_weight.to(param_data.device)
            weight = self._chunk_weight_tp(loaded_weight, dim=1, world_size=world_size, rank=rank, align=1)
        else:
            raise RuntimeError(f'Unknown shard_id: {shard_id}')
        param_data.copy_(weight)

    def weight_loader_with_quant(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int,
                                 shard_id: str):
        """Weight load with quant."""
        if loaded_weight.dtype != param.dtype:
            # quant loaded weight
            quanted_weight, scaling = quant_blocked_fp8(loaded_weight.to(param.device), param.dtype, self.block_size)
            self.weight._base_weight_loader(self.weight, quanted_weight, expert_id, shard_id)
            self.weight_scale_inv.weight_loader(self.weight_scale_inv, scaling, expert_id, shard_id)
        else:
            return self.weight._base_weight_loader(param, loaded_weight, expert_id, shard_id)


class FusedMoEBlockedF8(nn.Module):
    """Fused moe blocked f8."""

    def __init__(self,
                 hidden_dim: int,
                 ffn_dim: int,
                 num_experts: int,
                 top_k: int,
                 bias: bool = False,
                 renormalize: bool = False,
                 fp8_dtype: torch.dtype = torch.float8_e4m3fn,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 all_reduce: bool = True,
                 enable_ep: bool = False,
                 layer_idx: int = 0,
                 act_func: Callable = None):
        super().__init__()
        if device is None:
            device = torch.device('cpu')
        dtype = torch.float16 if dtype is None else dtype
        self.block_size = 128
        self.init_tp_args(all_reduce, enable_ep)
        dist_ctx = get_dist_manager().current_context()
        self.ep_size, rank = get_ep_world_rank()
        impl_builder = get_backend().get_layer_impl_builder(OpType.FusedMoEBlockedF8)
        self.impl = impl_builder.build(top_k,
                                       num_experts,
                                       hidden_dim,
                                       renormalize,
                                       block_size=self.block_size,
                                       ep_size=self.ep_size,
                                       ep_group=dist_ctx.ep_gpu_group,
                                       out_dtype=dtype,
                                       layer_idx=layer_idx,
                                       custom_gateup_act=act_func is not None)

        if self.ep_size > 1:
            expert_list = self.impl.ep_expert_list(self.ep_size, rank)
            num_experts = len(expert_list)
        else:
            hidden_dim, ffn_dim = self._update_args(hidden_dim, ffn_dim, align=self.block_size)
            expert_list = None
        self.expert_list = expert_list

        self.gate_up = LinearWeightsBlockedF8(num_experts,
                                              hidden_dim,
                                              ffn_dim * 2,
                                              weight_type='gate_up',
                                              block_size=self.block_size,
                                              dtype=fp8_dtype,
                                              device=device,
                                              bias=bias,
                                              expert_list=expert_list,
                                              ep=self.ep_size > 1)
        self.down = LinearWeightsBlockedF8(
            num_experts,
            ffn_dim,
            hidden_dim,
            weight_type='down',
            block_size=self.block_size,
            dtype=fp8_dtype,
            device=device,
            bias=bias,
            expert_list=expert_list,
            ep=self.ep_size > 1,
        )

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.dtype = dtype
        self.device = device
        self.act_func = act_func

    @staticmethod
    def _update_args(hidden_dim: int, ffn_dim: int, align: int):
        """Update args."""
        world_size, rank = get_tp_world_rank('moe')
        split_size = _split_size(ffn_dim, world_size, align)
        ffn_dim = split_size[rank]
        return hidden_dim, ffn_dim

    def init_tp_args(self, all_reduce: bool, enable_ep: bool):
        """Init tp args."""
        tp, tp_rank = get_tp_world_rank('moe')
        dist_ctx = get_dist_manager().current_context()
        dist_cfg = dist_ctx.dist_config
        _, tp_mode = dist_cfg.get_tp_by_layer('moe')
        tp = 1 if enable_ep else tp
        tp_rank = 0 if enable_ep else tp_rank
        all_reduce = all_reduce if tp > 1 else False
        all_reduce = False if enable_ep else all_reduce

        self.tp = tp
        self.tp_rank = tp_rank
        self.tp_mode = tp_mode
        self.all_reduce = all_reduce
        self.tp_group = dist_ctx.moe_tp_group.gpu_group
        self.gather_group = dist_ctx.moe_tp_group.gpu_gather_group
        if self.tp > 1 and self.tp_mode == TPMode.DP_TP:

            def __gemm_func(hidden_states, topk_weights, topk_ids):
                return self.gemm(
                    dict(
                        hidden_states=hidden_states,
                        topk_weights=topk_weights,
                        topk_idx=topk_ids,
                        moe_type=MoeType.Default,
                    ))['hidden_states']

            self.forward_dptp = MoEForwardDPTP(__gemm_func)

    def update_weights(self):
        """Update weights."""
        (gate_up_weights, down_weights, gate_up_scale,
         down_scale) = self.impl.update_weights(self.gate_up.weight, self.down.weight, self.gate_up.weight_scale_inv,
                                                self.down.weight_scale_inv)
        self.gate_up.update_weight(gate_up_weights, gate_up_scale)
        self.down.update_weight(down_weights, down_scale)

    def forward_default(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_idx: torch.LongTensor):
        state = {
            'hidden_states': hidden_states,
            'topk_idx': topk_idx,
            'topk_weights': topk_weights,
            'moe_type': MoeType.Default,
        }
        recv_state = self.dispatch(state)
        gemm_state = self.gemm(recv_state)
        out_state = self.combine(gemm_state)
        return out_state['hidden_states']

    def forward(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_idx: torch.LongTensor):

        if self.tp > 1 and self.tp_mode == TPMode.DP_TP:
            return self.forward_dptp.forward(hidden_states, topk_weights, topk_idx)
        else:
            return self.forward_default(hidden_states, topk_weights, topk_idx)

    def before_dispatch(self, state: Dict):
        moe_type = state['moe_type']
        if moe_type == MoeType.DSAsyncPrefill:
            fusedmoe = self.fusedmoe_build(low_latency_mode=False)
            state['fusedmoe'] = fusedmoe
            if hasattr(fusedmoe, 'per_token_group_quant_fp8'):
                state['hidden_states'] = fusedmoe.per_token_group_quant_fp8(state['hidden_states'])
            previous_event = fusedmoe.capture()
            state['previous_event'] = previous_event
        return state

    def dispatch(self, state: Dict):
        moe_type = state['moe_type']
        if moe_type == MoeType.DSAsyncPrefill:
            fusedmoe = state['fusedmoe']
            previous_event = state['previous_event']
            (
                recv_hidden_states,
                recv_topk_idx,
                recv_topk_weights,
                recv_tokens_per_expert,
                handle,
                event,
            ) = fusedmoe.dispatch_async(state['hidden_states'],
                                        state['topk_idx'],
                                        state['topk_weights'],
                                        previous_event=previous_event,
                                        async_finish=True)
            recv_state = {
                'fusedmoe': fusedmoe,
                'recv_hidden_states': recv_hidden_states,
                'recv_topk_idx': recv_topk_idx,
                'recv_topk_weights': recv_topk_weights,
                'recv_tokens_per_expert': recv_tokens_per_expert,
                'handle': handle,
                'event': event,
                'num_experts': self.num_experts,
                'moe_type': state['moe_type']
            }
        elif moe_type == MoeType.DSAsyncDecode:
            fusedmoe = self.fusedmoe_build(low_latency_mode=True)
            use_event = False
            (recv_hidden_states, recv_expert_count, handle, event,
             hook) = fusedmoe.dispatch_async(state['hidden_states'],
                                             state['topk_idx'],
                                             use_fp8=True,
                                             async_finish=use_event)
            recv_state = {
                'fusedmoe': fusedmoe,
                'recv_hidden_states': recv_hidden_states,
                'recv_expert_count': recv_expert_count,
                'topk_idx': state['topk_idx'],
                'topk_weights': state['topk_weights'],
                'raw_hidden_shape': state['raw_hidden_shape'],
                'handle': handle,
                'moe_type': state['moe_type']
            }
            if use_event:
                recv_state['event'] = event
            else:
                recv_state['hook'] = hook
        else:  # MoeType.Default
            hidden_states, topk_weights, topk_idx = _moe_gather_inputs(state['hidden_states'],
                                                                       state['topk_weights'],
                                                                       state['topk_idx'],
                                                                       group=self.gather_group)
            recv_state = {
                'hidden_states': hidden_states,
                'topk_idx': topk_idx,
                'topk_weights': topk_weights,
                'moe_type': state['moe_type']
            }
        return recv_state

    def gemm(self, state: Dict):
        moe_type = state['moe_type']
        if moe_type == MoeType.DSAsyncPrefill:
            if (state['recv_hidden_states'][0]
                    if isinstance(state['recv_hidden_states'], tuple) else state['recv_hidden_states']).shape[0] > 0:
                state['recv_hidden_states'] = state['fusedmoe'].fusedmoe_forward(state, self.gate_up.weight,
                                                                                 self.gate_up.weight_scale_inv,
                                                                                 self.down.weight,
                                                                                 self.down.weight_scale_inv)
            gemm_state = {
                'fusedmoe': state['fusedmoe'],
                'hidden_states': state['recv_hidden_states'],
                'handle': state['handle'],
                'moe_type': state['moe_type']
            }
        elif moe_type == MoeType.DSAsyncDecode:
            state['recv_hidden_states'] = state['fusedmoe'].fusedmoe_forward(state, self.gate_up.weight,
                                                                             self.gate_up.weight_scale_inv,
                                                                             self.down.weight,
                                                                             self.down.weight_scale_inv)
            gemm_state = {
                'fusedmoe': state['fusedmoe'],
                'hidden_states': state['recv_hidden_states'],
                'topk_idx': state['topk_idx'],
                'topk_weights': state['topk_weights'],
                'handle': state['handle'],
                'moe_type': state['moe_type']
            }
        else:  # MoeType.Default
            hidden_states = self.impl.forward(state['hidden_states'],
                                              state['topk_weights'],
                                              state['topk_idx'],
                                              self.gate_up.weight,
                                              self.gate_up.weight_scale_inv,
                                              self.down.weight,
                                              self.down.weight_scale_inv,
                                              gate_up_bias=self.gate_up.bias,
                                              down_bias=self.down.bias,
                                              expert_list=self.expert_list,
                                              act_func=self.act_func)
            gemm_state = {'hidden_states': hidden_states, 'moe_type': state['moe_type']}
        return gemm_state

    def combine(self, state: Dict):
        moe_type = state['moe_type']
        if moe_type == MoeType.DSAsyncPrefill:
            fusedmoe = state['fusedmoe']
            previous_event = fusedmoe.capture()
            out_hidden_states, event = fusedmoe.combine_async(state['hidden_states'],
                                                              state['handle'],
                                                              previous_event=previous_event,
                                                              async_finish=True)
            out_state = {
                'fusedmoe': state['fusedmoe'],
                'hidden_states': out_hidden_states,
                'event': event,
                'moe_type': state['moe_type']
            }
        elif moe_type == MoeType.DSAsyncDecode:
            fusedmoe = state['fusedmoe']
            use_event = False
            out_hidden_states, event, hook = fusedmoe.combine_async(state['hidden_states'],
                                                                    state['topk_idx'],
                                                                    state['topk_weights'],
                                                                    state['handle'],
                                                                    async_finish=use_event)
            out_state = {
                'fusedmoe': state['fusedmoe'],
                'hidden_states': out_hidden_states,
                'moe_type': state['moe_type']
            }
            if use_event:
                out_state['event'] = event
            else:
                out_state['hook'] = hook
        else:  # MoeType.Default
            if self.all_reduce:
                state['hidden_states'] = _moe_reduce(state['hidden_states'],
                                                     rank=self.tp_rank,
                                                     tp_mode=self.tp_mode,
                                                     group=self.tp_group)
            out_state = {'hidden_states': state['hidden_states'], 'moe_type': state['moe_type']}
        return out_state

    def wait(self, state):
        if state.get('event', None) is not None:
            state['fusedmoe'].wait(state['event'])
            return True
        elif state.get('hook', None) is not None:
            state['hook']()
            return True
        else:
            return False

    def renormalize(self, topk_weights):
        return self.impl.do_renormalize(topk_weights)

    def fusedmoe_build(self, low_latency_mode: bool = False):
        return self.impl.fusedmoe_build(low_latency_mode)


def build_fused_moe(
    hidden_dim: int,
    ffn_dim: int,
    num_experts: int,
    top_k: int,
    bias: bool = False,
    renormalize: bool = False,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    all_reduce: bool = True,
    enable_ep: bool = False,
    quant_config: Any = None,
    layer_idx: int = 0,
    act_func: Callable = None,
):
    """Fused moe builder."""

    if quant_config is None:
        return FusedMoE(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            bias=bias,
            renormalize=renormalize,
            dtype=dtype,
            device=device,
            all_reduce=all_reduce,
            enable_ep=enable_ep,
            act_func=act_func,
        )

    quant_method = quant_config['quant_method']
    if quant_method == 'smooth_quant':
        assert not bias, 'Quant model does not support bias for now.'
        assert act_func is None, ('Quant model does not support activation function for now.')
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
            bias=bias,
            renormalize=renormalize,
            fp8_dtype=fp8_dtype,
            dtype=dtype,
            device=device,
            all_reduce=all_reduce,
            enable_ep=enable_ep,
            layer_idx=layer_idx,
            act_func=act_func,
        )
    else:
        raise RuntimeError(f'Unsupported quant method: {quant_method}')

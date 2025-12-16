# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import torch
from torch import nn

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.distributed import get_dist_manager, get_ep_world_rank, get_tp_world_rank

from .base import DispatchInputs, FusedMoEBase, MoeType, moe_gather_inputs, moe_reduce, update_dims


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
                 expert_list: Optional[List[int]] = None):
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

        self.ep = expert_list is not None
        self.expert_list = expert_list
        self.weight_type = weight_type
        self.half_out = out_features // 2

        self.setup_weight_loader()

    def setup_weight_loader(self):
        """Setup weight loader."""
        if self.expert_list is not None:
            self.expert_map = defaultdict(list)
            for idx, eid in enumerate(self.expert_list):
                self.expert_map[eid].append(idx)
            self.weight.weight_loader = self.weight_loader_ep
            if self.bias is not None:
                self.bias.weight_loader = self.weight_loader_ep
        else:
            self.weight.weight_loader = self.weight_loader_tp
            if self.bias is not None:
                self.bias.weight_loader = self.weight_loader_tp

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


class FusedMoE(FusedMoEBase):
    """Fused MoE."""

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
                 layer_idx: int = 0,
                 act_func: Callable = None):

        device = device or torch.device('cpu')
        dtype = dtype or torch.float16
        # init distributed tp arguments
        self.init_dist_args(all_reduce)

        super().__init__(
            tp=self.tp,
            tp_mode=self.tp_mode,
            do_renormalize=renormalize,
        )

        # create implementation
        dist_ctx = get_dist_manager().current_context()
        self.ep_size, rank = get_ep_world_rank()
        impl_builder = get_backend().get_layer_impl_builder(OpType.FusedMoE)
        self.impl = impl_builder.build(
            top_k,
            num_experts,
            renormalize,
            hidden_dim=hidden_dim,
            ep_size=self.ep_size,
            ep_group=dist_ctx.ep_gpu_group,
            layer_idx=layer_idx,
        )

        # create weights
        if self.ep_size > 1:
            expert_list = self.impl.ep_expert_list(self.ep_size, rank)
            num_experts = len(expert_list)
        else:
            hidden_dim, ffn_dim = update_dims(hidden_dim, ffn_dim)
            expert_list = None
        self.expert_list = expert_list
        self.gate_up = LinearWeights(num_experts,
                                     hidden_dim,
                                     ffn_dim * 2,
                                     weight_type='gate_up',
                                     dtype=dtype,
                                     device=device,
                                     bias=bias,
                                     expert_list=expert_list)
        self.down = LinearWeights(
            num_experts,
            ffn_dim,
            hidden_dim,
            weight_type='down',
            dtype=dtype,
            device=device,
            bias=bias,
            expert_list=expert_list,
        )

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.dtype = dtype
        self.device = device
        self.act_func = act_func

    def update_weights(self):
        """Update weights."""
        gate_up_weights, down_weights = self.impl.update_weights(self.gate_up.weight, self.down.weight)
        self.gate_up.update_weight(gate_up_weights)
        self.down.update_weight(down_weights)

    def before_dispatch(self, state: DispatchInputs):
        """Before dispatch."""
        if not isinstance(state, Dict):
            state = state.to_dict()

        moe_type = state['moe_type']
        if moe_type == MoeType.DSAsyncPrefill:
            fusedmoe = self.fusedmoe_build(low_latency_mode=False)
            state['fusedmoe'] = fusedmoe
            previous_event = fusedmoe.capture()
            state['previous_event'] = previous_event
        return state

    def dispatch(self, state: Dict):
        """dispatch."""
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
                                             use_fp8=False,
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
        elif moe_type == MoeType.Default:
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
        else:
            hidden_states = state['hidden_states']
            topk_weights = state['topk_weights']
            topk_ids = state['topk_idx']

            hidden_states = self.impl.forward(hidden_states,
                                              topk_weights,
                                              topk_ids,
                                              self.gate_up.weight,
                                              self.down.weight,
                                              self.gate_up.bias,
                                              self.down.bias,
                                              self.expert_list,
                                              act_func=self.act_func)
            gemm_state = {'hidden_states': hidden_states, 'moe_type': state['moe_type']}
        return gemm_state

    def combine(self, state: Dict):
        """combine."""
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
        elif moe_type == MoeType.Default:
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
        if state.get('event', None) is not None:
            state['fusedmoe'].wait(state['event'])
            return True
        elif state.get('hook', None) is not None:
            state['hook']()
            return True
        else:
            return False

    def fusedmoe_build(self, low_latency_mode: bool = False):
        return self.impl.fusedmoe_build(low_latency_mode)

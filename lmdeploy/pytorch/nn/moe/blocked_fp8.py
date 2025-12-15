# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict, List, Optional

import torch

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.distributed import get_dist_manager, get_ep_world_rank, get_tp_world_rank

from ..quant_utils import quant_blocked_fp8
from ..utils import div_up
from .base import DispatchInputs, FusedMoEBase, MoeType, moe_gather_inputs, moe_reduce
from .base import split_size as _split_size
from .default import LinearWeights


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
                 scale_fmt: Optional[str] = None):
        super().__init__(num_experts=num_experts,
                         in_features=in_features,
                         out_features=out_features,
                         weight_type=weight_type,
                         dtype=dtype,
                         device=device,
                         bias=bias,
                         expert_list=expert_list)
        self.scale_fmt = scale_fmt
        self.block_size = block_size
        weight_scale_inv = torch.empty((num_experts, div_up(out_features, block_size), div_up(in_features, block_size)),
                                       dtype=torch.float32,
                                       device=device)
        weight_scale_inv = torch.nn.Parameter(weight_scale_inv, requires_grad=False)
        self.register_parameter('weight_scale_inv', weight_scale_inv)

        if self.ep:
            self.weight._base_weight_loader = self.weight.weight_loader
            self.weight_scale_inv.weight_loader = self.weight_loader_scale_ep
        else:
            self.weight._base_weight_loader = self.weight_loader_tp_blocked_fp8
            self.weight_scale_inv.weight_loader = self.weight_loader_scale_tp
        self.weight.weight_loader = self.weight_loader_with_quant

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

    def weight_loader_tp_blocked_fp8(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int,
                                     shard_id: str):
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
            quanted_weight, scaling = quant_blocked_fp8(loaded_weight.to(param.device),
                                                        param.dtype,
                                                        self.block_size,
                                                        scale_fmt=self.scale_fmt)
            self.weight._base_weight_loader(self.weight, quanted_weight, expert_id, shard_id)
            self.weight_scale_inv.weight_loader(self.weight_scale_inv, scaling, expert_id, shard_id)
        else:
            return self.weight._base_weight_loader(param, loaded_weight, expert_id, shard_id)


class FusedMoEBlockedF8(FusedMoEBase):
    """Fused moe blocked f8."""

    def __init__(self,
                 hidden_dim: int,
                 ffn_dim: int,
                 num_experts: int,
                 top_k: int,
                 bias: bool = False,
                 renormalize: bool = False,
                 fp8_dtype: torch.dtype = torch.float8_e4m3fn,
                 scale_fmt: Optional[str] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 all_reduce: bool = True,
                 layer_idx: int = 0,
                 act_func: Callable = None):

        device = device or torch.device('cpu')
        dtype = dtype or torch.float16
        # init distributed tp arguments
        self.block_size = 128
        self.init_dist_args(all_reduce)
        self.scale_fmt = scale_fmt

        super().__init__(
            tp=self.tp,
            tp_mode=self.tp_mode,
            do_renormalize=renormalize,
        )

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
        self.impl.set_scale_fmt(scale_fmt)

        if self.ep_size > 1:
            expert_list = self.impl.ep_expert_list(self.ep_size, rank)
            num_experts = len(expert_list)
        else:
            hidden_dim, ffn_dim = self._update_args(hidden_dim, ffn_dim, align=self.block_size)
            expert_list = None
        self.expert_list = expert_list

        # create weights
        self.gate_up = LinearWeightsBlockedF8(num_experts,
                                              hidden_dim,
                                              ffn_dim * 2,
                                              weight_type='gate_up',
                                              block_size=self.block_size,
                                              dtype=fp8_dtype,
                                              device=device,
                                              bias=bias,
                                              expert_list=expert_list,
                                              scale_fmt=scale_fmt)
        self.down = LinearWeightsBlockedF8(num_experts,
                                           ffn_dim,
                                           hidden_dim,
                                           weight_type='down',
                                           block_size=self.block_size,
                                           dtype=fp8_dtype,
                                           device=device,
                                           bias=bias,
                                           expert_list=expert_list,
                                           scale_fmt=scale_fmt)

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.dtype = dtype
        self.device = device
        self.act_func = act_func

    @staticmethod
    def _update_args(hidden_dim: int, ffn_dim: int, align: int):
        world_size, rank = get_tp_world_rank('moe')
        split_size = _split_size(ffn_dim, world_size, align)
        ffn_dim = split_size[rank]
        return hidden_dim, ffn_dim

    def update_weights(self):
        """Update weights."""
        (gate_up_weights, down_weights, gate_up_scale,
         down_scale) = self.impl.update_weights(self.gate_up.weight, self.down.weight, self.gate_up.weight_scale_inv,
                                                self.down.weight_scale_inv)
        self.gate_up.update_weight(gate_up_weights, gate_up_scale)
        self.down.update_weight(down_weights, down_scale)

    def before_dispatch(self, state: DispatchInputs):
        """Before dispatch."""
        if not isinstance(state, Dict):
            state = state.to_dict()

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
            hidden_states, topk_weights, topk_idx = moe_gather_inputs(state['hidden_states'],
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
            if self.gate_up.weight.numel() == 0:
                # current rank get no expert chunk
                # create a zero tensor with the same shape as hidden_states
                gemm_state = {'hidden_states': torch.zeros_like(state['hidden_states']), 'moe_type': state['moe_type']}
            else:
                # default fused moe
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
                state['hidden_states'] = moe_reduce(state['hidden_states'],
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

    def fusedmoe_build(self, low_latency_mode: bool = False):
        return self.impl.fusedmoe_build(low_latency_mode)

# Copyright (c) OpenMMLab. All rights reserved.

import torch
from torch import nn

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.distributed import get_dist_manager, get_ep_world_rank, get_tp_world_rank

from .base import moe_reduce
from .base import split_size as _split_size


class V4ExpertWeights(nn.Module):
    """Local expert-sharded V4 expert weights.

    The checkpoint-native storage is packed FP4 and is consumed directly by the Triton V4 FP4 MoE kernel.
    """

    def __init__(self,
                 num_local_experts: int,
                 in_features: int,
                 out_features: int,
                 expert_offset: int,
                 weight_type: str,
                 device: torch.device):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.in_features = in_features
        self.out_features = out_features
        self.expert_offset = expert_offset
        self.weight_type = weight_type
        self.half_out = out_features // 2

        weight = torch.empty((num_local_experts, out_features, in_features // 2), dtype=torch.int8, device=device)
        scale = torch.empty((num_local_experts, out_features, in_features // 32),
                            dtype=torch.float8_e8m0fnu,
                            device=device)
        self.register_parameter('weight', nn.Parameter(weight, requires_grad=False))
        self.register_parameter('scale', nn.Parameter(scale, requires_grad=False))

        self.weight.weight_loader = self.weight_loader
        self.scale.weight_loader = self.scale_loader

    def _get_local_param(self, param: nn.Parameter, expert_id: int, shard_id: str):
        local_id = expert_id - self.expert_offset
        if local_id < 0 or local_id >= self.num_local_experts:
            return None
        if self.weight_type == 'gate_up':
            if shard_id == 'gate':
                return param.data[local_id, :self.half_out]
            if shard_id == 'up':
                return param.data[local_id, self.half_out:]
        elif self.weight_type == 'down' and shard_id == 'down':
            return param.data[local_id]
        raise RuntimeError(f'Unsupported shard_id={shard_id} for weight_type={self.weight_type}')

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, expert_id: int, shard_id: str):
        param_data = self._get_local_param(param, expert_id, shard_id)
        if param_data is None:
            return
        param_data.copy_(loaded_weight.to(param_data.device, dtype=param_data.dtype))

    def scale_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, expert_id: int, shard_id: str):
        param_data = self._get_local_param(param, expert_id, shard_id)
        if param_data is None:
            return
        param_data.copy_(loaded_weight.to(param_data.device, dtype=param_data.dtype))

    def update(self, weight: torch.Tensor, scale: torch.Tensor):
        weight_loader = self.weight.weight_loader
        scale_loader = self.scale.weight_loader
        weight = nn.Parameter(weight, requires_grad=False)
        scale = nn.Parameter(scale, requires_grad=False)
        weight.weight_loader = weight_loader
        scale.weight_loader = scale_loader
        self.register_parameter('weight', weight)
        self.register_parameter('scale', scale)


class V4ExpertTPWeights(nn.Module):
    """Checkpoint-native V4 expert weights for TP-sharded fused MoE.

    Unlike the older experimental implementation, these tensors keep all routed experts and shard only the expert FFN
    dimension across TP ranks. That matches lmdeploy's generic fused-MoE runtime and the vLLM integration strategy we
    are aligning to.
    """

    def __init__(self,
                 num_experts: int,
                 hidden_dim: int,
                 ffn_dim: int,
                 weight_type: str,
                 device: torch.device):
        super().__init__()
        tp, rank = get_tp_world_rank('moe')
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.weight_type = weight_type
        self.tp = tp
        self.rank = rank
        self.local_ffn_dim = _split_size(ffn_dim, tp, 128)[rank]
        self.ffn_split_sizes = _split_size(ffn_dim, tp, 128)

        if weight_type == 'gate_up':
            weight = torch.empty((num_experts, self.local_ffn_dim * 2, hidden_dim // 2),
                                  dtype=torch.int8, device=device)
            scale = torch.empty((num_experts, self.local_ffn_dim * 2, hidden_dim // 32),
                                dtype=torch.float8_e8m0fnu,
                                device=device)
        elif weight_type == 'down':
            weight = torch.empty((num_experts, hidden_dim, self.local_ffn_dim // 2),
                                 dtype=torch.int8, device=device)
            scale = torch.empty((num_experts, hidden_dim, self.local_ffn_dim // 32),
                                dtype=torch.float8_e8m0fnu,
                                device=device)
        else:
            raise RuntimeError(f'Unsupported V4 expert weight_type: {weight_type}')

        self.register_parameter('weight', nn.Parameter(weight, requires_grad=False))
        self.register_parameter('scale', nn.Parameter(scale, requires_grad=False))
        self.weight.weight_loader = self.weight_loader
        self.scale.weight_loader = self.scale_loader

    def _split_loaded_weight(self, loaded_weight: torch.Tensor, dim: int, sizes: list[int]):
        return loaded_weight.split(sizes, dim=dim)[self.rank].contiguous()

    def _chunk_loaded_weight(self, loaded_weight: torch.Tensor, shard_id: str, is_scale: bool):
        if self.tp == 1:
            return loaded_weight
        if self.weight_type == 'gate_up':
            assert shard_id in ('gate', 'up')
            return self._split_loaded_weight(loaded_weight, dim=0, sizes=self.ffn_split_sizes)
        assert shard_id == 'down'
        if is_scale:
            sizes = [size // 32 for size in self.ffn_split_sizes]
        else:
            sizes = [size // 2 for size in self.ffn_split_sizes]
        return self._split_loaded_weight(loaded_weight, dim=1, sizes=sizes)

    def _get_target_slice(self, param: nn.Parameter, expert_id: int, shard_id: str):
        if self.weight_type == 'gate_up':
            half_out = self.local_ffn_dim
            if shard_id == 'gate':
                return param.data[expert_id, :half_out]
            if shard_id == 'up':
                return param.data[expert_id, half_out:]
        elif self.weight_type == 'down' and shard_id == 'down':
            return param.data[expert_id]
        raise RuntimeError(f'Unsupported shard_id={shard_id} for weight_type={self.weight_type}')

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, expert_id: int, shard_id: str):
        loaded_weight = self._chunk_loaded_weight(loaded_weight, shard_id, is_scale=False)
        target = self._get_target_slice(param, expert_id, shard_id)
        target.copy_(loaded_weight.to(target.device, dtype=target.dtype))

    def scale_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, expert_id: int, shard_id: str):
        loaded_weight = self._chunk_loaded_weight(loaded_weight, shard_id, is_scale=True)
        target = self._get_target_slice(param, expert_id, shard_id)
        target.copy_(loaded_weight.to(target.device, dtype=target.dtype))


class FusedMoEV4FP4(nn.Module):
    """Fused MoE for DeepSeek-V4 routed experts with TP + EP support.

    Follows the same pattern as FusedMoEBlockedF8: one class handles both
    TP and EP modes internally. The weight attribute names (gate_up, down)
    are the same regardless of mode; the weight_loader on each parameter
    handles TP/EP sharding internally.

    - TP mode (ep_size == 1): V4ExpertTPWeights — all experts, FFN dim sharded by TP rank.
      Uses TritonFusedMoEV4TPImpl + moe_reduce all-reduce.
    - EP mode (ep_size > 1): V4ExpertWeights — subset of experts, full FFN dim.
      Uses TritonFusedMoEV4FP4EPImpl (handles dispatch/combine internally, no all-reduce).
    """

    def __init__(self,
                 hidden_dim: int,
                 ffn_dim: int,
                 num_experts: int,
                 top_k: int,
                 swiglu_limit: float = 0.0,
                 device: torch.device | None = None):
        super().__init__()
        device = device or torch.device('cpu')

        dist_ctx = get_dist_manager().current_context()
        dist_config = dist_ctx.dist_config

        self.ep_size, ep_rank = get_ep_world_rank()
        tp, tp_rank = get_tp_world_rank('moe')
        _, tp_mode = dist_config.get_tp_by_layer('moe')

        self.tp = tp
        self.tp_rank = tp_rank
        self.tp_mode = tp_mode
        self.tp_group = dist_ctx.moe_tp_group.gpu_group
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.top_k = top_k

        impl_builder = get_backend().get_layer_impl_builder(OpType.FusedMoEV4FP4)

        if self.ep_size > 1:
            # EP mode: each rank holds num_experts // ep_size experts.
            if num_experts % self.ep_size != 0:
                raise RuntimeError(f'Number of experts ({num_experts}) must be divisible by ep ({self.ep_size}).')
            num_local_experts = num_experts // self.ep_size
            expert_offset = ep_rank * num_local_experts
            self.expert_offset = expert_offset

            self.gate_up = V4ExpertWeights(num_local_experts,
                                           hidden_dim,
                                           ffn_dim * 2,
                                           expert_offset=expert_offset,
                                           weight_type='gate_up',
                                           device=device)
            self.down = V4ExpertWeights(num_local_experts,
                                        ffn_dim,
                                        hidden_dim,
                                        expert_offset=expert_offset,
                                        weight_type='down',
                                        device=device)

            self.impl = impl_builder.build(top_k=top_k,
                                           num_experts=num_experts,
                                           hidden_dim=hidden_dim,
                                           ffn_dim=ffn_dim,
                                           expert_offset=expert_offset,
                                           swiglu_limit=swiglu_limit,
                                           ep_size=self.ep_size,
                                           ep_group=dist_ctx.ep_gpu_group)
        else:
            # TP-only mode: all experts, FFN dim sharded by TP rank.
            self.expert_offset = 0
            local_ffn_dim = _split_size(ffn_dim, tp, 128)[tp_rank]

            self.gate_up = V4ExpertTPWeights(num_experts,
                                             hidden_dim,
                                             ffn_dim,
                                             weight_type='gate_up',
                                             device=device)
            self.down = V4ExpertTPWeights(num_experts,
                                          hidden_dim,
                                          ffn_dim,
                                          weight_type='down',
                                          device=device)

            self.impl = impl_builder.build(top_k=top_k,
                                           num_experts=num_experts,
                                           hidden_dim=hidden_dim,
                                           ffn_dim=local_ffn_dim,
                                           expert_offset=0,
                                           swiglu_limit=swiglu_limit,
                                           scale_fmt='ue8m0')

    def update_weights(self):
        if self.ep_size > 1:
            gate_up_weight, gate_up_scale, down_weight, down_scale = self.impl.update_weights(
                self.gate_up.weight, self.gate_up.scale, self.down.weight, self.down.scale)
            self.gate_up.update(gate_up_weight, gate_up_scale)
            self.down.update(down_weight, down_scale)
        else:
            self.impl.update_weights(self.gate_up.weight,
                                     self.gate_up.scale,
                                     self.down.weight,
                                     self.down.scale)

    def forward(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.LongTensor):
        out = self.impl.forward(hidden_states,
                                topk_weights,
                                topk_ids,
                                self.gate_up.weight,
                                self.gate_up.scale,
                                self.down.weight,
                                self.down.scale)
        if self.ep_size > 1:
            # EP: impl handles dispatch/combine, no all-reduce needed.
            return out
        return moe_reduce(out, rank=self.tp_rank, tp_mode=self.tp_mode, group=self.tp_group)

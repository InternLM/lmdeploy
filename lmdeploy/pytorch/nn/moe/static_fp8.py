# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.pytorch.backends import OpType, get_backend

from .base import (
    FusedMoEBase,
    MoeType,
    moe_gather_inputs,
    moe_reduce,
    update_dims,
)
from .default import LinearWeights


class LinearWeightsStaticF8(LinearWeights):
    """Static per-tensor FP8 fused MoE weights."""

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        weight_type: str,
        device: torch.device,
        expert_list: list[int] = None,
        quant_dtype: torch.dtype = torch.float8_e4m3fn,
    ):
        if expert_list is not None:
            raise RuntimeError(
                'Static FP8 MoE does not support EP mode yet.'
            )

        super().__init__(
            num_experts=num_experts,
            in_features=in_features,
            out_features=out_features,
            weight_type=weight_type,
            dtype=quant_dtype,
            device=device,
            expert_list=expert_list,
        )

        weight_scale = torch.empty(
            (num_experts, out_features, 1),
            dtype=torch.float32,
            device=device,
        )
        weight_scale = torch.nn.Parameter(
            weight_scale,
            requires_grad=False,
        )
        self.register_parameter(
            'weight_scale',
            weight_scale,
        )

        input_scale = torch.empty(
            (1,),
            dtype=torch.float32,
            device=device,
        )
        input_scale = torch.nn.Parameter(
            input_scale,
            requires_grad=False,
        )
        self.register_parameter(
            'input_scale',
            input_scale,
        )

        self.weight_scale.weight_loader = (
            self.weight_loader_scale_tp
        )
        self.input_scale.weight_loader = (
            self.input_scale_loader
        )

        self._input_scale_loaded = False

    def weight_loader_scale_tp(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
        shard_id: str,
    ):
        """Load a per-tensor expert weight scale."""
        if loaded_weight.numel() != 1:
            raise ValueError(
                'Static FP8 weight scale must contain '
                f'one value, but got {loaded_weight.shape}.'
            )

        # The scale is scalar, so TP does not need to split it.
        # It is broadcast to the local output channels.
        if shard_id == 'gate':
            param_data = param.data[
                expert_id,
                :self.half_out,
            ]
        elif shard_id == 'up':
            param_data = param.data[
                expert_id,
                self.half_out:,
            ]
        elif shard_id == 'down':
            param_data = param.data[expert_id]
        else:
            raise RuntimeError(
                f'Unknown shard_id: {shard_id}'
            )

        scale = loaded_weight.to(
            device=param_data.device,
            dtype=param_data.dtype,
        ).reshape(1, 1)

        param_data.copy_(
            scale.expand_as(param_data)
        )

    def input_scale_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
        shard_id: str,
    ):
        """Load the shared static activation scale."""
        del expert_id, shard_id

        if loaded_weight.numel() != 1:
            raise ValueError(
                'Static FP8 input scale must contain '
                f'one value, but got {loaded_weight.shape}.'
            )

        scale = loaded_weight.to(
            device=param.device,
            dtype=param.dtype,
        ).reshape_as(param)

        if not self._input_scale_loaded:
            param.data.copy_(scale)
            self._input_scale_loaded = True
            return

        if not torch.equal(param.data, scale):
            raise ValueError(
                'Packed static FP8 projections must share '
                'the same input scale.'
            )

    def update_weight(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor,
    ):
        """Replace updated weights while preserving loaders."""
        super().update_weight(weight)

        weight_scale_loader = (
            self.weight_scale.weight_loader
        )
        weight_scale = torch.nn.Parameter(
            weight_scale,
            requires_grad=False,
        )
        weight_scale.weight_loader = (
            weight_scale_loader
        )
        self.register_parameter(
            'weight_scale',
            weight_scale,
        )

        input_scale_loader = (
            self.input_scale.weight_loader
        )
        input_scale = torch.nn.Parameter(
            input_scale,
            requires_grad=False,
        )
        input_scale.weight_loader = (
            input_scale_loader
        )
        self.register_parameter(
            'input_scale',
            input_scale,
        )

class FusedMoEStaticF8(FusedMoEBase):
    """Fused MoE with static per-tensor FP8 quantization."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int,
        renormalize: bool = False,
        dtype: torch.dtype | None = None,
        quant_dtype: torch.dtype = torch.float8_e4m3fn,
        device: torch.device | None = None,
        all_reduce: bool = True,
    ):
        device = device or torch.device('cpu')
        dtype = dtype or torch.float16

        self.init_dist_args(all_reduce)

        if self.ep > 1:
            raise RuntimeError(
                'FusedMoEStaticF8 does not support '
                'EP mode yet.'
            )

        super().__init__(
            tp=self.tp,
            tp_mode=self.tp_mode,
            do_renormalize=renormalize,
        )

        impl_builder = (
            get_backend().get_layer_impl_builder(
                OpType.FusedMoEStaticF8
            )
        )

        self.impl = impl_builder.build(
            top_k=top_k,
            num_experts=num_experts,
            renormalize=renormalize,
            out_dtype=dtype,
            quant_dtype=quant_dtype,
        )

        # TP shards the intermediate dimension.
        hidden_dim, ffn_dim = update_dims(
            hidden_dim,
            ffn_dim,
        )

        self.expert_list = None

        self.gate_up = LinearWeightsStaticF8(
            num_experts=num_experts,
            in_features=hidden_dim,
            out_features=ffn_dim * 2,
            weight_type='gate_up',
            device=device,
            expert_list=None,
            quant_dtype=quant_dtype,
        )

        self.down = LinearWeightsStaticF8(
            num_experts=num_experts,
            in_features=ffn_dim,
            out_features=hidden_dim,
            weight_type='down',
            device=device,
            expert_list=None,
            quant_dtype=quant_dtype,
        )

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.dtype = dtype
        self.device = device
        self.quant_dtype = quant_dtype

    def update_weights(self):
        """Update weights through the backend implementation."""
        (
            gate_up_weight,
            gate_up_weight_scale,
            gate_up_input_scale,
            down_weight,
            down_weight_scale,
            down_input_scale,
        ) = self.impl.update_weights(
            self.gate_up.weight,
            self.gate_up.weight_scale,
            self.gate_up.input_scale,
            self.down.weight,
            self.down.weight_scale,
            self.down.input_scale,
        )

        self.gate_up.update_weight(
            gate_up_weight,
            gate_up_weight_scale,
            gate_up_input_scale,
        )

        self.down.update_weight(
            down_weight,
            down_weight_scale,
            down_input_scale,
        )

    def dispatch(self, state: dict):
        """Gather MoE inputs before computation."""
        moe_type = state['moe_type']

        if moe_type != MoeType.Default:
            raise NotImplementedError(
                f'Not supported MoE type: {moe_type}'
            )

        (
            hidden_states,
            topk_weights,
            topk_idx,
        ) = moe_gather_inputs(
            state['hidden_states'],
            state['topk_weights'],
            state['topk_idx'],
            group=self.gather_group,
        )

        return {
            'hidden_states': hidden_states,
            'topk_weights': topk_weights,
            'topk_idx': topk_idx,
            'moe_type': moe_type,
        }

    def gemm(self, state: dict):
        """Run static FP8 expert computation."""
        output = self.impl.forward(
            state['hidden_states'],
            state['topk_weights'],
            state['topk_idx'],
            self.gate_up.weight,
            self.gate_up.weight_scale,
            self.gate_up.input_scale,
            self.down.weight,
            self.down.weight_scale,
            self.down.input_scale,
            expert_list=self.expert_list,
        )

        return {
            'hidden_states': output,
            'moe_type': state['moe_type'],
        }

    def combine(self, state: dict):
        """Reduce TP expert outputs."""
        moe_type = state['moe_type']

        if moe_type != MoeType.Default:
            raise NotImplementedError(
                f'Not supported MoE type: {moe_type}'
            )

        hidden_states = state['hidden_states']

        if self.all_reduce:
            hidden_states = moe_reduce(
                hidden_states,
                rank=self.tp_rank,
                tp_mode=self.tp_mode,
                group=self.tp_group,
            )

        return {
            'hidden_states': hidden_states,
            'moe_type': moe_type,
        }

    def wait(self, state: dict):
        """Static FP8 MoE has no asynchronous state."""
        raise NotImplementedError

# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.pytorch.backends.moe import (
    FusedMoEStaticF8BuildSpec,
    FusedMoEStaticF8Impl,
)
from lmdeploy.pytorch.kernels.cuda.moe.w8a8 import (
    fused_moe_static_fp8,
)


class TritonFusedMoEStaticF8Impl(FusedMoEStaticF8Impl):
    """Triton static FP8 fused MoE implementation."""

    def __init__(
        self,
        top_k: int,
        num_experts: int,
        renormalize: bool = False,
        out_dtype: torch.dtype = torch.float16,
        quant_dtype: torch.dtype = torch.float8_e4m3fn,
    ):
        self.top_k = top_k
        self.num_experts = num_experts
        self.renormalize = renormalize
        self.out_dtype = out_dtype
        self.quant_dtype = quant_dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.LongTensor,
        gate_up_weights: torch.Tensor,
        gate_up_weight_scale: torch.Tensor,
        gate_up_input_scale: torch.Tensor,
        down_weights: torch.Tensor,
        down_weight_scale: torch.Tensor,
        down_input_scale: torch.Tensor,
        expert_list: list[int] = None,
    ):
        """Forward."""
        hidden_states = hidden_states.contiguous()

        expert_offset = 0
        num_experts = None

        if (
            expert_list is not None
            and len(expert_list) != self.num_experts
        ):
            expert_offset = expert_list[0]
            num_experts = self.num_experts

        return fused_moe_static_fp8(
            hidden_states,
            gate_up_input_scale,
            gate_up_weights,
            gate_up_weight_scale,
            down_input_scale,
            down_weights,
            down_weight_scale,
            topk_weights,
            topk_ids,
            topk=self.top_k,
            out_dtype=self.out_dtype,
            quant_dtype=self.quant_dtype,
            expert_offset=expert_offset,
            num_experts=num_experts,
            renormalize=self.renormalize,
        )


def _build_fused_moe_static_f8(spec: FusedMoEStaticF8BuildSpec) -> FusedMoEStaticF8Impl:
    """Build a CUDA static-FP8 fused MoE implementation."""
    return TritonFusedMoEStaticF8Impl(
        top_k=spec.top_k,
        num_experts=spec.num_experts,
        renormalize=spec.renormalize,
        out_dtype=spec.output_dtype,
        quant_dtype=spec.quant_dtype,
    )

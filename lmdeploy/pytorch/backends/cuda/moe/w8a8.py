# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

import torch

from lmdeploy.pytorch.backends.moe import FusedMoEW8A8Builder, FusedMoEW8A8Impl
from lmdeploy.pytorch.kernels.cuda import fused_moe_w8a8
from lmdeploy.pytorch.kernels.cuda.w8a8_triton_kernels import per_token_quant_int8
from lmdeploy.pytorch.models.q_modules import QTensor
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class TritonFusedMoEW8A8Impl(FusedMoEW8A8Impl):
    """Triton fused moe w8a8 implementation."""

    def __init__(
        self,
        top_k: int,
        num_experts: int,
        renormalize: bool = False,
        out_dtype: torch.dtype = torch.float16,
        quant_dtype: torch.dtype = torch.int8,
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.renormalize = renormalize
        self.out_dtype = out_dtype
        self.quant_dtype = quant_dtype

    def update_weights(self, gate_up_weights: torch.Tensor, down_weights: torch.Tensor, gate_up_scale: torch.Tensor,
                       down_scale: torch.Tensor):
        # do not transpose weight for int8/fp8
        return gate_up_weights, down_weights, gate_up_scale, down_scale

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                gate_up_scale: torch.Tensor,
                down_weights: torch.Tensor,
                down_scale: torch.Tensor,
                expert_list: List[int] = None):
        """forward."""

        if isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states.contiguous()
            input_quant, input_scale = per_token_quant_int8(hidden_states, 1e-7, quant_dtype=self.quant_dtype)
        else:
            assert isinstance(hidden_states, QTensor)
            input_quant, input_scale = (hidden_states.tensor, hidden_states.scale)

        expert_offset = 0
        num_experts = None
        if expert_list is not None and len(expert_list) != self.num_experts:
            expert_offset = expert_list[0]
            num_experts = self.num_experts
        return fused_moe_w8a8(input_quant,
                              input_scale,
                              gate_up_weights,
                              gate_up_scale,
                              down_weights,
                              down_scale,
                              topk_weights=topk_weights,
                              topk_ids=topk_ids,
                              topk=self.top_k,
                              out_dtype=self.out_dtype,
                              quant_dtype=self.quant_dtype,
                              expert_offset=expert_offset,
                              num_experts=num_experts,
                              renormalize=self.renormalize)


class TritonFusedMoEW8A8Builder(FusedMoEW8A8Builder):
    """Triton fused moe w8a8 builder."""

    @staticmethod
    def build(
        top_k: int,
        num_experts: int,
        renormalize: bool = False,
        out_dtype: torch.dtype = torch.float16,
        quant_dtype: torch.dtype = torch.int8,
    ):
        """Build from mlp."""
        return TritonFusedMoEW8A8Impl(top_k=top_k,
                                      num_experts=num_experts,
                                      renormalize=renormalize,
                                      out_dtype=out_dtype,
                                      quant_dtype=quant_dtype)

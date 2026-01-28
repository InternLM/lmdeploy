# Copyright (c) OpenMMLab. All rights reserved.

from typing import Callable, List

import torch
import torch.distributed as dist

from lmdeploy.pytorch.backends.deepep_moe_checker import get_moe_backend
from lmdeploy.pytorch.backends.moe import FusedMoEBlockedF8Builder, FusedMoEBlockedF8Impl
from lmdeploy.pytorch.distributed import get_dist_manager
from lmdeploy.pytorch.kernels.cuda.blocked_fp8_fused_moe import fused_moe_blocked_fp8
from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import quant_fp8
from lmdeploy.pytorch.kernels.cuda.fused_moe import _renormalize
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager
from lmdeploy.utils import get_logger

from .ep_utils import gather_outputs_by_attn_tp, split_inputs_by_attn_tp

logger = get_logger('lmdeploy')


class TritonFusedMoEBlockedF8Impl(FusedMoEBlockedF8Impl):
    """Triton fused moe blocked f8 implementation."""

    def __init__(self,
                 top_k: int,
                 num_experts: int,
                 renormalize: bool = False,
                 block_size: int = 128,
                 out_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.renormalize = renormalize
        self.block_size = block_size
        self.out_dtype = out_dtype

    def ep_expert_list(self, world_size: int, rank: int):
        """Experts list of current rank."""
        num_experts = self.num_experts
        expert_per_rank = (num_experts + world_size - 1) // world_size
        first_expert = rank * expert_per_rank
        last_expert = min(first_expert + expert_per_rank, num_experts)
        return list(range(first_expert, last_expert))

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                gate_up_scale: torch.Tensor,
                down_weights: torch.Tensor,
                down_scale: torch.Tensor,
                gate_up_bias: torch.Tensor = None,
                down_bias: torch.Tensor = None,
                expert_list: List[int] = None,
                act_func: Callable = None):
        """forward."""
        input_size = hidden_states.shape
        hidden_states = hidden_states.flatten(0, -2)
        input_quant, input_scale = quant_fp8(hidden_states,
                                             self.block_size,
                                             dtype=gate_up_weights.dtype,
                                             scale_fmt=self.scale_fmt)
        expert_offset = 0
        num_experts = None
        if expert_list is not None and len(expert_list) != self.num_experts:
            expert_offset = expert_list[0]
            num_experts = self.num_experts
        output = fused_moe_blocked_fp8(input_quant,
                                       input_scale,
                                       gate_up_weights,
                                       gate_up_scale,
                                       down_weights,
                                       down_scale,
                                       topk_weights=topk_weights,
                                       topk_ids=topk_ids,
                                       topk=self.top_k,
                                       w1_bias=gate_up_bias,
                                       w2_bias=down_bias,
                                       out_dtype=hidden_states.dtype,
                                       expert_offset=expert_offset,
                                       num_experts=num_experts,
                                       renormalize=self.renormalize,
                                       act_func=act_func)
        output = output.unflatten(0, input_size[:-1])
        return output


class FusedDeepEpMoEBlockedF8Impl(TritonFusedMoEBlockedF8Impl):

    def __init__(self,
                 ep_size: int,
                 ep_group: dist.ProcessGroup,
                 top_k: int,
                 num_experts: int,
                 hidden_dim: int,
                 renormalize: bool = False,
                 block_size: int = 128,
                 out_dtype: torch.dtype = torch.bfloat16,
                 layer_idx: int = 0):
        super().__init__(top_k, num_experts, renormalize, block_size, out_dtype)
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.ep_group = ep_group
        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.out_dtype = out_dtype
        self.layer_idx = layer_idx
        try:
            import deep_gemm  # noqa: F401
            self.use_deep_gemm = True
        except ImportError:
            self.use_deep_gemm = False
            logger.warning('For higher performance, please install DeepGEMM https://github.com/deepseek-ai/DeepGEMM')

        try:
            from dlblas.layers.moe.token_dispatcher import DeepEPBuffer, DeepEPMode, use_deepep  # noqa: F401
            get_moe_backend().set_deepep_moe_backend()
            if hasattr(DeepEPBuffer, 'set_explicitly_destroy'):
                DeepEPBuffer.set_explicitly_destroy()
        except ImportError:
            logger.warning('For higher performance, please install DeepEP https://github.com/deepseek-ai/DeepEP')

        # pre-allocate buffer
        self.fusedmoe_build(True)

    def ep_expert_list(self, world_size: int, rank: int):
        """Experts list of current rank."""
        if get_dist_manager().current_context().dist_config.enable_eplb:
            from dlblas.layers.moe.eplb import get_eplb_phy2log_metadata_by_layer
            phy2log = get_eplb_phy2log_metadata_by_layer(self.layer_idx)
            expert_per_rank = (self.num_experts + world_size - 1) // world_size
            first_expert = rank * expert_per_rank
            last_expert = min(first_expert + expert_per_rank, self.num_experts)
            sliced_phy2log = phy2log[first_expert:last_expert].tolist()
            return sliced_phy2log
        else:
            return super().ep_expert_list(world_size=world_size, rank=rank)

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                gate_up_scale: torch.Tensor,
                down_weights: torch.Tensor,
                down_scale: torch.Tensor,
                gate_up_bias: torch.Tensor = None,
                down_bias: torch.Tensor = None,
                expert_list: List[int] = None,
                act_func: Callable = None,
                **kwargs):
        """forward."""
        hidden_states, topk_weights, topk_ids, split_size = split_inputs_by_attn_tp(hidden_states, topk_weights,
                                                                                    topk_ids)

        topk_weights = self.do_renormalize(topk_weights)
        step_ctx = get_step_ctx_manager().current_context()
        low_latency_mode = step_ctx.is_decoding and self.use_deep_gemm
        moe = self.fusedmoe_build(low_latency_mode)
        out_states = moe.forward(hidden_states, topk_weights, topk_ids, gate_up_weights, gate_up_scale, down_weights,
                                 down_scale, expert_list)

        out_states = gather_outputs_by_attn_tp(out_states, split_size)
        return out_states

    def do_renormalize(self, topk_weights):
        return _renormalize(topk_weights, self.renormalize)

    def fusedmoe_build(self, low_latency_mode: bool = False):
        from dlblas.layers.moe.ep_moe import build_deepep_moe
        deepep_moe = build_deepep_moe(low_latency_mode,
                                      self.ep_size,
                                      self.ep_group,
                                      self.num_experts,
                                      self.hidden_dim,
                                      self.block_size,
                                      self.top_k,
                                      self.out_dtype,
                                      layer_idx=self.layer_idx,
                                      chunk_size=16 * 1024)
        return deepep_moe


class TritonFusedMoEBlockedF8Builder(FusedMoEBlockedF8Builder):
    """Triton fused moe blocked f8 builder."""

    @staticmethod
    def build(top_k: int,
              num_experts: int,
              hidden_dim: int = 1,
              renormalize: bool = False,
              block_size: int = 128,
              ep_size: int = 1,
              ep_group: dist.ProcessGroup = None,
              out_dtype: torch.dtype = torch.float16,
              layer_idx: int = 0,
              custom_gateup_act: bool = False):
        """Build from mlp."""
        if ep_size > 1:
            assert custom_gateup_act is False, 'Custom gate up activation is not supported in EP MoE.'
            return FusedDeepEpMoEBlockedF8Impl(ep_size=ep_size,
                                               ep_group=ep_group,
                                               top_k=top_k,
                                               num_experts=num_experts,
                                               hidden_dim=hidden_dim,
                                               renormalize=renormalize,
                                               block_size=block_size,
                                               out_dtype=out_dtype,
                                               layer_idx=layer_idx)
        else:
            return TritonFusedMoEBlockedF8Impl(top_k=top_k,
                                               num_experts=num_experts,
                                               renormalize=renormalize,
                                               block_size=block_size,
                                               out_dtype=out_dtype)

# Copyright (c) OpenMMLab. All rights reserved.

from typing import List, Optional

import torch
import torch.distributed as dist

from lmdeploy.pytorch.backends.cuda.token_dispatcher import DeepEPTokenDispatcherLowLatency, TokenDispatcherBuilder
from lmdeploy.pytorch.distributed import get_dist_manager
from lmdeploy.pytorch.kernels.cuda import fused_moe, fused_moe_w8a8
from lmdeploy.pytorch.kernels.cuda.blocked_fp8_fused_moe import fused_moe_blocked_fp8
from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import quant_fp8
from lmdeploy.pytorch.kernels.cuda.ep_moe import (grouped_gemm_triton, silu_and_mul_masked_post_quant_fwd,
                                                  silu_and_mul_triton_kernel)
from lmdeploy.pytorch.kernels.cuda.fused_moe import _renormalize
from lmdeploy.pytorch.kernels.cuda.w8a8_triton_kernels import per_token_quant_int8
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager
from lmdeploy.pytorch.models.q_modules import QTensor
from lmdeploy.utils import get_logger

from ..moe import (FusedMoEBlockedF8Builder, FusedMoEBlockedF8Impl, FusedMoEBuilder, FusedMoEImpl, FusedMoEW8A8Builder,
                   FusedMoEW8A8Impl)

logger = get_logger('lmdeploy')


class TritonFusedMoEImpl(FusedMoEImpl):
    """Triton fused moe implementation."""

    def __init__(self, top_k: int, num_experts: int, renormalize: bool = False):
        self.num_experts = num_experts
        self.top_k = top_k
        self.renormalize = renormalize

    def update_weights(self, gate_up_weights: torch.Tensor, down_weights: torch.Tensor):
        gate_up_weights = gate_up_weights.transpose(1, 2).contiguous().transpose(1, 2)
        down_weights = down_weights.transpose(1, 2).contiguous().transpose(1, 2)
        return gate_up_weights, down_weights

    def support_ep(self):
        """Support expert parallelism."""
        return True

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
                down_weights: torch.Tensor,
                expert_list: List[int] = None):
        """forward."""
        expert_offset = 0
        num_experts = None
        if expert_list is not None and len(expert_list) != self.num_experts:
            expert_offset = expert_list[0]
            num_experts = self.num_experts
        return fused_moe(hidden_states,
                         gate_up_weights,
                         down_weights,
                         topk_weights=topk_weights,
                         topk_ids=topk_ids,
                         topk=self.top_k,
                         expert_offset=expert_offset,
                         num_experts=num_experts,
                         renormalize=self.renormalize)


class TritonFusedMoEBuilder(FusedMoEBuilder):
    """Triton fused moe builder."""

    @staticmethod
    def build(top_k: int, num_experts: int, renormalize: bool = False):
        """Build from mlp."""
        return TritonFusedMoEImpl(top_k=top_k, num_experts=num_experts, renormalize=renormalize)


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

    def support_ep(self):
        """Support expert parallelism."""
        return True

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


class TritonFusedMoEBlockedF8Impl(FusedMoEBlockedF8Impl):
    """Triton fused moe blocked f8 implementation."""

    def __init__(self,
                 top_k: int,
                 num_experts: int,
                 renormalize: bool = False,
                 block_size: int = 128,
                 out_dtype: torch.dtype = torch.float16):
        self.num_experts = num_experts
        self.top_k = top_k
        self.renormalize = renormalize
        self.block_size = block_size
        self.out_dtype = out_dtype

    def support_ep(self):
        """Support expert parallelism."""
        return True

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
                expert_list: List[int] = None):
        """forward."""
        input_size = hidden_states.shape
        hidden_states = hidden_states.flatten(0, -2)
        input_quant, input_scale = quant_fp8(hidden_states, self.block_size, dtype=gate_up_weights.dtype)

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
                                       out_dtype=hidden_states.dtype,
                                       expert_offset=expert_offset,
                                       num_experts=num_experts,
                                       renormalize=self.renormalize)
        output = output.unflatten(0, input_size[:-1])
        return output


class DeepEPExpertsGroupedGEMM:
    """MoE Expert Parallel Impl based on DeepEP (https://github.com/deepseek-
    ai/DeepEP/tree/main)"""

    def __init__(
        self,
        num_experts: int,
        ep_size: int,
        block_shape: list[int],
    ):
        self.num_experts = num_experts
        self.ep_size = ep_size
        assert self.num_experts % self.ep_size == 0
        self.num_experts_per_partition = self.num_experts // self.ep_size
        self.block_shape = block_shape
        self.use_fp8_w8a8 = True

    def forward(self, hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor, gate_up_weight: torch.Tensor,
                gate_up_scale: torch.Tensor, gate_down_weight: torch.Tensor, gate_down_scale: torch.Tensor):
        seg_indptr_cur_rank = torch.cat([
            torch.zeros(1, device=tokens_per_expert.device, dtype=tokens_per_expert.dtype),
            torch.cumsum(tokens_per_expert, dim=0),
        ])
        reorder_topk_ids = torch.repeat_interleave(tokens_per_expert)
        weight_indices_cur_rank = torch.arange(
            0,
            self.num_experts_per_partition,
            device=hidden_states.device,
            dtype=torch.int64,
        )

        # GroupGemm-0
        gateup_output = torch.empty(
            hidden_states.shape[0],
            gate_up_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if hidden_states.shape[0] > 0:
            input, input_scale = quant_fp8(hidden_states, 128, dtype=gate_up_weight.dtype)
            gateup_output = grouped_gemm_triton(
                a=input,
                b=gate_up_weight,
                c=gateup_output,
                batch_size=self.num_experts_per_partition,
                weight_column_major=True,
                seg_indptr=seg_indptr_cur_rank,
                weight_indices=weight_indices_cur_rank,
                use_fp8_w8a8=self.use_fp8_w8a8,
                scale_a=input_scale,
                scale_b=gate_up_scale,
                block_shape=self.block_shape,
            )

        # Act
        down_input = torch.empty(
            gateup_output.shape[0],
            gateup_output.shape[1] // 2,
            device=gateup_output.device,
            dtype=hidden_states.dtype,
        )
        silu_and_mul_triton_kernel[(gateup_output.shape[0], )](
            gateup_output,
            down_input,
            gateup_output.shape[1],
            reorder_topk_ids,
            None,
            0,
            self.num_experts_per_partition - 1,
            BLOCK_SIZE=512,
        )

        # GroupGemm-1
        down_output = torch.empty(
            down_input.shape[0],
            gate_down_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if down_input.shape[0] > 0:
            down_input, down_input_scale = quant_fp8(down_input, 128, dtype=gate_down_weight.dtype)
            down_output = grouped_gemm_triton(
                a=down_input,
                b=gate_down_weight,
                c=down_output,
                batch_size=self.num_experts_per_partition,
                weight_column_major=True,
                seg_indptr=seg_indptr_cur_rank,
                weight_indices=weight_indices_cur_rank,
                use_fp8_w8a8=self.use_fp8_w8a8,
                scale_a=down_input_scale,
                scale_b=gate_down_scale,
                block_shape=self.block_shape,
            )
        return down_output


class DeepEPExpertsDeepGEMM:
    deep_gemm = None

    def __init__(self, num_experts: int, ep_size: int, block_size: int, out_dtype: torch.dtype = torch.bfloat16):
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.num_experts_per_partition = self.num_experts // self.ep_size
        self.block_size = block_size
        self.use_fp8_w8a8 = True
        self.out_dtype = out_dtype

    def forward(
        self,
        hidden_states_fp8,
        gate_up_weight: torch.Tensor,
        gate_up_scale: torch.Tensor,
        gate_down_weight: torch.Tensor,
        gate_down_scale: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
    ):

        gate_up_weight_fp8 = (gate_up_weight, gate_up_scale)
        gate_down_weight_fp8 = (gate_down_weight, gate_down_scale)
        assert (hidden_states_fp8[0].size(0) % 4 == 0), f'TMA alignment error: {hidden_states_fp8[0].size(0)}'
        num_groups, m, k = hidden_states_fp8[0].size()
        n = gate_up_weight.size(1)
        expected_m = min(expected_m, m)
        gateup_output = torch.empty((num_groups, m, n), device=hidden_states_fp8[0].device, dtype=self.out_dtype)
        DeepEPExpertsDeepGEMM.deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(hidden_states_fp8, gate_up_weight_fp8,
                                                                              gateup_output, masked_m, expected_m)
        down_input = torch.empty((
            gateup_output.shape[0],
            gateup_output.shape[1],
            gateup_output.shape[2] // 2,
        ),
                                 device=gateup_output.device,
                                 dtype=gate_down_weight.dtype)

        down_input_scale = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2 // self.block_size,
            ),
            device=gateup_output.device,
            dtype=torch.float32,
        )
        silu_and_mul_masked_post_quant_fwd(
            gateup_output,
            down_input,
            down_input_scale,
            self.block_size,
            masked_m,
        )
        n = gate_down_weight.size(1)
        down_input_fp8 = (
            down_input,
            DeepEPExpertsDeepGEMM.deep_gemm.get_col_major_tma_aligned_tensor(down_input_scale),
        )
        down_output = torch.empty((num_groups, m, n), device=down_input.device, dtype=self.out_dtype)
        DeepEPExpertsDeepGEMM.deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(down_input_fp8, gate_down_weight_fp8,
                                                                              down_output, masked_m, expected_m)
        return down_output


class FusedMoENormal:

    def __init__(self,
                 ep_size: int,
                 ep_group: dist.ProcessGroup,
                 num_experts: int,
                 hidden_dim: int,
                 block_size: int = 128,
                 out_dtype: torch.dtype = torch.bfloat16):
        self.experts = DeepEPExpertsGroupedGEMM(num_experts, ep_size, [block_size, block_size])
        self.token_dispatcher = TokenDispatcherBuilder.build(
            group=ep_group,
            num_experts=num_experts,
            num_local_experts=num_experts // ep_size,
            hidden_size=hidden_dim,
            params_dtype=out_dtype,
        )

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                up_weights: torch.Tensor,
                up_scale: torch.Tensor,
                down_weights: torch.Tensor,
                down_scale: torch.Tensor,
                expert_list: List[int] = None):
        """forward."""
        recv_hidden_states, recv_topk_ids, recv_topk_weights, tokens_per_expert = self.token_dispatcher.dispatch(
            hidden_states,
            topk_ids,
            topk_weights,
            expert_list,
        )
        out_states = self.experts.forward(recv_hidden_states, tokens_per_expert, up_weights, up_scale, down_weights,
                                          down_scale)
        out_states = self.token_dispatcher.combine(out_states)
        return out_states

    def capture(self):
        return self.token_dispatcher.buffer_normal.capture()

    def wait(self, event):
        self.token_dispatcher.release()
        event.current_stream_wait()

    def dispatch_async(self,
                       x: torch.Tensor,
                       topk_idx: torch.Tensor,
                       topk_weights: torch.Tensor,
                       num_experts: Optional[int] = None,
                       previous_event=None,
                       async_finish=True):
        return self.token_dispatcher.dispatch_normal_async(x, topk_idx, topk_weights, num_experts, previous_event,
                                                           async_finish)

    def combine_async(self, x: torch.Tensor, handle: tuple, previous_event=None, async_finish=True):
        return self.token_dispatcher.combine_normal_async(x, handle, previous_event, async_finish)

    def release(self):
        return self.token_dispatcher.release()

    def fusedmoe_forward(self, state, up_weight, up_scale, down_weight, down_scale):
        (
            hidden_states,
            recv_hidden_states_shape,
            dispatched_routing_map,
            topk_weights,
            reversed_mapping_for_combine,
        ) = self.token_dispatcher.get_permuted_hidden_states_by_experts(state['recv_hidden_states'],
                                                                        state['recv_topk_idx'],
                                                                        state['recv_topk_weights'],
                                                                        state['num_experts'])
        tokens_per_expert = torch.tensor(
            state['recv_tokens_per_expert'],
            device=hidden_states.device,
            dtype=torch.int64,
        )
        hidden_states = self.experts.forward(hidden_states, tokens_per_expert, up_weight, up_scale, down_weight,
                                             down_scale)
        hidden_states = self.token_dispatcher.get_restored_hidden_states_by_experts(hidden_states,
                                                                                    reversed_mapping_for_combine,
                                                                                    recv_hidden_states_shape,
                                                                                    dispatched_routing_map,
                                                                                    topk_weights)
        return hidden_states


class FusedMoELowLatency:

    def __init__(self,
                 ep_size: int,
                 ep_group: dist.ProcessGroup,
                 num_experts: int,
                 hidden_dim: int,
                 block_size: int = 128,
                 out_dtype: torch.dtype = torch.bfloat16):
        self.num_experts = num_experts
        self.experts = DeepEPExpertsDeepGEMM(num_experts, ep_size, block_size, out_dtype)
        self.token_dispatcher = DeepEPTokenDispatcherLowLatency(
            group=ep_group,
            num_experts=num_experts,
            num_local_experts=num_experts // ep_size,
            hidden_size=hidden_dim,
            params_dtype=out_dtype,
        )

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                up_weights: torch.Tensor,
                up_scale: torch.Tensor,
                down_weights: torch.Tensor,
                down_scale: torch.Tensor,
                expert_list: List[int] = None):
        """forward."""
        recv_hidden_states, topk_idx, topk_weights, masked_m, expected_m = self.token_dispatcher.dispatch(
            hidden_states,
            topk_ids,
            topk_weights,
            self.num_experts,
        )
        out_states = self.experts.forward(recv_hidden_states, up_weights, up_scale, down_weights, down_scale, masked_m,
                                          expected_m)
        out_states = self.token_dispatcher.combine(out_states, topk_idx, topk_weights)
        return out_states

    def wait(self, event):
        event.current_stream_wait()

    def dispatch_async(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        num_experts: Optional[int] = None,
        use_fp8: bool = True,
        async_finish: bool = True,
    ):
        return self.token_dispatcher.dispatch_async(hidden_states, topk_idx, num_experts, use_fp8, async_finish)

    def combine_async(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: tuple,
        async_finish: bool,
    ):
        return self.token_dispatcher.combine_async(hidden_states, topk_idx, topk_weights, handle, async_finish)

    def fusedmoe_forward(self, state, up_weight, up_scale, down_weight, down_scale):
        recv_hidden_states = state['recv_hidden_states']
        recv_expert_count = state['recv_expert_count']
        hidden_shape = state['raw_hidden_shape']
        topk_idx = state['topk_idx']
        expected_m = (hidden_shape[0] * self.token_dispatcher.buffer_low_latency.group_size * topk_idx.shape[1] +
                      self.token_dispatcher.num_experts) // self.token_dispatcher.num_experts
        hidden_states = self.experts.forward(recv_hidden_states, up_weight, up_scale, down_weight, down_scale,
                                             recv_expert_count, expected_m)
        return hidden_states


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
            import deep_gemm
            DeepEPExpertsDeepGEMM.deep_gemm = deep_gemm
            self.use_deep_gemm = True
        except ImportError:
            self.use_deep_gemm = False
            logger.warning('For higher performance, please install DeepGEMM https://github.com/deepseek-ai/DeepGEMM')

        try:
            from dlblas.layers.moe.ep_moe import build_deepep_moe
            from dlblas.layers.moe.eplb import get_eplb_phy2log_metadata_by_layer
            self.use_dlblas = True
            self.build_deepep_moe = build_deepep_moe
            self.get_eplb_phy2log_metadata_by_layer = get_eplb_phy2log_metadata_by_layer
        except ImportError:
            raise ImportError('To enable ep, please install dlBLAS https://github.com/DeepLink-org/dlBLAS')

    def ep_expert_list(self, world_size: int, rank: int):
        """Experts list of current rank."""
        if get_dist_manager().current_context().dist_config.enable_eplb:
            assert self.use_dlblas, 'Please install dlBLAS https://github.com/DeepLink-org/dlBLAS'
            phy2log = self.get_eplb_phy2log_metadata_by_layer(self.layer_idx)
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
                expert_list: List[int] = None):
        """forward."""
        topk_weights = self.do_renormalize(topk_weights)
        step_ctx = get_step_ctx_manager().current_context()
        low_latency_mode = step_ctx.is_decoding and self.use_deep_gemm
        moe = self.fusedmoe_build(low_latency_mode)
        out_states = moe.forward(hidden_states, topk_weights, topk_ids, gate_up_weights, gate_up_scale, down_weights,
                                 down_scale, expert_list)
        return out_states

    def do_renormalize(self, topk_weights):
        return _renormalize(topk_weights, self.renormalize)

    def fusedmoe_build(self, low_latency_mode: bool = False):
        if self.use_dlblas:
            return self.build_deepep_moe(low_latency_mode,
                                         self.ep_size,
                                         self.ep_group,
                                         self.num_experts,
                                         self.hidden_dim,
                                         self.block_size,
                                         self.top_k,
                                         self.out_dtype,
                                         layer_idx=self.layer_idx,
                                         chunk_size=16 * 1024)
        elif low_latency_mode:
            return FusedMoELowLatency(self.ep_size, self.ep_group, self.num_experts, self.hidden_dim, self.block_size,
                                      self.out_dtype)
        else:
            return FusedMoENormal(self.ep_size, self.ep_group, self.num_experts, self.hidden_dim, self.block_size,
                                  self.out_dtype)


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
              layer_idx: int = 0):
        """Build from mlp."""
        if ep_size > 1:
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

# Copyright (c) OpenMMLab. All rights reserved.

from collections.abc import Callable

import torch
import torch.distributed as dist

from lmdeploy.pytorch.backends.cuda.token_dispatcher import (
    DeepEPBuffer,
    DeepEPTokenDispatcherLowLatency,
    DeepEPTokenDispatcherNormal,
    DisposibleTensor,
    use_deepep,
)
from lmdeploy.pytorch.backends.deepep_state import get_deepep_state
from lmdeploy.pytorch.backends.moe import FusedMoEBlockedF8Builder, FusedMoEBlockedF8Impl
from lmdeploy.pytorch.distributed import get_dist_manager
from lmdeploy.pytorch.kernels.cuda.activation import silu_and_mul_masked_post_quant_fwd
from lmdeploy.pytorch.kernels.cuda.blocked_fp8_fused_moe import fused_moe_blocked_fp8
from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import per_token_group_quant_fp8, quant_fp8
from lmdeploy.pytorch.kernels.cuda.fused_moe import _renormalize
from lmdeploy.pytorch.kernels.cuda.fused_moe_ep_fp8 import fused_moe_v3_fp8
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager
from lmdeploy.utils import get_logger

from .ep_utils import gather_outputs_by_attn_tp, split_inputs_by_attn_tp

logger = get_logger('lmdeploy')


class FusedMoENormal:

    def __init__(
        self,
        ep_size: int,
        ep_group: dist.ProcessGroup,
        num_experts: int,
        hidden_dim: int,
        layer_index: int = 0,
        block_size: int = 128,
        top_k: int = 8,
        out_dtype: torch.dtype = torch.bfloat16,
        fp8_dtype: torch.dtype | None = None,
        scale_fmt: str | None = None,
        num_max_dispatch_tokens_per_rank: int = 128,
        chunk_size: int | None = 32 * 1024,
        expert_alignment: int = 128,
    ):
        self.layer_index = layer_index
        self.top_k = top_k
        self.num_experts = num_experts
        self.block_size = block_size
        self.num_local_experts = num_experts // ep_size
        self.out_dtype = out_dtype
        self.fp8_dtype = fp8_dtype
        self.scale_fmt = scale_fmt
        self.token_dispatcher = DeepEPTokenDispatcherNormal(
            group=ep_group,
            num_experts=num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=hidden_dim,
            params_dtype=out_dtype,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            expert_alignment=expert_alignment,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.LongTensor,
        up_weights: torch.Tensor,
        up_scale: torch.Tensor,
        down_weights: torch.Tensor,
        down_scale: torch.Tensor,
        expert_list: list[int] = None,
    ):
        hs_quant, hs_scale = per_token_group_quant_fp8(hidden_states,
                                                       self.block_size,
                                                       dtype=up_weights.dtype,
                                                       scale_fmt=self.scale_fmt)
        x, recv_topk_ids, recv_topk_weights, recv_tokens_per_expert = self.token_dispatcher.dispatch(
            (hs_quant, hs_scale),
            topk_ids,
            topk_weights,
            expert_list,
        )
        out_states = fused_moe_v3_fp8(x, recv_topk_ids, recv_topk_weights, (up_weights, up_scale),
                                      (down_weights, down_scale), recv_tokens_per_expert)
        return self.token_dispatcher.combine(out_states)

    def capture(self):
        return self.token_dispatcher.buffer_normal.capture()

    def wait(self, event):
        self.token_dispatcher.release()
        event.current_stream_wait()

    def dispatch_async(self,
                       x: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
                       topk_idx: torch.Tensor,
                       topk_weights: torch.Tensor,
                       num_experts: int | None = None,
                       previous_event=None,
                       async_finish=True):
        if isinstance(x, torch.Tensor):
            x = self.per_token_group_quant_fp8(x)
        return self.token_dispatcher.dispatch_normal_async(x, topk_idx, topk_weights, num_experts, previous_event,
                                                           async_finish)

    def combine_async(self, x: torch.Tensor, handle: tuple, previous_event=None, async_finish=True):
        return self.token_dispatcher.combine_normal_async(x, handle, previous_event, async_finish)

    def release(self):
        return self.token_dispatcher.release()

    def fusedmoe_forward(self, state, up_weight, up_scale, down_weight, down_scale):
        return fused_moe_v3_fp8(state['recv_hidden_states'], state['recv_topk_idx'], state['recv_topk_weights'],
                                (up_weight, up_scale), (down_weight, down_scale), state['recv_tokens_per_expert'])

    def per_token_group_quant_fp8(self,
                                  x: torch.Tensor,
                                  dtype: torch.dtype | None = None,
                                  scale_fmt: str | None = None):
        dtype = dtype if dtype is not None else self.fp8_dtype
        scale_fmt = scale_fmt if scale_fmt is not None else self.scale_fmt
        return per_token_group_quant_fp8(x, self.block_size, dtype=dtype, scale_fmt=scale_fmt)


class FusedMoELowLatency:

    def __init__(
        self,
        ep_size: int,
        ep_group: dist.ProcessGroup,
        num_experts: int,
        hidden_dim: int,
        layer_index: int,
        block_size: int = 128,
        out_dtype: torch.dtype = torch.bfloat16,
        num_max_dispatch_tokens_per_rank: int = 128,
    ):
        self.num_experts = num_experts
        self.layer_index = layer_index
        self.block_size = block_size
        self.out_dtype = out_dtype
        self.token_dispatcher = DeepEPTokenDispatcherLowLatency(
            group=ep_group,
            num_experts=num_experts,
            num_local_experts=num_experts // ep_size,
            hidden_size=hidden_dim,
            params_dtype=out_dtype,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
        )

    def _deepgemm_grouped_fp8_nt_masked(self, input_tuple, w_tuple, out: torch.Tensor, masked_m: torch.Tensor,
                                        expected_m: int):
        from lmdeploy.pytorch.third_party.deep_gemm import m_grouped_fp8_gemm_nt_masked
        return m_grouped_fp8_gemm_nt_masked(input_tuple, w_tuple, out, masked_m, expected_m)

    def experts(
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
        num_groups, m, _ = hidden_states_fp8[0].shape
        n = gate_up_weight.size(1)
        expected_m = min(expected_m, m)
        gateup_output = torch.empty((num_groups, m, n), device=hidden_states_fp8[0].device, dtype=self.out_dtype)
        self._deepgemm_grouped_fp8_nt_masked([DisposibleTensor.maybe_unwrap(x) for x in hidden_states_fp8],
                                            gate_up_weight_fp8, gateup_output, masked_m, expected_m)
        DisposibleTensor.maybe_dispose(hidden_states_fp8[0])
        DisposibleTensor.maybe_dispose(hidden_states_fp8[1])
        down_input = torch.empty((gateup_output.shape[0], gateup_output.shape[1], gateup_output.shape[2] // 2),
                                 device=gateup_output.device,
                                 dtype=gate_down_weight.dtype)
        down_input_scale = torch.empty(
            (gateup_output.shape[0], gateup_output.shape[1], gateup_output.shape[2] // 2 // self.block_size),
            device=gateup_output.device,
            dtype=torch.float32)
        silu_and_mul_masked_post_quant_fwd(gateup_output, down_input, down_input_scale, self.block_size, masked_m)
        del gateup_output
        down_output = torch.empty((num_groups, m, gate_down_weight.size(1)),
                                  device=down_input.device,
                                  dtype=self.out_dtype)
        self._deepgemm_grouped_fp8_nt_masked((down_input, down_input_scale), gate_down_weight_fp8, down_output,
                                            masked_m, expected_m)
        return down_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.LongTensor,
        up_weights: torch.Tensor,
        up_scale: torch.Tensor,
        down_weights: torch.Tensor,
        down_scale: torch.Tensor,
        expert_list: list[int] = None,
    ):
        recv_hidden_states, topk_idx, topk_weights, masked_m, expected_m = self.token_dispatcher.dispatch(
            hidden_states, topk_ids, topk_weights, self.num_experts)
        out_states = self.experts(recv_hidden_states, up_weights, up_scale, down_weights, down_scale, masked_m,
                                  expected_m)
        return self.token_dispatcher.combine(out_states, topk_idx, topk_weights)

    def wait(self, event):
        event.current_stream_wait()

    def dispatch_async(self,
                       hidden_states: torch.Tensor,
                       topk_idx: torch.Tensor,
                       num_experts: int | None = None,
                       use_fp8: bool = True,
                       async_finish: bool = True):
        return self.token_dispatcher.dispatch_async(hidden_states, topk_idx, num_experts, use_fp8, async_finish)

    def combine_async(self,
                      hidden_states: torch.Tensor,
                      topk_idx: torch.Tensor,
                      topk_weights: torch.Tensor,
                      handle: tuple,
                      async_finish: bool):
        return self.token_dispatcher.combine_async(hidden_states, topk_idx, topk_weights, handle, async_finish)

    def fusedmoe_forward(self, state, up_weight, up_scale, down_weight, down_scale):
        recv_hidden_states = state['recv_hidden_states']
        masked_m = state['recv_expert_count']
        hidden_shape = state['raw_hidden_shape']
        topk_idx = state['topk_idx']
        expected_m = (hidden_shape[0] * self.token_dispatcher.buffer_low_latency.group_size * topk_idx.shape[1] +
                      self.token_dispatcher.num_experts) // self.token_dispatcher.num_experts
        return self.experts(recv_hidden_states, up_weight, up_scale, down_weight, down_scale, masked_m, expected_m)


def build_deepep_moe(
    low_latency_mode: bool,
    ep_size: int,
    ep_group: dist.ProcessGroup,
    num_experts: int,
    hidden_dim: int,
    block_size: int,
    top_k: int,
    out_dtype: torch.dtype,
    fp8_dtype: torch.dtype | None = None,
    scale_fmt: str | None = None,
    layer_idx: int = 0,
    num_max_dispatch_tokens_per_rank: int = 128,
    chunk_size: int | None = 32 * 1024,
    expert_alignment: int = 128,
):
    if low_latency_mode:
        return FusedMoELowLatency(ep_size=ep_size,
                                  ep_group=ep_group,
                                  num_experts=num_experts,
                                  hidden_dim=hidden_dim,
                                  layer_index=layer_idx,
                                  block_size=block_size,
                                  out_dtype=out_dtype,
                                  num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank)
    return FusedMoENormal(ep_size=ep_size,
                          ep_group=ep_group,
                          num_experts=num_experts,
                          hidden_dim=hidden_dim,
                          layer_index=layer_idx,
                          block_size=block_size,
                          top_k=top_k,
                          out_dtype=out_dtype,
                          fp8_dtype=fp8_dtype,
                          scale_fmt=scale_fmt,
                          num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
                          chunk_size=chunk_size,
                          expert_alignment=expert_alignment)


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
                expert_list: list[int] = None,
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
                 fp8_dtype: torch.dtype = torch.float8_e4m3fn,
                 num_max_dispatch_tokens_per_rank: int = 128,
                 layer_idx: int = 0):
        super().__init__(top_k, num_experts, renormalize, block_size, out_dtype)
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.ep_group = ep_group
        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.out_dtype = out_dtype
        self.fp8_dtype = fp8_dtype
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        self.layer_idx = layer_idx
        try:
            import deep_gemm  # noqa: F401
            self.use_deep_gemm = True
        except ImportError:
            logger.exception('DeepGEMM is required for DeepEP MoE implementation.')
            raise

        if not use_deepep:
            raise ImportError('DeepEP is required for DeepEP MoE implementation. Please install '
                              'https://github.com/deepseek-ai/DeepEP.')
        get_deepep_state().enable()
        if hasattr(DeepEPBuffer, 'set_explicitly_destroy'):
            DeepEPBuffer.set_explicitly_destroy()

        # pre-allocate buffer
        self.fusedmoe_build(True)

    def ep_expert_list(self, world_size: int, rank: int):
        """Experts list of current rank."""
        if get_dist_manager().current_context().dist_config.enable_eplb:
            from lmdeploy.pytorch.nn.eplb import get_eplb_phy2log_metadata_by_layer
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
                expert_list: list[int] = None,
                act_func: Callable = None,
                **kwargs):
        """forward."""
        hidden_states, topk_weights, topk_ids, split_size = split_inputs_by_attn_tp(hidden_states, topk_weights,
                                                                                    topk_ids)

        topk_weights = self.do_renormalize(topk_weights)
        step_ctx = get_step_ctx_manager().current_context()
        low_latency_mode = step_ctx.global_is_decoding() and self.use_deep_gemm
        moe = self.fusedmoe_build(low_latency_mode)
        out_states = moe.forward(hidden_states, topk_weights, topk_ids, gate_up_weights, gate_up_scale, down_weights,
                                 down_scale, expert_list)

        out_states = gather_outputs_by_attn_tp(out_states, split_size)
        return out_states

    def do_renormalize(self, topk_weights):
        return _renormalize(topk_weights, self.renormalize)

    def fusedmoe_build(self, low_latency_mode: bool = False):
        deepep_moe = build_deepep_moe(low_latency_mode,
                                      self.ep_size,
                                      self.ep_group,
                                      self.num_experts,
                                      self.hidden_dim,
                                      self.block_size,
                                      self.top_k,
                                      self.out_dtype,
                                      fp8_dtype=self.fp8_dtype,
                                      scale_fmt=self.scale_fmt,
                                      layer_idx=self.layer_idx,
                                      num_max_dispatch_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
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
              fp8_dtype: torch.dtype = torch.float8_e4m3fn,
              num_max_dispatch_tokens_per_rank: int = 128,
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
                                               fp8_dtype=fp8_dtype,
                                               num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
                                               layer_idx=layer_idx)
        else:
            return TritonFusedMoEBlockedF8Impl(top_k=top_k,
                                               num_experts=num_experts,
                                               renormalize=renormalize,
                                               block_size=block_size,
                                               out_dtype=out_dtype)

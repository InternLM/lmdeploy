# Copyright (c) OpenMMLab. All rights reserved.

from typing import Callable, List, Optional

import torch

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.backends.deepep_moe_checker import get_moe_backend
from lmdeploy.pytorch.backends.moe import FusedMoEBuilder, FusedMoEImpl
from lmdeploy.pytorch.distributed import get_dist_manager
from lmdeploy.pytorch.kernels.cuda import fused_moe
from lmdeploy.pytorch.kernels.cuda.fused_moe import _renormalize
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager
from lmdeploy.utils import get_logger

from .ep_utils import gather_outputs_by_attn_tp, split_inputs_by_attn_tp

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
                gate_up_bias: torch.Tensor = None,
                down_bias: torch.Tensor = None,
                expert_list: List[int] = None,
                act_func: Callable = None):
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
                         w1_bias=gate_up_bias,
                         w2_bias=down_bias,
                         expert_offset=expert_offset,
                         num_experts=num_experts,
                         renormalize=self.renormalize,
                         act_func=act_func)


# modify from dlblas: https://github.com/DeepLink-org/DLBlas
class FusedMoENormal:

    def __init__(
        self,
        ep_size: int,
        ep_group: dist.ProcessGroup,
        num_experts: int,
        hidden_dim: int,
        layer_index: int = 0,
        top_k: int = 8,
        out_dtype: torch.dtype = torch.bfloat16,
    ):
        from dlblas.layers.moe.token_dispatcher import DeepEPTokenDispatcherNormal
        self.layer_index = layer_index
        self.top_k = top_k
        self.num_experts = num_experts
        self.num_local_experts = num_experts // ep_size
        self.out_dtype = out_dtype
        self.token_dispatcher = DeepEPTokenDispatcherNormal(
            group=ep_group,
            num_experts=num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=hidden_dim,
            params_dtype=out_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.LongTensor,
        up_weights: torch.Tensor,
        down_weights: torch.Tensor,
        expert_list: List[int] = None,
    ):
        """forward."""
        from lmdeploy.pytorch.kernels.cuda.fused_moe_ep import fused_moe_v3
        x, recv_topk_ids, recv_topk_weights, recv_tokens_per_expert = self.token_dispatcher.dispatch(
            hidden_states,
            topk_ids,
            topk_weights,
            expert_list,
        )
        topk_ids, topk_weights = None, None
        out_states = fused_moe_v3(x, recv_topk_ids, recv_topk_weights, up_weights, down_weights, recv_tokens_per_expert)
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

    def fusedmoe_forward(self, state, up_weight, down_weight):
        from lmdeploy.pytorch.kernels.cuda.fused_moe_ep import fused_moe_v3
        return fused_moe_v3(state['recv_hidden_states'], state['recv_topk_idx'], state['recv_topk_weights'], up_weight,
                            down_weight, state['recv_tokens_per_expert'])


def _disposible_tensor(tensor):
    from dlblas.utils.utils import DisposibleTensor
    if isinstance(tensor, torch.Tensor):
        tensor = DisposibleTensor(tensor)
    else:
        tensor = [DisposibleTensor(x) for x in tensor]
    return tensor


def dispatch_ll(
    self,
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    use_fp8: bool = True,
):
    """Dispatch low latency."""
    if num_experts is not None and self.num_experts is not None:
        assert self.num_experts == num_experts
    topk_idx = topk_idx.to(torch.int64)
    expected_m = (hidden_states.shape[0] * self.get_buffer().group_size * topk_idx.shape[1] +
                  num_experts) // num_experts

    (
        packed_recv_hidden,
        masked_m,
        self.handle,
        event,
        hook,
    ) = self.get_buffer().low_latency_dispatch(
        hidden_states,
        topk_idx,
        self.num_max_dispatch_tokens_per_rank,
        num_experts,
        use_fp8=use_fp8,
        async_finish=not self.return_recv_hook,
        return_recv_hook=self.return_recv_hook,
    )
    hook() if self.return_recv_hook else event.current_stream_wait()
    packed_recv_hidden = _disposible_tensor(packed_recv_hidden)
    return (
        packed_recv_hidden,
        topk_idx,
        topk_weights,
        masked_m,
        expected_m,
    )


def dispatch_async_ll(
    self,
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    num_experts: Optional[int] = None,
    use_fp8: bool = True,
    async_finish: bool = True,
):
    assert topk_idx.dtype == torch.int64
    if num_experts is not None and self.num_experts is not None:
        assert self.num_experts == num_experts
    (
        recv_hidden_states,
        recv_expert_count,
        handle,
        event,
        hook,
    ) = self.get_buffer().low_latency_dispatch(
        hidden_states,
        topk_idx,
        self.num_max_dispatch_tokens_per_rank,
        num_experts=self.num_experts,
        use_fp8=use_fp8,
        async_finish=async_finish,
        return_recv_hook=not async_finish,
    )
    recv_hidden_states = _disposible_tensor(recv_hidden_states)
    return recv_hidden_states, recv_expert_count, handle, event, hook


class FusedMoELowLatency:

    def __init__(
        self,
        ep_size: int,
        ep_group: dist.ProcessGroup,
        num_experts: int,
        hidden_dim: int,
        layer_index: int,
        out_dtype: torch.dtype = torch.bfloat16,
    ):
        from dlblas.layers.moe.token_dispatcher import DeepEPTokenDispatcherLowLatency
        self.num_experts = num_experts
        self.layer_index = layer_index
        self.out_dtype = out_dtype
        self.token_dispatcher = DeepEPTokenDispatcherLowLatency(
            group=ep_group,
            num_experts=num_experts,
            num_local_experts=num_experts // ep_size,
            hidden_size=hidden_dim,
            params_dtype=out_dtype,
        )

    def experts(
        self,
        hidden_states: torch.Tensor,
        gate_up_weight: torch.Tensor,
        gate_down_weight: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
    ):
        from dlblas.utils.utils import DisposibleTensor

        from lmdeploy.pytorch.kernels.cuda.activation import silu_and_mul_moe_ep
        from lmdeploy.pytorch.third_party.deep_gemm import m_grouped_bf16_gemm_nt_masked
        num_groups, m, _ = hidden_states.shape
        n = gate_up_weight.size(1)
        expected_m = min(expected_m, m)
        gateup_output = gate_up_weight.new_empty((num_groups, m, n))
        m_grouped_bf16_gemm_nt_masked(DisposibleTensor.maybe_unwrap(hidden_states), gate_up_weight, gateup_output,
                                      masked_m, expected_m)
        DisposibleTensor.maybe_dispose(hidden_states)
        down_input = silu_and_mul_moe_ep(gateup_output, masked_m)
        del gateup_output
        n = gate_down_weight.size(1)
        down_output = down_input.new_empty((num_groups, m, n))
        m_grouped_bf16_gemm_nt_masked(down_input, gate_down_weight, down_output, masked_m, expected_m)
        return down_output

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                up_weights: torch.Tensor,
                down_weights: torch.Tensor,
                expert_list: List[int] = None):
        """forward."""
        recv_hidden_states, topk_idx, topk_weights, masked_m, expected_m = dispatch_ll(
            self.token_dispatcher,
            hidden_states,
            topk_ids,
            topk_weights,
            self.num_experts,
            use_fp8=False,
        )
        hidden_states = None
        out_states = self.experts(recv_hidden_states, up_weights, down_weights, masked_m, expected_m)
        out_states = self.token_dispatcher.combine(out_states, topk_idx, topk_weights)
        return out_states

    def wait(self, event):
        event.current_stream_wait()

    def dispatch_async(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        num_experts: Optional[int] = None,
        use_fp8: bool = False,
        async_finish: bool = True,
    ):
        return dispatch_async_ll(self.token_dispatcher, hidden_states, topk_idx, num_experts, use_fp8, async_finish)

    def combine_async(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: tuple,
        async_finish: bool,
    ):
        return self.token_dispatcher.combine_async(hidden_states, topk_idx, topk_weights, handle, async_finish)

    def fusedmoe_forward(self, state, up_weight, down_weight):
        recv_hidden_states = state['recv_hidden_states']
        masked_m = state['recv_expert_count']
        hidden_shape = state['raw_hidden_shape']
        topk_idx = state['topk_idx']
        expected_m = (hidden_shape[0] * self.token_dispatcher.buffer_low_latency.group_size * topk_idx.shape[1] +
                      self.token_dispatcher.num_experts) // self.token_dispatcher.num_experts
        return self.experts(recv_hidden_states, up_weight, down_weight, masked_m, expected_m)


def build_deepep_moe(
    low_latency_mode: bool,
    ep_size: int,
    ep_group: dist.ProcessGroup,
    num_experts: int,
    hidden_dim: int,
    top_k: int,
    layer_idx: int = 0,
    out_dtype: torch.dtype = torch.bfloat16,
):
    if low_latency_mode:
        return FusedMoELowLatency(ep_size=ep_size,
                                  ep_group=ep_group,
                                  num_experts=num_experts,
                                  hidden_dim=hidden_dim,
                                  layer_index=layer_idx,
                                  out_dtype=out_dtype)
    else:
        return FusedMoENormal(ep_size=ep_size,
                              ep_group=ep_group,
                              num_experts=num_experts,
                              hidden_dim=hidden_dim,
                              layer_index=layer_idx,
                              top_k=top_k,
                              out_dtype=out_dtype)


class FusedMoEEPImpl(TritonFusedMoEImpl):
    """Fused moe implementation."""

    def __init__(
        self,
        ep_size: int,
        ep_group: dist.ProcessGroup,
        top_k: int,
        num_experts: int,
        hidden_dim: int,
        renormalize: bool = False,
        layer_idx: int = 0,
        out_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(top_k, num_experts, renormalize)
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.ep_group = ep_group
        self.hidden_dim = hidden_dim
        self.layer_idx = layer_idx
        self.out_dtype = out_dtype

        try:
            import deep_gemm  # noqa: F401
        except ImportError:
            logger.exception('DeepGEMM is required for DeepEP MoE implementation.')

        try:
            from dlblas.layers.moe.token_dispatcher import DeepEPBuffer, DeepEPMode, use_deepep  # noqa: F401
            get_moe_backend().set_deepep_moe_backend()
            if hasattr(DeepEPBuffer, 'set_explicitly_destroy'):
                DeepEPBuffer.set_explicitly_destroy()
        except ImportError:
            logger.warning('For higher performance, please install DeepEP https://github.com/deepseek-ai/DeepEP')

        # pre-allocate buffer
        self.fusedmoe_build(True)

    def update_weights(self, gate_up_weights: torch.Tensor, down_weights: torch.Tensor):
        return gate_up_weights, down_weights

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                down_weights: torch.Tensor,
                gate_up_bias: torch.Tensor = None,
                down_bias: torch.Tensor = None,
                expert_list: List[int] = None,
                act_func: Callable = None):
        """forward."""
        assert act_func is None, 'Activation function is not supported in DeepEP MoE.'
        hidden_states, topk_weights, topk_ids, split_size = split_inputs_by_attn_tp(hidden_states, topk_weights,
                                                                                    topk_ids)

        topk_weights = self.do_renormalize(topk_weights)
        step_ctx = get_step_ctx_manager().current_context()
        low_latency_mode = step_ctx.is_decoding
        moe = self.fusedmoe_build(low_latency_mode)
        out_states = moe.forward(hidden_states, topk_weights, topk_ids, gate_up_weights, down_weights, expert_list)

        out_states = gather_outputs_by_attn_tp(out_states, split_size)
        return out_states

    def ep_expert_list(self, world_size: int, rank: int):
        """Experts list of current rank."""
        if get_dist_manager().current_context().dist_config.enable_eplb:
            raise NotImplementedError('float16/bfloat16 enable_eplb is not Implemented.')
        else:
            return super().ep_expert_list(world_size=world_size, rank=rank)

    def do_renormalize(self, topk_weights):
        return _renormalize(topk_weights, self.renormalize)

    def fusedmoe_build(self, low_latency_mode: bool = False):
        deepep_moe = build_deepep_moe(low_latency_mode,
                                      self.ep_size,
                                      self.ep_group,
                                      self.num_experts,
                                      self.hidden_dim,
                                      self.top_k,
                                      layer_idx=self.layer_idx,
                                      out_dtype=self.out_dtype)
        return deepep_moe


class TritonFusedMoEBuilder(FusedMoEBuilder):
    """Triton fused moe builder."""

    @staticmethod
    def build(
        top_k: int,
        num_experts: int,
        renormalize: bool = False,
        hidden_dim: int = 1,
        ep_size: int = 1,
        ep_group: dist.ProcessGroup = None,
        layer_idx: int = 0,
        out_dtype: torch.dtype = torch.bfloat16,
    ):
        """Build from mlp."""
        if ep_size > 1:
            return FusedMoEEPImpl(ep_size=ep_size,
                                  ep_group=ep_group,
                                  top_k=top_k,
                                  num_experts=num_experts,
                                  hidden_dim=hidden_dim,
                                  renormalize=renormalize,
                                  layer_idx=layer_idx,
                                  out_dtype=out_dtype)
        return TritonFusedMoEImpl(top_k=top_k, num_experts=num_experts, renormalize=renormalize)

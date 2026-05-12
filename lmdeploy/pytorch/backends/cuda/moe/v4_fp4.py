# Copyright (c) OpenMMLab. All rights reserved.


import torch
import torch.distributed as dist

from lmdeploy.pytorch.backends.deepep_state import get_deepep_state
from lmdeploy.pytorch.backends.moe import FusedMoEBuilder
from lmdeploy.pytorch.distributed import get_dist_manager
from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import quant_fp8
from lmdeploy.pytorch.kernels.cuda.v4_fp4_fused_moe import fused_moe_v4_fp4
from lmdeploy.pytorch.kernels.cuda.v4_fp4_grouped_gemm import (
    fused_moe_v4_fp4_ep_low_latency,
    fused_moe_v4_fp4_ep_normal,
)
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def _v4_swiglu(intermediate: torch.Tensor, swiglu_limit: float) -> torch.Tensor:
    """Match DeepSeek-V4 expert activation in the Triton FP4 fused MoE path."""
    hidden = intermediate.size(-1) // 2
    gate = intermediate[..., :hidden].float()
    up = intermediate[..., hidden:].float()
    if swiglu_limit > 0:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
    return (torch.nn.functional.silu(gate) * up).to(intermediate.dtype)


def _slice_local_topk(topk_ids: torch.LongTensor,
                      topk_weights: torch.Tensor,
                      expert_offset: int,
                      num_experts: int,
                      invalid_expert: int = -1) -> tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
    """Map global routed experts into the current TP rank's local expert id
    space.

    `invalid_expert` is backend-specific. Triton FP4 uses 0 with a zero route
    weight; the legacy DeepGEMM path uses -1 to match TileKernels routing.
    """
    local_topk_ids = topk_ids.to(torch.int64) - expert_offset
    local_mask = (local_topk_ids >= 0) & (local_topk_ids < num_experts)
    local_topk_ids = torch.where(local_mask, local_topk_ids, local_topk_ids.new_full((), invalid_expert))
    local_topk_weights = torch.where(local_mask, topk_weights, topk_weights.new_zeros(()))
    return local_topk_ids, local_topk_weights, local_mask


class TritonFusedMoEV4FP4TPImpl:
    """TP-only Triton FP8xFP4 fused MoE for DeepSeek-V4.

    This path keeps checkpoint-native packed FP4 expert weights resident in memory. The Triton GEMM unpacks FP4 weights
    in-kernel, casts the E2M1 values to FP8, then uses the same two-stage fused-MoE flow as the generic fused MoE path.
    """

    def __init__(self,
                 top_k: int,
                 num_experts: int,
                 hidden_dim: int,
                 ffn_dim: int,
                 expert_offset: int,
                 swiglu_limit: float = 0.0,
                 scale_fmt: str | None = 'ue8m0'):
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.expert_offset = expert_offset
        self.swiglu_limit = swiglu_limit
        self.scale_fmt = scale_fmt

    def _act_func(self, intermediate: torch.Tensor) -> torch.Tensor:
        return _v4_swiglu(intermediate, self.swiglu_limit)

    def update_weights(self,
                       gate_up_weight: torch.Tensor,
                       gate_up_scale: torch.Tensor,
                       down_weight: torch.Tensor,
                       down_scale: torch.Tensor):
        return gate_up_weight, gate_up_scale, down_weight, down_scale

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weight: torch.Tensor,
                gate_up_scale: torch.Tensor,
                down_weight: torch.Tensor,
                down_scale: torch.Tensor):
        num_tokens = hidden_states.size(0)
        if num_tokens == 0:
            return hidden_states.new_empty((0, self.hidden_dim))

        input_quant, input_scale = quant_fp8(hidden_states,
                                             128,
                                             dtype=torch.float8_e4m3fn,
                                             scale_fmt=self.scale_fmt)
        if self.swiglu_limit > 0:
            act_func = self._act_func
        else:
            act_func = None
        out = fused_moe_v4_fp4(input_quant,
                               input_scale,
                               gate_up_weight,
                               gate_up_scale,
                               down_weight,
                               down_scale,
                               topk_weights=topk_weights,
                               topk_ids=topk_ids,
                               topk=self.top_k,
                               out_dtype=hidden_states.dtype,
                               renormalize=False,
                               act_func=act_func)
        return out


class DeepGemmFusedMoEV4TPImpl:
    """TP-only fused MoE implementation for DeepSeek-V4 routed experts.

    This path keeps the official V4 routing semantics: experts are sharded by
    expert id across TP ranks, each rank computes its local expert
    contributions, then the token outputs are all-reduced across the TP group.

    Note:
        This implementation is intentionally kept for future SM100 support.
        DeepGEMM's grouped FP8xFP4 kernels accept packed FP4 expert weights
        only on arch_major == 10, so Hopper/SM90 should not select this path.
    """

    def __init__(self,
                 top_k: int,
                 num_experts: int,
                 hidden_dim: int,
                 ffn_dim: int,
                 expert_offset: int,
                 swiglu_limit: float = 0.0):
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.expert_offset = expert_offset
        self.swiglu_limit = swiglu_limit

    @staticmethod
    def _import_impl():
        try:
            import deep_gemm
            from deep_gemm.utils import per_token_cast_to_fp8
        except ImportError as e:
            raise ImportError('DeepSeek-V4 fused MoE requires deep_gemm.') from e

        try:
            from tile_kernels.moe import expand_to_fused_with_sf, get_fused_mapping, reduce_fused
            from tile_kernels.torch import swiglu_forward
        except ImportError as e:
            raise ImportError('DeepSeek-V4 fused MoE requires tile_kernels.') from e

        return deep_gemm, per_token_cast_to_fp8, get_fused_mapping, expand_to_fused_with_sf, reduce_fused, swiglu_forward   # noqa: E501

    def update_weights(self,
                       gate_up_weight: torch.Tensor,
                       gate_up_scale: torch.Tensor,
                       down_weight: torch.Tensor,
                       down_scale: torch.Tensor):
        # `m_grouped_fp8_fp4_gemm_nt_contiguous` uses the plain grouped scale
        # layout. Unlike `mega_moe`, it does not require the pre-transformed
        # layout from `transform_sf_into_required_layout`.
        gate_up_scale = gate_up_scale.float()
        down_scale = down_scale.float()
        return gate_up_weight, gate_up_scale, down_weight, down_scale

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weight: torch.Tensor,
                gate_up_scale: torch.Tensor,
                down_weight: torch.Tensor,
                down_scale: torch.Tensor):
        deep_gemm, per_token_cast_to_fp8, get_fused_mapping, expand_to_fused_with_sf, reduce_fused, swiglu_forward = \
            self._import_impl()

        num_tokens = hidden_states.size(0)
        if num_tokens == 0:
            return hidden_states.new_empty((0, self.hidden_dim))

        local_topk_ids, _, _ = _slice_local_topk(topk_ids,
                                                 topk_weights,
                                                 self.expert_offset,
                                                 self.num_experts,
                                                 invalid_expert=-1)

        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        num_expanded_tokens = ((num_tokens * self.top_k + alignment - 1) // alignment) * alignment
        (pos_to_expert, _, pos_to_token_topk, token_topk_to_pos, _, expert_end, _, _) = get_fused_mapping(
            local_topk_ids.contiguous(),
            self.num_experts,
            num_expanded_tokens,
            alignment,
            force_no_sync=True,
        )

        major = torch.cuda.get_device_capability(hidden_states.device)[0]
        use_ue8m0 = major >= 10
        hidden_fp8 = per_token_cast_to_fp8(hidden_states,
                                           use_ue8m0=use_ue8m0,
                                           gran_k=32,
                                           use_packed_ue8m0=use_ue8m0)
        fused_hidden = expand_to_fused_with_sf(hidden_fp8,
                                               32,
                                               token_topk_to_pos,
                                               pos_to_expert,
                                               use_tma_aligned_col_major_sf=use_ue8m0)

        gate_up_out = hidden_states.new_empty((num_expanded_tokens, 2 * self.ffn_dim))
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            fused_hidden,
            (gate_up_weight, gate_up_scale),
            gate_up_out,
            expert_end,
            recipe=(1, 1, 32),
            disable_ue8m0_cast=not use_ue8m0,
        )

        act_out = swiglu_forward(gate_up_out,
                                 pos_to_token_topk=pos_to_token_topk,
                                 topk_weights=topk_weights,
                                 swiglu_clamp_value=(self.swiglu_limit if self.swiglu_limit > 0 else None))
        act_out = act_out.to(torch.bfloat16)
        act_fp8 = per_token_cast_to_fp8(act_out,
                                        use_ue8m0=use_ue8m0,
                                        gran_k=32,
                                        use_packed_ue8m0=use_ue8m0)

        down_out = hidden_states.new_empty((num_expanded_tokens, self.hidden_dim))
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            act_fp8,
            (down_weight, down_scale),
            down_out,
            expert_end,
            recipe=(1, 1, 32),
            disable_ue8m0_cast=not use_ue8m0,
        )

        out = reduce_fused(down_out, None, token_topk_to_pos)
        return out


class V4FP4FusedMoENormal:
    """Prefill EP MoE: dispatch -> scatter -> FP4 GEMM -> gather -> combine."""

    def __init__(self, ep_size, ep_group, num_experts, num_local_experts,
                 hidden_dim, ffn_dim, top_k, swiglu_limit, scale_fmt, layer_idx, out_dtype):
        from lmdeploy.pytorch.backends.cuda.token_dispatcher import DeepEPTokenDispatcher
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.top_k = top_k
        self.swiglu_limit = swiglu_limit
        self.scale_fmt = scale_fmt
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.out_dtype = out_dtype
        self.token_dispatcher = DeepEPTokenDispatcher(
            group=ep_group,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_dim,
            params_dtype=out_dtype,
        )

    def forward(self, hidden_states, topk_weights, topk_ids,
                gate_up_weight, gate_up_scale, down_weight, down_scale,
                expert_list=None):
        topk_weights = topk_weights.to(torch.float32)
        recv_x, recv_topk_idx, recv_topk_weights, recv_tokens_per_expert = \
            self.token_dispatcher.dispatch(hidden_states, topk_ids, topk_weights, expert_list)

        expert_offset = expert_list[0] if expert_list else 0
        out = fused_moe_v4_fp4_ep_normal(
            recv_x, recv_topk_idx, recv_topk_weights, recv_tokens_per_expert,
            gate_up_weight, gate_up_scale, down_weight, down_scale,
            num_local_experts=self.num_local_experts,
            expert_offset=expert_offset,
            swiglu_limit=self.swiglu_limit,
            group_size=128,
            out_dtype=self.out_dtype,
        )

        out = self.token_dispatcher.combine(out)
        return out


class V4FP4FusedMoELowLatency:
    """Decode EP MoE: low_latency_dispatch -> masked FP4 GEMM -> low_latency_combine."""

    def __init__(self, ep_size, ep_group, num_experts, num_local_experts,
                 hidden_dim, ffn_dim, top_k, swiglu_limit, scale_fmt, layer_idx, out_dtype):
        from lmdeploy.pytorch.backends.cuda.token_dispatcher import DeepEPTokenDispatcherLowLatency
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.top_k = top_k
        self.swiglu_limit = swiglu_limit
        self.scale_fmt = scale_fmt
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.out_dtype = out_dtype
        self.token_dispatcher = DeepEPTokenDispatcherLowLatency(
            group=ep_group,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_dim,
            params_dtype=out_dtype,
        )

    def forward(self, hidden_states, topk_weights, topk_ids,
                gate_up_weight, gate_up_scale, down_weight, down_scale,
                expert_list=None):
        topk_weights = topk_weights.to(torch.float32)
        recv_hidden_states, topk_idx, topk_weights, masked_m, expected_m = \
            self.token_dispatcher.dispatch(hidden_states, topk_ids, topk_weights, self.num_experts)

        out = fused_moe_v4_fp4_ep_low_latency(
            recv_hidden_states, masked_m, expected_m,
            gate_up_weight, gate_up_scale, down_weight, down_scale,
            swiglu_limit=self.swiglu_limit,
            group_size=128,
            out_dtype=self.out_dtype,
        )

        out = self.token_dispatcher.combine(out, topk_idx, topk_weights)
        return out


class TritonFusedMoEV4FP4EPImpl:
    """V4 FP4 MoE with Expert Parallelism.

    Follows the same pattern as FusedDeepEpMoEBlockedF8Impl: dispatch/combine
    via DeepEP, GEMM via V4 FP4 Triton kernels.
    """

    def __init__(self, ep_size, ep_group, top_k, num_experts, num_local_experts,
                 hidden_dim, ffn_dim, swiglu_limit=0.0, scale_fmt='ue8m0',
                 layer_idx=0):
        self.ep_size = ep_size
        self.ep_group = ep_group
        self.top_k = top_k
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.swiglu_limit = swiglu_limit
        self.scale_fmt = scale_fmt
        self.layer_idx = layer_idx

        try:
            from deep_ep import Buffer  # noqa: F401
            get_deepep_state().enable()
        except ImportError:
            logger.warning('DeepEP is not installed. V4 FP4 EP MoE requires DeepEP '
                           'from https://github.com/deepseek-ai/DeepEP')

        # Pre-allocate buffer.
        self.fusedmoe_build(True)

    def update_weights(self, gate_up_weight, gate_up_scale, down_weight, down_scale):
        return gate_up_weight, gate_up_scale, down_weight, down_scale

    def ep_expert_list(self, world_size, rank):
        if get_dist_manager().current_context().dist_config.enable_eplb:
            from lmdeploy.pytorch.nn.eplb import EPLBManager
            return EPLBManager.get_dispatch_info(rank, self.layer_idx)
        expert_per_rank = (self.num_experts + world_size - 1) // world_size
        first_expert = rank * expert_per_rank
        last_expert = min(first_expert + expert_per_rank, self.num_experts)
        return list(range(first_expert, last_expert))

    def fusedmoe_build(self, low_latency_mode=False):
        if low_latency_mode:
            return V4FP4FusedMoELowLatency(
                ep_size=self.ep_size, ep_group=self.ep_group,
                num_experts=self.num_experts, num_local_experts=self.num_local_experts,
                hidden_dim=self.hidden_dim, ffn_dim=self.ffn_dim,
                top_k=self.top_k, swiglu_limit=self.swiglu_limit,
                scale_fmt=self.scale_fmt, layer_idx=self.layer_idx,
                out_dtype=torch.bfloat16)
        return V4FP4FusedMoENormal(
            ep_size=self.ep_size, ep_group=self.ep_group,
            num_experts=self.num_experts, num_local_experts=self.num_local_experts,
            hidden_dim=self.hidden_dim, ffn_dim=self.ffn_dim,
            top_k=self.top_k, swiglu_limit=self.swiglu_limit,
            scale_fmt=self.scale_fmt, layer_idx=self.layer_idx,
            out_dtype=torch.bfloat16)

    def forward(self, hidden_states, topk_weights, topk_ids,
                gate_up_weight, gate_up_scale, down_weight, down_scale,
                expert_list=None):
        from lmdeploy.pytorch.backends.cuda.moe.ep_utils import gather_outputs_by_attn_tp, split_inputs_by_attn_tp
        from lmdeploy.pytorch.model_inputs import get_step_ctx_manager

        hidden_states, topk_weights, topk_ids, split_size = \
            split_inputs_by_attn_tp(hidden_states, topk_weights, topk_ids)

        step_ctx = get_step_ctx_manager().current_context()
        low_latency_mode = step_ctx.is_decoding
        moe = self.fusedmoe_build(low_latency_mode)
        out_states = moe.forward(hidden_states, topk_weights, topk_ids,
                                  gate_up_weight, gate_up_scale,
                                  down_weight, down_scale,
                                  expert_list=expert_list)

        out_states = gather_outputs_by_attn_tp(out_states, split_size)
        return out_states



class DeepGemmFusedMoEV4Builder(FusedMoEBuilder):
    """Builder for DeepSeek-V4 fused MoE (DeepGEMM path)."""

    @staticmethod
    def build(top_k: int,
              num_experts: int,
              renormalize: bool = False,
              hidden_dim: int = 1,
              ep_size: int = 1,
              ep_group: dist.ProcessGroup = None,
              layer_idx: int = 0,
              out_dtype: torch.dtype = torch.bfloat16,
              ffn_dim: int = 1,
              expert_offset: int = 0,
              swiglu_limit: float = 0.0):
        del renormalize, out_dtype
        if ep_size > 1:
            num_local_experts = num_experts // ep_size
            return TritonFusedMoEV4FP4EPImpl(
                ep_size=ep_size, ep_group=ep_group, top_k=top_k,
                num_experts=num_experts, num_local_experts=num_local_experts,
                hidden_dim=hidden_dim, ffn_dim=ffn_dim,
                swiglu_limit=swiglu_limit, layer_idx=layer_idx)
        return DeepGemmFusedMoEV4TPImpl(top_k=top_k,
                                        num_experts=num_experts,
                                        hidden_dim=hidden_dim,
                                        ffn_dim=ffn_dim,
                                        expert_offset=expert_offset,
                                        swiglu_limit=swiglu_limit)


class TritonFusedMoEV4FP4Builder(FusedMoEBuilder):
    """Builder for DeepSeek-V4 Triton FP4 fused MoE."""

    @staticmethod
    def build(top_k: int,
              num_experts: int,
              renormalize: bool = False,
              hidden_dim: int = 1,
              ep_size: int = 1,
              ep_group: dist.ProcessGroup = None,
              layer_idx: int = 0,
              out_dtype: torch.dtype = torch.bfloat16,
              ffn_dim: int = 1,
              expert_offset: int = 0,
              swiglu_limit: float = 0.0,
              scale_fmt: str | None = 'ue8m0'):
        del renormalize, out_dtype
        if ep_size > 1:
            num_local_experts = num_experts // ep_size
            return TritonFusedMoEV4FP4EPImpl(
                ep_size=ep_size, ep_group=ep_group, top_k=top_k,
                num_experts=num_experts, num_local_experts=num_local_experts,
                hidden_dim=hidden_dim, ffn_dim=ffn_dim,
                swiglu_limit=swiglu_limit, scale_fmt=scale_fmt,
                layer_idx=layer_idx)
        return TritonFusedMoEV4FP4TPImpl(top_k=top_k,
                                         num_experts=num_experts,
                                         hidden_dim=hidden_dim,
                                         ffn_dim=ffn_dim,
                                         expert_offset=expert_offset,
                                         swiglu_limit=swiglu_limit,
                                         scale_fmt=scale_fmt)

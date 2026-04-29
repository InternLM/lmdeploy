# Copyright (c) OpenMMLab. All rights reserved.
import torch

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.backends.moe import FusedMoEBuilder
from lmdeploy.pytorch.kernels.cuda.blocked_fp8_fused_moe import fused_moe_blocked_fp8
from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import quant_fp8
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def _v4_swiglu(intermediate: torch.Tensor, swiglu_limit: float) -> torch.Tensor:
    """Match DeepSeek-V4 expert activation in the blocked FP8 fused MoE
    path."""
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

    `invalid_expert` is backend-specific:
    - `-1` for the SM100 grouped FP8xFP4 path, matching TileKernels'
      fused-routing helpers.
    - `0` for the Hopper blocked FP8 path, because lmdeploy's blocked fused MoE
      routing kernel does not accept negative expert ids.
    """
    local_topk_ids = topk_ids.to(torch.int64) - expert_offset
    local_mask = (local_topk_ids >= 0) & (local_topk_ids < num_experts)
    local_topk_ids = torch.where(local_mask, local_topk_ids, local_topk_ids.new_full((), invalid_expert))
    local_topk_weights = torch.where(local_mask, topk_weights, topk_weights.new_zeros(()))
    return local_topk_ids, local_topk_weights, local_mask


class TritonFusedMoEV4BlockedF8TPImpl:
    """TP-only blocked FP8 fused MoE for DeepSeek-V4 on Hopper.

    DeepGEMM grouped FP8xFP4 kernels currently require SM100 for packed FP4 expert weights. On SM90/Hopper we instead
    convert expert weights to blocked FP8 at load time, then reuse lmdeploy's graph-friendly blocked FP8 fused MoE
    kernel while keeping DeepSeek-V4's expert-parallel routing semantics.
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
                down_scale: torch.Tensor,
                group: dist.ProcessGroup | None = None):
        num_tokens = hidden_states.size(0)
        if num_tokens == 0:
            return hidden_states.new_empty((0, self.hidden_dim))

        local_topk_ids, local_topk_weights, local_mask = _slice_local_topk(topk_ids,
                                                                           topk_weights,
                                                                           self.expert_offset,
                                                                           self.num_experts,
                                                                           invalid_expert=0)

        if not local_mask.any():
            out = hidden_states.new_zeros((num_tokens, self.hidden_dim))
            if group is not None:
                dist.all_reduce(out, group=group)
            return out

        input_quant, input_scale = quant_fp8(hidden_states,
                                             128,
                                             dtype=gate_up_weight.dtype,
                                             scale_fmt=self.scale_fmt)
        out = fused_moe_blocked_fp8(input_quant,
                                    input_scale,
                                    gate_up_weight,
                                    gate_up_scale,
                                    down_weight,
                                    down_scale,
                                    topk_weights=local_topk_weights.contiguous(),
                                    topk_ids=local_topk_ids.contiguous(),
                                    topk=self.top_k,
                                    out_dtype=hidden_states.dtype,
                                    renormalize=False,
                                    act_func=self._act_func)
        out = out.float()
        if group is not None:
            dist.all_reduce(out, group=group)
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
                down_scale: torch.Tensor,
                group: dist.ProcessGroup | None = None):
        deep_gemm, per_token_cast_to_fp8, get_fused_mapping, expand_to_fused_with_sf, reduce_fused, swiglu_forward = \
            self._import_impl()

        num_tokens = hidden_states.size(0)
        if num_tokens == 0:
            return hidden_states.new_empty((0, self.hidden_dim))

        local_topk_ids, _, local_mask = _slice_local_topk(topk_ids,
                                                          topk_weights,
                                                          self.expert_offset,
                                                          self.num_experts,
                                                          invalid_expert=-1)
        if not local_mask.any():
            out = hidden_states.new_zeros((num_tokens, self.hidden_dim))
            if group is not None:
                dist.all_reduce(out, group=group)
            return out

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
        if group is not None:
            dist.all_reduce(out, group=group)
        return out


class DeepGemmFusedMoEV4Builder(FusedMoEBuilder):
    """Builder for TP-only DeepSeek-V4 fused MoE."""

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
        del renormalize, ep_group, layer_idx, out_dtype
        if ep_size > 1:
            raise RuntimeError('DeepSeek-V4 fused MoE currently supports TP only; EP is not implemented.')
        return DeepGemmFusedMoEV4TPImpl(top_k=top_k,
                                        num_experts=num_experts,
                                        hidden_dim=hidden_dim,
                                        ffn_dim=ffn_dim,
                                        expert_offset=expert_offset,
                                        swiglu_limit=swiglu_limit)


class TritonFusedMoEV4BlockedF8Builder(FusedMoEBuilder):
    """Builder for TP-only DeepSeek-V4 blocked FP8 fused MoE."""

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
        del renormalize, ep_group, layer_idx, out_dtype
        if ep_size > 1:
            raise RuntimeError('DeepSeek-V4 fused MoE currently supports TP only; EP is not implemented.')
        return TritonFusedMoEV4BlockedF8TPImpl(top_k=top_k,
                                               num_experts=num_experts,
                                               hidden_dim=hidden_dim,
                                               ffn_dim=ffn_dim,
                                               expert_offset=expert_offset,
                                               swiglu_limit=swiglu_limit,
                                               scale_fmt=scale_fmt)

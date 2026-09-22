# Copyright (c) OpenMMLab. All rights reserved.
"""PyTorch engine implementation of multimodal GLM-5.3-Flash."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torch import distributed as dist
from torch import nn

from lmdeploy.pytorch.backends.cuda.attention.tilelang_sparse_mla import (
    TilelangSparseMLADecode,
)
from lmdeploy.pytorch.backends.cuda.kpool import (
    kpool_compress_quantize_cuda,
    kpool_score_contiguous_cuda,
    kpool_score_paged_cuda,
    kpool_select_groups_cuda,
)
from lmdeploy.pytorch.configurations.glm5_next import is_glm5_kda_layer
from lmdeploy.pytorch.consts import (
    GLM5_KDA_CONV_STATE,
    GLM5_KDA_RECURRENT_STATE,
    GLM5_KPOOL_TAIL_K_STATE,
    GLM5_KPOOL_TAIL_SCORE_STATE,
)
from lmdeploy.pytorch.distributed import get_dist_manager, get_tp_world_rank
from lmdeploy.pytorch.engine.cache_engine.schema import BlockCacheRequest
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager, get_step_ctx_manager
from lmdeploy.pytorch.nn import (
    FlashAttention,
    FP32LayerNorm,
    HcPrePost,
    Kda,
    KPoolIndexer,
    ParallelLMHead,
    RMSNorm,
    apply_rotary_pos_emb_fp32,
)
from lmdeploy.pytorch.nn.gated_delta import GatedDeltaMeta, GatedDeltaMetaBuilder, build_rmsnorm_gated
from lmdeploy.pytorch.nn.kpool import (
    kpool_decode_update,
    kpool_expand_selected_groups,
    kpool_partition_update,
    kpool_pooled_block_offsets,
    kpool_read_packed_cache,
    kpool_rotate_query,
    kpool_write_packed_cache,
    kpool_write_packed_cache_batched,
)
from lmdeploy.pytorch.nn.linear import (
    build_colwise_linear,
    build_merged_colwise_linear,
    build_o_proj,
    build_qkv_proj,
    build_rowwise_linear,
)
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight
from lmdeploy.vl.constants import Modality

from .deepseek_v2 import DeepseekV2MLP, DeepseekV2MoE
from .deepseek_v32 import DeepseekV32Attention, DeepseekV32ForCausalLM
from .glm4_1v import (
    Glm4vVisionAttention,
    Glm4vVisionPatchEmbed,
    Glm4vVisionRotaryEmbedding,
)
from .glm_moe_dsa import DSATopKIndicesBuffer
from .glm_moe_dsa_mtp import GlmMoeDsaMTPModel, GlmMoeDsaMultiTokenPredictor
from .qwen3_vl import Qwen3VLInputProcessor
from .utils.model import build_embedding, vlm_model

Glm5NextVisionRMSNorm = RMSNorm


class Glm5NextLayerNorm(FP32LayerNorm):
    """GLM-5.3 FP32 LayerNorm using LMDeploy's reusable implementation."""


# Backward-compatible name retained for the vision numerical contract tests
# and downstream imports.  The provider is also shared by the KPool indexer.
Glm5NextVisionLayerNorm = Glm5NextLayerNorm


def _build_glm53_latent_norm(hidden_size: int, eps: float,
                             dtype: torch.dtype | None,
                             device: torch.device | None) -> RMSNorm:
    """Build the unquantized BF16 latent norm used by GLM sparse attention."""
    if dtype is None:
        dtype = torch.get_default_dtype()
    return RMSNorm(hidden_size,
                   eps,
                   quant_config=None,
                   dtype=dtype,
                   device=device)


def _glm_swiglu_impl(intermediate: torch.Tensor,
                     swiglu_limit: float,
                     precise_mul: bool = False) -> torch.Tensor:
    """GLM/DeepSeek-V4 clamped SwiGLU used by dense and routed experts."""
    from lmdeploy.pytorch.kernels.cuda.activation import silu_and_mul
    input_shape = intermediate.shape
    intermediate = intermediate.flatten(0, -2)
    output = silu_and_mul(intermediate,
                          swiglu_limit=swiglu_limit,
                          precise_mul=precise_mul)
    return output.unflatten(0, input_shape[:-1])


_GLM53_COMPACT_FP8_MOE_ACT = partial(
    _glm_swiglu_impl, swiglu_limit=10.0, precise_mul=True)


class Glm5NextVisionPatchEmbed(Glm4vVisionPatchEmbed):
    """Patch embedding with SGLang's unfold-plus-linear reduction order."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ).to(dtype=target_dtype)
        hidden_states = hidden_states.flatten(1)
        weight = self.proj.weight.flatten(1)
        return F.linear(hidden_states, weight, self.proj.bias)


class Glm5NextVisionAttention(Glm4vVisionAttention):
    """GLM-OCR attention with GLM-5.3's per-head Q/K RMSNorm."""

    # SGLang's GLM-5.3 block leaves VisionAttention.layer_norm_eps at its
    # default instead of forwarding vision_config.rms_norm_eps.  Keep this
    # explicit because the outer block norms do use the config value.
    qk_norm_eps = 1e-6

    def __init__(self,
                 config: Any,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None):
        # Reuse LMDeploy's QKV/row-parallel projections and rotary operator.
        # Vision weights stay in BF16 even when the language tower is block-FP8.
        super().__init__(config, dtype=dtype, device=device)
        self.q_norm = Glm5NextVisionRMSNorm(self.head_dim,
                                           eps=self.qk_norm_eps,
                                           quant_config=None,
                                           dtype=dtype,
                                           device=device)
        self.k_norm = Glm5NextVisionRMSNorm(self.head_dim,
                                           eps=self.qk_norm_eps,
                                           quant_config=None,
                                           dtype=dtype,
                                           device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rotary_pos_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        qkv_states = self.qkv(hidden_states).flatten(0, -2)
        query, key, value = self.qkv.split_qkv(qkv_states)

        query = self.q_norm(query)
        key = self.k_norm(key)
        cos, sin = rotary_pos_emb
        query, key = apply_rotary_pos_emb_fp32(query, key, cos, sin)
        output = self.attention(
            query,
            key,
            value,
            q_start_loc=cu_seqlens[:-1],
            q_seqlens=cu_seqlens[1:] - cu_seqlens[:-1],
            max_q_seqlen=max_seqlen,
        )
        return self.proj(output.reshape(seq_length, -1))


class Glm5NextVisionMLP(nn.Module):
    """TP-aware vision MLP with GLM-5.3's clamped SwiGLU."""

    def __init__(self,
                 config: Any,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None):
        super().__init__()
        self.swiglu_limit = config.swiglu_limit
        self.gate_up_proj = build_merged_colwise_linear(
            in_features=config.hidden_size,
            all_out_features=[config.intermediate_size,
                              config.intermediate_size],
            bias=True,
            dtype=dtype,
            device=device,
            quant_config=None,
            is_tp=True,
        )
        self.down_proj = build_rowwise_linear(
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            bias=True,
            dtype=dtype,
            device=device,
            quant_config=None,
            is_tp=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.gate_up_proj(hidden_states)
        hidden_states = _glm_swiglu_impl(hidden_states,
                                         self.swiglu_limit,
                                         precise_mul=True)
        return self.down_proj(hidden_states)


class Glm5NextVisionPatchMerger(nn.Module):
    """GLM-5.3 vision projector after spatial downsampling."""

    def __init__(self,
                 config: Any,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None):
        super().__init__()
        dim = config.out_hidden_size
        context_dim = getattr(config, 'projection_intermediate_size',
                              config.intermediate_size)
        self.swiglu_limit = config.swiglu_limit
        self.proj = nn.Linear(dim,
                              dim,
                              bias=False,
                              dtype=dtype,
                              device=device)
        self.post_projection_norm = Glm5NextLayerNorm(dim,
                                                      eps=1e-6,
                                                      device=device)
        self.gate_up_proj = build_merged_colwise_linear(
            in_features=dim,
            all_out_features=[context_dim, context_dim],
            bias=False,
            dtype=dtype,
            device=device,
            quant_config=None,
            is_tp=True,
        )
        self.down_proj = build_rowwise_linear(
            in_features=context_dim,
            out_features=dim,
            bias=False,
            dtype=dtype,
            device=device,
            quant_config=None,
            is_tp=True,
        )
        self.act1 = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = self.act1(self.post_projection_norm(hidden_states))
        hidden_states = self.gate_up_proj(hidden_states)
        hidden_states = _glm_swiglu_impl(hidden_states,
                                         self.swiglu_limit,
                                         precise_mul=True)
        return self.down_proj(hidden_states)


class Glm5NextVisionBlock(nn.Module):
    """A GLM-5.3 vision block built from public LMDeploy operators."""

    def __init__(self,
                 config: Any,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None):
        super().__init__()
        self.norm1 = Glm5NextVisionRMSNorm(config.hidden_size,
                                          eps=config.rms_norm_eps,
                                          quant_config=None,
                                          dtype=dtype,
                                          device=device)
        self.norm2 = Glm5NextVisionRMSNorm(config.hidden_size,
                                          eps=config.rms_norm_eps,
                                          quant_config=None,
                                          dtype=dtype,
                                          device=device)
        self.attn = Glm5NextVisionAttention(config,
                                            dtype=dtype,
                                            device=device)
        self.mlp = Glm5NextVisionMLP(config,
                                     dtype=dtype,
                                     device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rotary_pos_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states,
                                  cu_seqlens=cu_seqlens,
                                  max_seqlen=max_seqlen,
                                  rotary_pos_emb=rotary_pos_emb)
        hidden_states, residual = self.norm2(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


@vlm_model
class Glm5NextVisionModel(nn.Module):
    """Native GLM-5.3 image/video encoder."""

    def __init__(self,
                 config: Any,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = Glm5NextVisionPatchEmbed(config,
                                                    dtype=dtype,
                                                    device=device)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Glm4vVisionRotaryEmbedding(
            head_dim // 2, device=device)
        self.blocks = nn.ModuleList([
            Glm5NextVisionBlock(config, dtype=dtype, device=device)
            for _ in range(config.depth)
        ])
        self.post_layernorm = Glm5NextVisionRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            quant_config=None,
            dtype=dtype,
            device=device)
        self.downsample = nn.Conv2d(
            in_channels=config.hidden_size,
            out_channels=config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
            dtype=dtype,
            device=device,
        )
        self.merger = Glm5NextVisionPatchMerger(config,
                                                dtype=dtype,
                                                device=device)

    def _rotary_embedding(
        self,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_ids = []
        merge = self.spatial_merge_size
        for t, h, w in grid_thw.tolist():
            if h % merge or w % merge:
                raise ValueError(
                    f'vision grid {(t, h, w)} is not divisible by merge={merge}.')
            hpos = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos = hpos.reshape(h // merge, merge, w // merge,
                                merge).permute(0, 2, 1, 3).flatten()
            wpos = wpos.reshape(h // merge, merge, w // merge,
                                merge).permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = int(grid_thw[:, 1:].max().item())
        full = self.rotary_pos_emb(max_grid_size)
        rotary = full[pos_ids.to(full.device)].flatten(1).repeat(1, 2)
        return rotary.cos(), rotary.sin()

    def forward(self, pixel_values: torch.Tensor,
                grid_thw: torch.Tensor) -> torch.Tensor:
        expected_patches = int(grid_thw.prod(dim=-1).sum().item())
        hidden_states = self.patch_embed(pixel_values)
        if hidden_states.shape[0] != expected_patches:
            raise ValueError(
                'vision patch count mismatch: '
                f'got {hidden_states.shape[0]}, expected {expected_patches}.')

        lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                          grid_thw[:, 0])
        max_seqlen = int(lengths.max().item())
        cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32),
                           (1, 0),
                           value=0).to(hidden_states.device)
        rotary_pos_emb = self._rotary_embedding(grid_thw)
        # Keep the trigonometric tables in FP32.  The vision RoPE helper casts
        # q/k to FP32 for the rotation and rounds only the final result.
        rotary_pos_emb = tuple(x.to(device=hidden_states.device)
                               for x in rotary_pos_emb)
        for block in self.blocks:
            hidden_states = block(hidden_states,
                                  cu_seqlens=cu_seqlens,
                                  max_seqlen=max_seqlen,
                                  rotary_pos_emb=rotary_pos_emb)

        hidden_states = self.post_layernorm(hidden_states)
        merge = self.spatial_merge_size
        hidden_states = hidden_states.view(-1, merge, merge,
                                           hidden_states.shape[-1])
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states)
        hidden_states = hidden_states.reshape(-1,
                                              self.config.out_hidden_size)
        hidden_states = self.merger(hidden_states)
        expected_tokens = expected_patches // (merge * merge)
        if hidden_states.shape[0] != expected_tokens:
            raise ValueError(
                'vision embedding count mismatch: '
                f'got {hidden_states.shape[0]}, expected {expected_tokens}.')
        return hidden_states


class Glm5NextInputProcessor(Qwen3VLInputProcessor):
    """Reuse the common image/video ``MultiModalData`` ownership boundary."""

    def __init__(self, config: Any, dtype: torch.dtype | None) -> None:
        super().__init__(config=config, dtype=dtype)
        if config.vision_config.spatial_merge_size != 2:
            raise ValueError('GLM-5.3 mRoPE currently requires merge size 2.')


class Glm5NextMLP(DeepseekV2MLP):
    """DeepSeek MLP projections with GLM-5.3's activation clamp."""

    def __init__(self, config: Any, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.swiglu_limit = config.swiglu_limit
        if get_dist_manager().current_config().dp == 1:
            self.down_proj.tp_reduce_dtype = torch.float32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        return self.down_proj(
            _glm_swiglu_impl(gate_up,
                              self.swiglu_limit,
                              precise_mul=True))


class Glm5NextNoauxTCRouter(nn.Module):
    """GLM noaux router using LMDeploy's existing Triton implementation."""

    def __init__(self, config: Any):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.scoring_func = config.scoring_func
        self.renormalize = bool(config.norm_topk_prob and self.top_k > 1)
        # Keep the generic router output normalized.  GLM's model-level 2.5
        # factor is owned by ``fused_moe_output_scale`` below so it is applied
        # once, after the FP32 routed-expert reduction.
        self.routed_scaling_factor = 1.0
        self.router_n_groups = getattr(config, 'router_n_groups', -1)
        contract = (
            getattr(config, 'model_type', None) == 'glm5_next_text'
            and config.topk_method == 'noaux_tc'
            and self.top_k == 8
            and self.num_experts == 288
            and self.n_group == 1
            and self.topk_group == 1
            and self.scoring_func == 'sigmoid'
            and self.renormalize
            and config.routed_scaling_factor == 2.5
            and self.router_n_groups == -1
            and getattr(config, 'moe_router_dtype', None) == 'float32'
            and getattr(config, 'n_shared_experts', None) == 1
        )
        if not contract:
            raise ValueError(
                'The shared GLM-5.3 router requires the exact official '
                'glm5_next_text noaux/FP32/288-expert/topk8 contract.')

    def forward(
        self,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized weights and expert ids."""
        from lmdeploy.pytorch.kernels.cuda.moe.route_noaux_tc import (
            fused_noaux_tc_routing,
        )

        return fused_noaux_tc_routing(
            logits=router_logits,
            bias=correction_bias,
            top_k=self.top_k,
            num_experts=self.num_experts,
            n_group=self.n_group,
            topk_group=self.topk_group,
            renormalize=self.renormalize,
            routed_scaling_factor=self.routed_scaling_factor,
        )


class Glm5NextMoE(DeepseekV2MoE):
    """DeepSeek routed experts with the same clamp as GLM-5.3."""

    # Keep only the model-specific clamp; dispatch, quantization and grouped
    # GEMMs are owned by LMDeploy's generic blocked-FP8 MoE implementation.
    fused_moe_act_func = staticmethod(_GLM53_COMPACT_FP8_MOE_ACT)
    fused_moe_fp32_acc = True
    # Match the GLM-5.3 contract: routing returns normalized, unscaled weights;
    # the 2.5 routed scale is applied once to the FP32 expert reduction before
    # its BF16 store.
    router_routed_scaling_factor = 1.0
    fused_moe_output_scale = 2.5
    shared_expert_cls = Glm5NextMLP

    def __init__(self, config: Any, layer_idx: int, *args, **kwargs):
        kwargs.setdefault('prefix', f'model.layers.{layer_idx}.mlp')
        super().__init__(config, layer_idx, *args, **kwargs)
        # Keep the shared+routed local sum and the generic expert kernels.
        # Promote only the final TP collective: BF16 collective reduction
        # order depends on message size (AR versus multi-token verification).
        self._fp32_tp_reduce = self._all_reduce
        self._all_reduce = False
        if self.gate.fake_eplb or self.gate.eplb_dispatch_info is not None:
            raise RuntimeError(
                'The GLM-5.3 router does not permit fake '
                'routing or EPLB expert remapping.')
        self.gate.noaux_tc_router = Glm5NextNoauxTCRouter(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_routed_experts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if all_routed_experts is not None:
            raise RuntimeError(
                'GLM-5.3 routed-expert capture is not supported.')
        out = super().forward(hidden_states, all_routed_experts=None)
        if self._fp32_tp_reduce:
            output_dtype = out.dtype
            out = out.float()
            dist.all_reduce(out, group=self.experts.tp_group)
            out = out.to(output_dtype)
        return out


def _load_vector_shard(param: nn.Parameter,
                       loaded_weight: torch.Tensor) -> None:
    """Load a flattened attention-head vector for the local TP rank."""
    world_size, rank = get_tp_world_rank('attn')
    loaded_weight = loaded_weight.flatten()
    if world_size > 1:
        loaded_weight = loaded_weight.chunk(world_size, dim=0)[rank]
    param.data.copy_(loaded_weight.to(device=param.device, dtype=param.dtype))


class Glm5NextQKVConv1d(nn.Module):
    """FP32 depthwise-convolution weight container for fused Q/K/V KDA."""

    _SHARD_IDS = {'q': 0, 'k': 1, 'v': 2, 0: 0, 1: 1, 2: 2}

    def __init__(self,
                 local_projection_size: int,
                 kernel_size: int,
                 device: torch.device | None = None):
        super().__init__()
        self.local_projection_size = local_projection_size
        weight = torch.empty(3 * local_projection_size,
                             1,
                             kernel_size,
                             dtype=torch.float32,
                             device=device)
        self.weight = nn.Parameter(weight)
        self.weight.weight_loader = self._weight_loader

    def _weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                       shard_id: str | int) -> None:
        world_size, rank = get_tp_world_rank('attn')
        if world_size > 1:
            loaded_weight = loaded_weight.chunk(world_size, dim=0)[rank]
        shard = self._SHARD_IDS[shard_id]
        start = shard * self.local_projection_size
        target = param.data.narrow(0, start, self.local_projection_size)
        target.copy_(loaded_weight.to(device=target.device,
                                      dtype=target.dtype))


class Glm5NextLinearAttention(nn.Module):
    """GLM-5.3 Kimi Delta Attention projections and backend dispatch."""

    def __init__(self,
                 config: Any,
                 layer_idx: int,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None,
                 all_reduce: bool = True):
        super().__init__()
        linear_config = config.linear_attn_config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = linear_config['num_heads']
        self.head_dim = linear_config['head_dim']
        self.conv_kernel_size = linear_config['short_conv_kernel_size']
        self.lower_bound = linear_config.get('gate_lower_bound', -5.0)

        tp, _ = get_tp_world_rank('attn')
        if self.num_heads % tp:
            raise ValueError(
                f'KDA heads={self.num_heads} is not divisible by attention TP={tp}.'
            )
        self.local_num_heads = self.num_heads // tp
        projection_size = self.num_heads * self.head_dim
        local_projection_size = self.local_num_heads * self.head_dim

        # The official checkpoint keeps every KDA projection in BF16 even
        # though the MLA/MLP weights use block FP8.  Do not inherit the global
        # quantization policy for these modules.
        self.qkv_proj = build_qkv_proj(
            self.hidden_size,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_size=self.head_dim,
            bias=False,
            quant_config=None,
            dtype=dtype,
            device=device,
            is_tp=True,
        )
        self.b_proj = build_colwise_linear(
            self.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=None,
            dtype=dtype,
            device=device,
            is_tp=True,
        )
        self.f_a_proj = build_colwise_linear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=None,
            dtype=dtype,
            device=device,
            is_tp=False,
        )
        self.f_b_proj = build_colwise_linear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=None,
            dtype=dtype,
            device=device,
            is_tp=True,
        )
        self.g_b_proj = build_colwise_linear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=None,
            dtype=dtype,
            device=device,
            is_tp=True,
        )
        self.g_a_proj = build_colwise_linear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=None,
            dtype=dtype,
            device=device,
            is_tp=False,
        )

        self.qkv_conv1d = Glm5NextQKVConv1d(local_projection_size,
                                            self.conv_kernel_size,
                                            device=device)
        self.A_log = nn.Parameter(
            torch.empty(self.local_num_heads,
                        dtype=torch.float32,
                        device=device))
        self.dt_bias = nn.Parameter(
            torch.empty(local_projection_size,
                        dtype=torch.float32,
                        device=device))
        self.A_log.weight_loader = _load_vector_shard
        self.dt_bias.weight_loader = _load_vector_shard

        self.o_norm = build_rmsnorm_gated(
            self.head_dim,
            eps=config.rms_norm_eps,
            activation='sigmoid',
            dtype=dtype,
            device=device,
        )
        self.o_proj = build_o_proj(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=None,
            dtype=dtype,
            device=device,
            is_tp=True,
            all_reduce=all_reduce,
        )
        if get_dist_manager().current_config().dp == 1:
            self.o_proj.tp_reduce_dtype = torch.float32
        self.kda = Kda()

    def forward(self, hidden_states: torch.Tensor,
                past_key_value: Sequence[torch.Tensor],
                kda_metadata: GatedDeltaMeta) -> torch.Tensor:
        mixed_qkv = self.qkv_proj(hidden_states)
        raw_beta = self.b_proj(hidden_states)
        raw_gate = self.f_b_proj(self.f_a_proj(hidden_states))
        norm_gate = self.g_b_proj(self.g_a_proj(hidden_states))

        core_output = self.kda(
            mixed_qkv=mixed_qkv,
            raw_gate=raw_gate,
            raw_beta=raw_beta,
            conv_weight=self.qkv_conv1d.weight,
            conv_bias=None,
            a_log=self.A_log,
            dt_bias=self.dt_bias,
            conv_state=past_key_value[0],
            recurrent_state=past_key_value[1],
            metadata=kda_metadata,
            num_heads=self.local_num_heads,
            head_dim=self.head_dim,
            lower_bound=self.lower_bound,
        )
        norm_gate = norm_gate.unflatten(-1,
                                        (self.local_num_heads, self.head_dim))
        output_shape = core_output.shape
        core_output = self.o_norm(core_output.reshape(-1, self.head_dim),
                                  norm_gate.reshape(-1, self.head_dim))
        core_output = core_output.view(output_shape).flatten(-2, -1)
        return self.o_proj(core_output)


class Glm5NextSparseAttention(DeepseekV32Attention):
    """GLM MLA without RoPE and with a pageable KPool-4 indexer."""

    use_sparse_mla = False
    mla_head_padding = 64

    def __init__(self,
                 config: Any,
                 layer_idx: int,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None,
                 all_reduce: bool = True):
        super().__init__(config,
                         layer_idx,
                         dtype=dtype,
                         device=device,
                         all_reduce=all_reduce,
                         prefix=f'model.layers.{layer_idx}.self_attn')
        if get_dist_manager().current_config().dp == 1:
            self.o_proj.tp_reduce_dtype = torch.float32
        # DeepSeek keeps these latent-norm parameters in FP32.  GLM-5.3's
        # checkpoint and SGLang runtime keep them in the activation dtype;
        # rebuild only these two containers before weight loading.
        if self.q_lora_rank is not None:
            self.q_a_layernorm = _build_glm53_latent_norm(
                config.q_lora_rank, config.rms_norm_eps, dtype, device)
        self.kv_a_layernorm = _build_glm53_latent_norm(
            config.kv_lora_rank, config.rms_norm_eps, dtype, device)
        self.index_topk = config.index_topk
        self.index_kpool = config.index_kpool
        try:
            full_layer_ids = list(config.full_attention_layer_ids)
            full_layer_ids.extend(range(config.num_hidden_layers,
                                        config.num_hidden_layers +
                                        getattr(config, 'num_nextn_predict_layers', 0)))
            self.cache_layer_idx = full_layer_ids.index(layer_idx)
        except ValueError as error:
            raise ValueError(
                f'GLM-5.3 full-attention layer {layer_idx} is missing from '
                'the compact cache map.') from error

        # Keep the checkpoint's BF16 KV-B projection alongside the absorbed
        # KC/VC views. Short prefill uses the former to reproduce SGLang's
        # decompressed dense MHA; decode continues to use KC/VC. Both must
        # shard by attention TP, including when attention DP is enabled.
        self.kv_b_proj = build_colwise_linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=True,
            quant_config=None,
        )
        self.prefill_attn_fwd = FlashAttention(
            self.num_heads,
            self.q_head_dim,
            scale=self.softmax_scale,
            v_head_dim=self.v_head_dim,
            causal=True,
        )
        self.decode_attn_fwd = TilelangSparseMLADecode(
            index_topk=self.index_topk,
            index_kpool=getattr(config, 'index_kpool', 1),
        )

    def _build_indexer(self, config: Any, layer_idx: int, dtype: torch.dtype,
                       device: torch.device, prefix: str = ''):
        del layer_idx, prefix
        return KPoolIndexer(
            hidden_size=config.hidden_size,
            index_n_heads=config.index_n_heads,
            index_head_dim=config.index_head_dim,
            index_topk=config.index_topk,
            q_lora_rank=config.q_lora_rank,
            index_kpool=config.index_kpool,
            dtype=dtype,
            device=device,
            key_norm=Glm5NextLayerNorm(config.index_head_dim,
                                       eps=1e-6,
                                       device=device),
        )

    def _qkv_proj_unabsorbed(self, hidden_states: torch.Tensor,
                             num_heads: int):
        """Project raw per-head Q and latent KV without absorption."""
        nope_size = self.kv_lora_rank
        pe_size = self.qk_rope_head_dim
        if self.q_lora_rank is None:
            q_a_states = hidden_states
            key_states = self.kv_a_proj_with_mqa(hidden_states[0, :, None])
        else:
            q_a_states, key_states = self.fused_qkv_a_proj(
                hidden_states).split([self.q_lora_rank, nope_size + pe_size],
                                     dim=-1)
            key_states = key_states[0, :, None]

        q_len = q_a_states.size(1)
        if self.q_lora_rank is None:
            q_lora = q_a_states
            query = self.q_proj(q_a_states)
        else:
            q_lora = self.q_a_layernorm(q_a_states)
            query = self.q_b_proj(q_lora)
        query = query.view(q_len, num_heads, self.q_head_dim)

        key_states, value_states, _ = self._kv_proj(key_states, nope_size)
        return query, key_states, value_states, q_lora

    def _kv_proj(self, key_states: torch.Tensor, nope_size: int):
        """Normalize latent KV with GLM-5.3's exact FP32 math order."""
        k_pe = key_states[..., nope_size:]
        value_states = key_states[..., :nope_size]
        value_states = self.kv_a_layernorm(value_states)
        key_states[..., :nope_size] = value_states
        return key_states, value_states, k_pe

    def _update_kpool_cache(
        self,
        hidden_states: torch.Tensor,
        tail_state: Sequence[torch.Tensor],
        state_ids: torch.Tensor,
        attn_metadata: Any,
    ) -> torch.Tensor:
        """Compress closed pools and persist each request's unfinished tail."""
        if tail_state is None or len(tail_state) != 2:
            raise RuntimeError(
                'GLM-5.3 KPool requires key and score tail state caches.')
        if state_ids is None:
            raise RuntimeError('GLM-5.3 KPool requires stable state cache ids.')

        tail_k_state, tail_score_state = tail_state
        history_lengths = attn_metadata.kv_seqlens - attn_metadata.q_seqlens
        ring_states = None
        if tail_k_state.ndim == 4:
            ring_states = tail_state
            ring_size = tail_k_state.size(1)
            request_ids = state_ids.clamp_min(0).long()
            valid_requests = state_ids >= 0
            read_slots = history_lengths.long().remainder(ring_size)
            tail_k_state = tail_k_state[request_ids, read_slots].clone()
            tail_score_state = tail_score_state[request_ids, read_slots].clone()
            local_ids = torch.arange(state_ids.numel(), device=state_ids.device)
            # Padding uses its own scratch row, never live request row zero.
            state_ids = local_ids

        def save_ring(lengths):
            if ring_states is None:
                return
            slots = lengths.long().remainder(ring_size)
            for state, value in zip(ring_states, (tail_k_state, tail_score_state)):
                previous = state[request_ids, slots]
                state[request_ids, slots] = torch.where(
                    valid_requests[:, None, None], value, previous)

        indexer_k_cache = self.indexer.get_block_cache()
        key = self.indexer.project_key(hidden_states)[0]
        score = self.indexer.project_compress_score(hidden_states)[0]
        if attn_metadata.is_decoding:
            batch_size = state_ids.numel()
            if key.size(0) % batch_size:
                raise RuntimeError(
                    'KPool decode rows must be divisible by request count.')
            steps = key.size(0) // batch_size
            key = key.unflatten(0, (batch_size, steps))
            score = score.unflatten(0, (batch_size, steps))
            for step in range(steps):
                update = kpool_decode_update(
                    key[:, step], score[:, step], tail_k_state,
                    tail_score_state, state_ids, history_lengths + step,
                    self.index_kpool)
                pooled_fp8, pooled_scale = kpool_compress_quantize_cuda(
                    update.closed_keys, update.closed_scores,
                    self.indexer.index_kpool_compress_ape,
                    mode='decode', round_scale=self.indexer.scale_fmt is not None)
                kpool_write_packed_cache_batched(
                    indexer_k_cache, attn_metadata.block_offsets,
                    update.group_ids, pooled_fp8, pooled_scale,
                    self.index_kpool, update.should_close)
                tail_k_state.index_copy_(0, update.safe_state_ids,
                                        update.next_tail_keys)
                tail_score_state.index_copy_(0, update.safe_state_ids,
                                            update.next_tail_scores)
                save_ring(history_lengths + step + 1)
            return indexer_k_cache

        q_seqlens = attn_metadata.q_seqlens.tolist()
        kv_seqlens = attn_metadata.kv_seqlens.tolist()
        if len(q_seqlens) != len(kv_seqlens):
            raise RuntimeError('KPool query/KV batch lengths do not match.')
        if state_ids.numel() != len(q_seqlens):
            raise RuntimeError(
                'KPool state id count does not match the request batch.')

        token_offset = 0
        for batch_idx, (q_len, kv_len) in enumerate(
                zip(q_seqlens, kv_seqlens)):
            q_len = int(q_len)
            kv_len = int(kv_len)
            history_len = kv_len - q_len
            if history_len < 0:
                raise RuntimeError(
                    f'KPool received q_len={q_len} greater than kv_len={kv_len}.')
            state_id = int(state_ids[batch_idx].item())
            previous_tail_len = history_len % self.index_kpool
            if state_id >= 0:
                previous_tail_k = tail_k_state[
                    state_id, :previous_tail_len]
                previous_tail_score = tail_score_state[
                    state_id, :previous_tail_len]
            else:
                previous_tail_k = key.new_zeros(
                    previous_tail_len, self.indexer.head_dim)
                previous_tail_score = score.new_zeros(
                    previous_tail_len, self.indexer.head_dim)

            token_end = token_offset + q_len
            update = kpool_partition_update(
                key[token_offset:token_end],
                score[token_offset:token_end],
                history_length=history_len,
                pool_size=self.index_kpool,
                tail_keys=previous_tail_k,
                tail_scores=previous_tail_score,
            )
            if update.closed_group_ids.numel():
                pooled_fp8, pooled_scale = kpool_compress_quantize_cuda(
                    update.closed_keys,
                    update.closed_scores,
                    self.indexer.index_kpool_compress_ape,
                    mode=('decode'
                          if attn_metadata.is_decoding else 'extend'),
                    round_scale=self.indexer.scale_fmt is not None,
                )
                kpool_write_packed_cache(
                    indexer_k_cache,
                    attn_metadata.block_offsets[batch_idx],
                    update.closed_group_ids,
                    pooled_fp8,
                    pooled_scale,
                    self.index_kpool,
                )

            if state_id >= 0:
                tail_k_state[state_id].zero_()
                tail_score_state[state_id].zero_()
                tail_len = update.tail_keys.size(0)
                if tail_len:
                    tail_k_state[state_id, :tail_len].copy_(
                        update.tail_keys)
                    tail_score_state[state_id, :tail_len].copy_(
                        update.tail_scores)
            token_offset = token_end

        if token_offset != key.size(0):
            raise RuntimeError(
                f'KPool metadata accounts for {token_offset} tokens, '
                f'but projections contain {key.size(0)}.')
        save_ring(attn_metadata.kv_seqlens)
        return indexer_k_cache

    def _select_kpool_indices(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        indexer_k_cache: torch.Tensor,
        attn_metadata: Any,
    ) -> torch.Tensor:
        """Score/select on attention-TP rank 0, then broadcast logical ids."""
        dist_ctx = get_dist_manager().current_context()
        tp_group = dist_ctx.attn_tp_group
        is_owner = tp_group.rank == 0
        total_rows = hidden_states.size(1)
        output_width = self.index_topk + self.index_kpool - 1

        if is_owner:
            query = self.indexer.project_query(q_lora)[0]
            query = kpool_rotate_query(query)
            query_fp8, query_scale = self.indexer.quantize_fp8(query)
            head_gate = self.indexer.project_head_gate(hidden_states)[0]
            query_weight = (head_gate * query_scale.squeeze(-1)
                            * self.indexer.softmax_scale)
            if attn_metadata.is_decoding:
                batch_size = attn_metadata.kv_seqlens.numel()
                steps = total_rows // batch_size
                history = attn_metadata.kv_seqlens - attn_metadata.q_seqlens
                step_ids = torch.arange(1, steps + 1, device=query_fp8.device)
                seq_lens = (history[:, None] + step_ids).flatten().to(torch.int64)
                group_lengths = torch.div(
                    seq_lens,
                    self.index_kpool,
                    rounding_mode='floor',
                )
                pooled_block_offsets = kpool_pooled_block_offsets(
                    attn_metadata.block_offsets.repeat_interleave(steps, dim=0),
                    self.index_kpool,
                )
                logits = kpool_score_paged_cuda(
                    query_fp8,
                    query_weight,
                    indexer_k_cache,
                    group_lengths,
                    pooled_block_offsets,
                )
                selected_groups = kpool_select_groups_cuda(
                    logits.contiguous(),
                    group_lengths,
                    group_topk=self.index_topk // self.index_kpool,
                )
                logical_indices = kpool_expand_selected_groups(
                    selected_groups,
                    group_lengths,
                    self.index_kpool,
                    self.index_topk,
                    seq_lens=seq_lens,
                )
            else:
                logical_indices = self._select_kpool_indices_prefill(
                    query_fp8,
                    query_weight,
                    indexer_k_cache,
                    attn_metadata,
                )
        else:
            logical_indices = torch.empty(
                total_rows,
                output_width,
                dtype=torch.int32,
                device=hidden_states.device,
            )

        if dist_ctx.dist_config.attn_tp > 1:
            group = tp_group.gpu_group
            source_rank = dist_ctx.rank - tp_group.rank
            dist.broadcast(logical_indices, src=source_rank, group=group)
        return logical_indices

    def _select_kpool_indices_prefill(
        self,
        query_fp8: torch.Tensor,
        query_weight: torch.Tensor,
        indexer_k_cache: torch.Tensor,
        attn_metadata: Any,
    ) -> torch.Tensor:
        """Retain the ragged eager implementation for chunked prefill."""
        q_seqlens = attn_metadata.q_seqlens.tolist()
        kv_seqlens = attn_metadata.kv_seqlens.tolist()
        logical_parts = []
        token_offset = 0
        for batch_idx, (q_len, kv_len) in enumerate(
                zip(q_seqlens, kv_seqlens)):
            q_len = int(q_len)
            kv_len = int(kv_len)
            history_len = kv_len - q_len
            num_groups = kv_len // self.index_kpool
            token_end = token_offset + q_len
            seq_lens = history_len + torch.arange(
                1,
                q_len + 1,
                dtype=torch.int64,
                device=query_fp8.device,
            )
            group_lengths = torch.div(
                seq_lens,
                self.index_kpool,
                rounding_mode='floor',
            )
            query_slice = query_fp8[token_offset:token_end]
            weight_slice = query_weight[token_offset:token_end]
            pooled_key, pooled_scale = kpool_read_packed_cache(
                indexer_k_cache,
                attn_metadata.block_offsets[batch_idx],
                num_groups,
                self.index_kpool,
            )
            logits = kpool_score_contiguous_cuda(
                query_slice,
                weight_slice,
                pooled_key,
                pooled_scale,
                group_lengths,
            )
            group_budget = self.index_topk // self.index_kpool
            selected_groups = kpool_select_groups_cuda(
                logits.contiguous(),
                group_lengths,
                group_topk=group_budget,
                max_group_length=num_groups,
            )
            logical_parts.append(
                kpool_expand_selected_groups(
                    selected_groups,
                    group_lengths,
                    self.index_kpool,
                    self.index_topk,
                    seq_lens=seq_lens,
                ))
            token_offset = token_end
        return torch.cat(logical_parts, dim=0)

    def _kpool_indices(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        tail_state: Sequence[torch.Tensor],
        state_ids: torch.Tensor,
        attn_metadata: Any,
        return_indices: bool,
        topk_indices_buffer: DSATopKIndicesBuffer | None = None,
        skip_topk: bool = False,
    ) -> torch.Tensor | None:
        indexer_k_cache = self._update_kpool_cache(
            hidden_states, tail_state, state_ids, attn_metadata)
        if topk_indices_buffer is not None and skip_topk:
            return (topk_indices_buffer.read(hidden_states.size(1), hidden_states.device)
                    if return_indices else None)
        if not return_indices and topk_indices_buffer is None:
            return None
        indices = self._select_kpool_indices(
            hidden_states, q_lora, indexer_k_cache, attn_metadata)
        if topk_indices_buffer is not None:
            # MTP needs seed indices even for dense short prefill: subsequent
            # draft steps reuse its last-token rows through the shared proposer.
            indices = topk_indices_buffer.write(indices)
        return indices if return_indices else None

    def _absorbed_query(self, query: torch.Tensor,
                        num_heads: int) -> torch.Tensor:
        query_states = query.new_empty(
            query.size(0), num_heads,
            self.kv_lora_rank + self.qk_rope_head_dim)
        self.kc(query[..., :self.qk_nope_head_dim],
                query_states[..., :self.kv_lora_rank])
        return query_states

    def _forward_prefill_mha(
        self,
        query: torch.Tensor,
        key_states: torch.Tensor,
        past_key_value: Sequence[torch.Tensor],
        attn_metadata: Any,
        num_heads: int,
    ) -> torch.Tensor:
        """Run short prefill as decompressed dense MHA over full latent KV."""
        k_scales_zeros = None if len(
            past_key_value) == 2 else past_key_value[2]
        v_scales_zeros = None if len(
            past_key_value) == 2 else past_key_value[3]
        flatten_latent = self.attn_fwd.fill_and_flatten_latent_kv_cache(
            key_states,
            past_key_value[0],
            attn_metadata,
            out_dtype=query.dtype,
            k_scales_zeros=k_scales_zeros,
            v_scales_zeros=v_scales_zeros,
        )
        flatten_latent = flatten_latent[..., :self.kv_lora_rank].flatten(0, 1)
        kv_states = self.kv_b_proj(flatten_latent)
        kv_states = kv_states.view(-1, num_heads,
                                   self.qk_nope_head_dim + self.v_head_dim)
        key, value = kv_states.split([self.qk_nope_head_dim, self.v_head_dim],
                                     dim=-1)
        attn_output = self.prefill_attn_fwd(
            query,
            key,
            value,
            q_start_loc=attn_metadata.cu_seqlens_q[:-1],
            q_seqlens=(attn_metadata.cu_seqlens_q[1:]
                       - attn_metadata.cu_seqlens_q[:-1]),
            kv_start_loc=attn_metadata.cu_seqlens_k[:-1],
            kv_seqlens=(attn_metadata.cu_seqlens_k[1:]
                        - attn_metadata.cu_seqlens_k[:-1]),
            max_q_seqlen=attn_metadata.max_q_seqlen,
        )
        return self.o_proj(attn_output.flatten(-2, -1)[None])

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Sequence[torch.Tensor],
        attn_metadata: Any = None,
        kpool_tail_state: Sequence[torch.Tensor] | None = None,
        state_ids: torch.Tensor | None = None,
        topk_indices_buffer: DSATopKIndicesBuffer | None = None,
        skip_topk: bool = False,
    ) -> torch.Tensor:
        dist_ctx = get_dist_manager().current_context()
        num_heads = self.num_heads // dist_ctx.dist_config.attn_tp
        nope_size = self.kv_lora_rank
        q_len = hidden_states.size(1)

        (unabsorbed_query, key_states, value_states,
         q_lora) = self._qkv_proj_unabsorbed(
             hidden_states, num_heads=num_heads)
        # The latent cache retains FlashMLA's 576-wide DeepSeek layout. GLM
        # has no RoPE tail, so its final 64 dimensions are exact zeros.
        key_states = F.pad(key_states, (0, self.mla_head_padding))

        if not attn_metadata.is_decoding:
            use_sparse = int(attn_metadata.max_kv_seqlen) > self.index_topk
            logical_indices = self._kpool_indices(
                hidden_states,
                q_lora,
                kpool_tail_state,
                state_ids,
                attn_metadata,
                return_indices=use_sparse,
                topk_indices_buffer=topk_indices_buffer,
                skip_topk=skip_topk,
            )
            if not use_sparse:
                return self._forward_prefill_mha(
                    unabsorbed_query,
                    key_states,
                    past_key_value,
                    attn_metadata,
                    num_heads,
                )

            query_states = self._absorbed_query(
                unabsorbed_query, num_heads)
            attn_output = self.decode_attn_fwd.forward_prefill(
                query_states,
                key_states,
                past_key_value[0],
                attn_metadata,
                scale=self.softmax_scale,
                cache_writer=self.attn_fwd,
                logical_indices=logical_indices,
                k_scales_zeros=(None if len(past_key_value) == 2 else
                                past_key_value[2]),
                v_scales_zeros=(None if len(past_key_value) == 2 else
                                past_key_value[3]),
            )
            attn_bmm_out = attn_output.new_empty(
                q_len, num_heads, self.v_head_dim)
            self.vc(attn_output, attn_bmm_out)
            return self.o_proj(attn_bmm_out.flatten(-2, -1)[None])

        logical_indices = self._kpool_indices(
            hidden_states,
            q_lora,
            kpool_tail_state,
            state_ids,
            attn_metadata,
            return_indices=True,
            topk_indices_buffer=topk_indices_buffer,
            skip_topk=skip_topk,
        )
        # GLM has no RoPE tail, so the absorbed query contains exactly 512
        # values; the cache retains its 576-wide FlashMLA storage alignment.
        query_states = self._absorbed_query(unabsorbed_query, num_heads)
        attn_output = self.decode_attn_fwd.forward(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[0][..., :nope_size],
            attn_metadata,
            scale=self.softmax_scale,
            cache_writer=self.attn_fwd,
            k_scales_zeros=(None if len(past_key_value) == 2 else
                            past_key_value[2]),
            v_scales_zeros=(None if len(past_key_value) == 2 else
                            past_key_value[3]),
            logical_indices=logical_indices,
        )
        attn_bmm_out = attn_output.new_empty(q_len, num_heads, self.v_head_dim)
        self.vc(attn_output, attn_bmm_out)
        return self.o_proj(attn_bmm_out.flatten(-2, -1)[None])


class Glm5NextDecoderLayer(nn.Module):
    """Hybrid KDA/MLA decoder block with mHC pre/post mixing."""

    def __init__(self,
                 config: Any,
                 layer_idx: int,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_linear_attention = is_glm5_kda_layer(config, layer_idx)
        if self.is_linear_attention:
            self.self_attn = Glm5NextLinearAttention(config,
                                                     layer_idx,
                                                     dtype=dtype,
                                                     device=device)
        else:
            self.self_attn = Glm5NextSparseAttention(config,
                                                     layer_idx,
                                                     dtype=dtype,
                                                     device=device)

        mlp_layer_types = getattr(config, 'mlp_layer_types', None)
        is_sparse = (mlp_layer_types is not None
                     and mlp_layer_types[layer_idx] == 'sparse')
        if mlp_layer_types is None:
            is_sparse = (config.n_routed_experts is not None
                         and layer_idx >= config.first_k_dense_replace
                         and layer_idx % config.moe_layer_freq == 0)
        self.mlp = (Glm5NextMoE(config, layer_idx, dtype=dtype, device=device)
                    if is_sparse else Glm5NextMLP(
                        config, dtype=dtype, device=device,
                        prefix=f'model.layers.{layer_idx}.mlp'))

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.rms_norm_eps,
                                       quant_config=None,
                                       dtype=dtype,
                                       device=device)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps,
                                                quant_config=None,
                                                dtype=dtype,
                                                device=device)
        self.hc_prepost = HcPrePost(config.hc_mult,
                                    config.hc_sinkhorn_iters,
                                    config.hc_eps,
                                    avoid_gemv=True)
        mix_hc = (2 + config.hc_mult) * config.hc_mult
        hc_dim = config.hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc,
                                                   hc_dim,
                                                   dtype=torch.float32,
                                                   device=device),
                                       requires_grad=False)
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc,
                                                  hc_dim,
                                                  dtype=torch.float32,
                                                  device=device),
                                      requires_grad=False)
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc,
                                                     dtype=torch.float32,
                                                     device=device),
                                         requires_grad=False)
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc,
                                                    dtype=torch.float32,
                                                    device=device),
                                        requires_grad=False)
        self.hc_attn_scale = nn.Parameter(torch.empty(3,
                                                      dtype=torch.float32,
                                                      device=device),
                                          requires_grad=False)
        self.hc_ffn_scale = nn.Parameter(torch.empty(3,
                                                     dtype=torch.float32,
                                                     device=device),
                                         requires_grad=False)

    def _hc_pre(self, hidden_states: torch.Tensor, fn: torch.Tensor,
                scale: torch.Tensor, base: torch.Tensor, norm: RMSNorm):
        return self.hc_prepost.pre(
            hidden_states, fn, scale, base, norm.eps)

    def forward(self, hidden_states: torch.Tensor,
                past_key_value: Sequence[torch.Tensor], attn_metadata: Any,
                kda_metadata: GatedDeltaMeta,
                kpool_tail_state: Sequence[torch.Tensor] | None = None,
                state_ids: torch.Tensor | None = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states, post, comb = self._hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.input_layernorm,
        )
        hidden_states = self.input_layernorm(hidden_states)
        if self.is_linear_attention:
            hidden_states = self.self_attn(hidden_states,
                                           past_key_value=past_key_value,
                                           kda_metadata=kda_metadata)
        else:
            hidden_states = self.self_attn(hidden_states,
                                           past_key_value=past_key_value,
                                           attn_metadata=attn_metadata,
                                           kpool_tail_state=kpool_tail_state,
                                           state_ids=state_ids)
        hidden_states = self.hc_prepost.post_expand(hidden_states, residual,
                                                    post, comb)

        residual = hidden_states
        hidden_states, post, comb = self._hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.post_attention_layernorm,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return self.hc_prepost.post_expand(hidden_states, residual, post, comb)


class Glm5NextModel(nn.Module):
    """GLM-5.3 hybrid backbone shared by text and multimodal generation."""

    def __init__(self,
                 config: Any,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None):
        super().__init__()
        self.config = config
        self.gated_delta_meta_builder = GatedDeltaMetaBuilder()
        self.embed_tokens = build_embedding(config.vocab_size,
                                            config.hidden_size,
                                            config.pad_token_id,
                                            dtype=dtype,
                                            device=device,
                                            is_tp=True)
        self.layers = nn.ModuleList([
            Glm5NextDecoderLayer(config, layer_idx, dtype=dtype, device=device)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size,
                            config.rms_norm_eps,
                            quant_config=None,
                            dtype=dtype,
                            device=device)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None,
        past_key_values: list[Sequence[torch.Tensor]],
        attn_metadata: Any,
        state_ids: torch.Tensor,
        kpool_tail_states: list[Sequence[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del position_ids  # GLM-5.3 text attention has qk_rope_head_dim == 0.
        if state_ids is None:
            raise RuntimeError('GLM-5.3 KDA requires stable state cache ids.')
        if kpool_tail_states is None:
            raise RuntimeError('GLM-5.3 requires KPool tail state caches.')
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds.unsqueeze(2).repeat(
            1, 1, self.config.hc_mult, 1)
        kda_metadata = self.gated_delta_meta_builder(hidden_states.size(1),
                                      self.config.linear_conv_kernel_dim,
                                      state_ids, attn_metadata)
        if len(past_key_values) != len(self.layers):
            raise RuntimeError(
                f'GLM-5.3 expects {len(self.layers)} layer caches, got {len(past_key_values)}.'
            )
        expected_full_layers = len(self.config.full_attention_layer_ids)
        if len(kpool_tail_states) != expected_full_layers:
            raise RuntimeError(
                f'GLM-5.3 expects {expected_full_layers} KPool tail rows, '
                f'got {len(kpool_tail_states)}.')
        full_layer_row = 0
        for layer, past_key_value in zip(self.layers, past_key_values):
            kpool_tail_state = None
            if not layer.is_linear_attention:
                kpool_tail_state = kpool_tail_states[full_layer_row]
                full_layer_row += 1
            hidden_states = layer(hidden_states,
                                  past_key_value=past_key_value,
                                  attn_metadata=attn_metadata,
                                  kda_metadata=kda_metadata,
                                  kpool_tail_state=kpool_tail_state,
                                  state_ids=state_ids)
        hidden_states = hidden_states.mean(dim=2)
        return self.norm(hidden_states)

    def get_input_embeddings(self):
        return self.embed_tokens


class Glm5NextForConditionalGeneration(DeepseekV32ForCausalLM):
    """GLM-5.3 conditional-generation wrapper for text, image and video."""

    def __init__(self,
                 config: Any,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None):
        nn.Module.__init__(self)
        self.mm_config = config
        self.config = config.text_config
        self.quantization_config = getattr(config, 'quantization_config', None)
        if self.quantization_config is not None:
            self.config.quantization_config = self.quantization_config
        self.dtype = dtype
        self.ctx_mgr = ctx_mgr
        self.input_processor = Glm5NextInputProcessor(config, dtype)
        self.visual = Glm5NextVisionModel(config.vision_config,
                                          dtype=dtype,
                                          device=device)
        self.model = Glm5NextModel(self.config, dtype=dtype, device=device)
        self.lm_head = ParallelLMHead(self.config.vocab_size,
                                      self.config.hidden_size,
                                      bias=False,
                                      dtype=dtype,
                                      device=device)
        if self.config.tie_word_embeddings:
            self.lm_head.tie_weights(self.model.get_input_embeddings())
        self._load_buffers = {}

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[Sequence[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        state_ids: torch.Tensor | None = None,
        kpool_tail_states: list[Sequence[torch.Tensor]] | None = None,
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        vision_groups: list[dict[str, Any]] | None = None,
        vision_prompt_order: list[int] | None = None,
        multimodal_mask: torch.Tensor | None = None,
        return_input_embeds: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is None and (vision_groups or
                                      pixel_values is not None):
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if vision_groups:
                grouped_outputs = []
                for group in vision_groups:
                    group_values = group['pixel_values'].to(
                        dtype=inputs_embeds.dtype)
                    grouped_outputs.append(
                        self.visual(group_values, group['grid_thw']))
                if vision_prompt_order is None:
                    if len(vision_groups) != 1:
                        raise ValueError(
                            'vision_prompt_order is required for multiple '
                            'modality groups.')
                    vision_prompt_order = vision_groups[0]['flat_indices']
                vision_embeddings = self._restore_vision_prompt_order(
                    vision_groups, grouped_outputs, vision_prompt_order)
            else:
                # Keep the legacy single visual-call input contract available
                # to callers that still pass pixel_values/grid_thw directly.
                pixel_values = pixel_values.to(dtype=inputs_embeds.dtype)
                vision_embeddings = self.visual(pixel_values, grid_thw)
            num_slots = int(multimodal_mask.sum().item())
            if num_slots != vision_embeddings.shape[0]:
                raise ValueError(
                    'multimodal token/embedding count mismatch: '
                    f'{num_slots} token slots for '
                    f'{vision_embeddings.shape[0]} embeddings.')
            scatter_mask = multimodal_mask.unsqueeze(-1).expand_as(
                inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                scatter_mask, vision_embeddings.to(inputs_embeds))
        if return_input_embeds and inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        hidden_states = self.model(input_ids=input_ids,
                          position_ids=position_ids,
                          past_key_values=past_key_values,
                          attn_metadata=attn_metadata,
                          inputs_embeds=inputs_embeds,
                          state_ids=state_ids,
                          kpool_tail_states=kpool_tail_states)
        if return_input_embeds:
            return dict(hidden_states=hidden_states,
                        target_inputs_embeds=inputs_embeds)
        return hidden_states

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_input_processor(self):
        """Return the model-specific image/video input processor."""
        return self.input_processor

    @staticmethod
    def _multimodal_token_mask(
        input_ids: torch.Tensor,
        mm_inputs: list[Any],
    ) -> torch.Tensor:
        """Build a mask from the token ids owned by the MM input records."""
        token_ids = set()
        for item in mm_inputs:
            meta = item.meta or {}
            token_id = (meta.get('image_token_id')
                        if item.modality == Modality.IMAGE else
                        meta.get('video_token_id'))
            if token_id is not None:
                token_ids.add(int(token_id))
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in token_ids:
            mask |= input_ids == token_id
        return mask

    @staticmethod
    def _vision_grid(item: Any) -> list[torch.Tensor]:
        """Return image grids, splitting video temporal units like SGLang."""
        grid = torch.as_tensor(item.meta['grid_thw'],
                               dtype=torch.long).reshape(3).cpu()
        if item.modality != Modality.VIDEO:
            return [grid]
        t, h, w = grid.tolist()
        return [torch.tensor([1, h, w], dtype=torch.long) for _ in range(t)]

    def _group_vision_inputs(
        self,
        input_multimodals: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[int], list[Any]]:
        """Pack visual calls by modality while retaining prompt item order."""
        records = []
        for batch_index, batch_item in enumerate(input_multimodals):
            for item in batch_item.get('mm_data', []):
                records.append(
                    dict(batch_index=batch_index,
                         flat_index=len(records),
                         item=item))

        groups = []
        for modality in (Modality.IMAGE, Modality.VIDEO, Modality.AUDIO):
            selected = [
                record for record in records
                if record['item'].modality == modality
            ]
            if not selected:
                continue
            split_sizes = [
                int(record['item'].end - record['item'].start)
                for record in selected
            ]
            if any(size <= 0 for size in split_sizes):
                raise ValueError(
                    'multimodal spans must have positive lengths.')
            groups.append(
                dict(modality=modality,
                     pixel_values=torch.cat(
                         [record['item'].data for record in selected], dim=0),
                     grid_thw=torch.stack([
                         grid for record in selected
                         for grid in self._vision_grid(record['item'])
                     ],
                                          dim=0),
                     flat_indices=[
                         record['flat_index'] for record in selected
                     ],
                     split_sizes=split_sizes))

        prompt_order = [
            record['flat_index']
            for record in sorted(
                records,
                key=lambda record: (record['batch_index'],
                                    record['item'].start,
                                    record['item'].end,
                                    record['flat_index']))
        ]
        return groups, prompt_order, [record['item'] for record in records]

    @staticmethod
    def _restore_vision_prompt_order(
        vision_groups: list[dict[str, Any]],
        grouped_outputs: list[torch.Tensor],
        prompt_order: list[int],
    ) -> torch.Tensor:
        """Split modality outputs per item and concatenate by prompt span."""
        if len(vision_groups) != len(grouped_outputs):
            raise ValueError(
                'one visual output tensor is required per modality group.')

        chunks_by_flat_index = {}
        for group, output in zip(vision_groups, grouped_outputs):
            split_sizes = group['split_sizes']
            expected_rows = sum(split_sizes)
            if output.shape[0] != expected_rows:
                modality = group['modality']
                raise ValueError(
                    f'{modality.value} visual output has {output.shape[0]} '
                    f'rows; expected {expected_rows} from prompt spans.')
            chunks = torch.split(output, split_sizes, dim=0)
            for flat_index, chunk in zip(group['flat_indices'], chunks):
                if flat_index in chunks_by_flat_index:
                    raise ValueError(
                        f'duplicate visual flat index: {flat_index}.')
                chunks_by_flat_index[flat_index] = chunk

        if set(chunks_by_flat_index) != set(prompt_order):
            raise ValueError(
                'visual items and prompt-order items do not match.')
        return torch.cat(
            [chunks_by_flat_index[index] for index in prompt_order], dim=0)

    def prepare_inputs_for_generation(
        self,
        past_key_values: list[Sequence[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        context: StepContext | None = None,
    ):
        named_states = context.named_state_caches
        required_states = (
            GLM5_KDA_CONV_STATE,
            GLM5_KDA_RECURRENT_STATE,
            GLM5_KPOOL_TAIL_K_STATE,
            GLM5_KPOOL_TAIL_SCORE_STATE,
        )
        if named_states is None:
            raise RuntimeError('GLM-5.3 requires named state caches.')
        missing = [name for name in required_states if name not in named_states]
        if missing:
            raise RuntimeError(
                f'GLM-5.3 is missing named state caches: {missing}.')

        # These specs are deliberately unlayered: their leading dimension is
        # the compact KDA/full-attention row.  Runtime storage leads with the
        # request-state slot, so transpose once into layer-major views.
        kda_conv = named_states[GLM5_KDA_CONV_STATE].transpose(0, 1)
        kda_recurrent = named_states[
            GLM5_KDA_RECURRENT_STATE].transpose(0, 1)
        tail_k = named_states[GLM5_KPOOL_TAIL_K_STATE].transpose(0, 1)
        tail_score = named_states[
            GLM5_KPOOL_TAIL_SCORE_STATE].transpose(0, 1)
        linear_caches = list(zip(kda_conv, kda_recurrent))
        kpool_tail_states = list(zip(tail_k, tail_score))
        full_caches = list(past_key_values)
        interleaved_caches = []
        for layer_idx in range(self.config.num_hidden_layers):
            if is_glm5_kda_layer(self.config, layer_idx):
                interleaved_caches.append(linear_caches.pop(0))
            else:
                interleaved_caches.append(full_caches.pop(0))
        if linear_caches or full_caches:
            raise RuntimeError(
                'GLM-5.3 cache counts do not match its hybrid layer map.')

        input_ids = context.input_ids
        pixel_values = None
        grid_thw = None
        vision_groups = None
        vision_prompt_order = None
        multimodal_mask = None
        if context.input_multimodals is not None:
            (vision_groups, vision_prompt_order,
             mm_inputs) = self._group_vision_inputs(
                 context.input_multimodals)
            if vision_groups:
                multimodal_mask = self._multimodal_token_mask(
                    input_ids, mm_inputs)

        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:,
                          vision_embedding_indexing, :] = vision_embeddings.to(
                              inputs_embeds)

        return dict(input_ids=input_ids,
                    position_ids=context.position_ids,
                    past_key_values=interleaved_caches,
                    attn_metadata=context.attn_metadata,
                    inputs_embeds=inputs_embeds,
                    state_ids=context.state_offsets,
                    kpool_tail_states=kpool_tail_states,
                    pixel_values=pixel_values,
                    grid_thw=grid_thw,
                    vision_groups=vision_groups,
                    vision_prompt_order=vision_prompt_order,
                    multimodal_mask=multimodal_mask,
                    return_input_embeds=(
                        self.ctx_mgr.build_ctx.num_spec_tokens > 0
                        and not context.is_decoding))

    @staticmethod
    def _layer_idx(name: str) -> int | None:
        match = re.search(r'\.layers\.(\d+)\.', name)
        return None if match is None else int(match.group(1))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]], *,
                     is_mtp: bool = False):
        """Load both towers through LMDeploy's TP-aware weight loaders."""
        stacked_params_mapping = [
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]
        kda_params_mapping = [
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
            ('.qkv_conv1d', '.q_conv1d', 'q'),
            ('.qkv_conv1d', '.k_conv1d', 'k'),
            ('.qkv_conv1d', '.v_conv1d', 'v'),
        ]
        expert_params_mapping = []
        for expert_id in range(self.config.n_routed_experts):
            expert_params_mapping.extend([
                ('.experts.gate_up', f'.experts.{expert_id}.gate_proj',
                 expert_id, 'gate'),
                ('.experts.gate_up', f'.experts.{expert_id}.up_proj',
                 expert_id, 'up'),
                ('.experts.down', f'.experts.{expert_id}.down_proj', expert_id,
                 'down'),
            ])

        params_dict = dict(self.named_parameters())
        for checkpoint_name, loaded_weight in weights:
            is_visual_weight = checkpoint_name.startswith('model.visual.')
            if checkpoint_name.startswith('model.visual.'):
                if getattr(self.visual, '_is_dummy_mod', False):
                    continue
                name = checkpoint_name.replace('model.visual.', 'visual.', 1)
            else:
                name = checkpoint_name.replace('model.language_model.',
                                               'model.', 1)
            layer_idx = self._layer_idx(name)
            if is_mtp:
                if layer_idx != self.config.num_hidden_layers:
                    continue
                name = self._rewrite_spec_layer_name(layer_idx, name)
            elif layer_idx is not None and layer_idx >= self.config.num_hidden_layers:
                # Predictor weights are loaded by Glm5NextMTPModel.
                continue
            if 'rotary_emb.' in name:
                continue
            if self.config.tie_word_embeddings and name == 'lm_head.weight':
                continue

            if is_visual_weight and '.attn.qkv.' in name:
                param = params_dict[name]
                query, key, value = param.weight_spliter(loaded_weight)
                load_weight(param, query, shard_id='q')
                load_weight(param, key, shard_id='k')
                load_weight(param, value, shard_id='v')
                continue

            if '.experts.' in name:
                self._load_weight_experts(name, loaded_weight, params_dict,
                                          expert_params_mapping)
                continue

            is_kda = (layer_idx is not None
                      and is_glm5_kda_layer(self.config, layer_idx)
                      and '.self_attn.' in name)
            if is_kda:
                for param_name, weight_name, shard_id in kda_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    load_weight(params_dict[name],
                                loaded_weight,
                                shard_id=shard_id)
                    break
                else:
                    load_weight(params_dict[name], loaded_weight)
                continue

            if layer_idx is not None and '.self_attn.' in name:
                if '.self_attn.indexer.' in name:
                    # KPool projections are replicated (not TP-sharded), and
                    # their seven checkpoint tensors map one-to-one.
                    load_weight(params_dict[name], loaded_weight)
                    continue
                self._load_weight_attention(name,
                                            loaded_weight,
                                            params_dict,
                                            update_pe_mapping=[])
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                load_weight(params_dict[name],
                            loaded_weight,
                            shard_id=shard_id)
                break
            else:
                load_weight(params_dict[name], loaded_weight)


class Glm5NextMTPAttention(Glm5NextSparseAttention):
    """The predictor's KPool tail is reconstructed from pageable token data.

    Draft forwards can revisit accepted positions after multiple proposals.
    Keeping raw index keys/scores in its cache avoids private mutable request
    state and reuses the normal cache allocation, sizing and sleep lifecycle.
    Only the single MTP layer requests this additional cache.
    """

    _TOKEN_CACHE = 'glm5_mtp_kpool_tokens'

    def get_block_cache_requests(self, context):
        return (BlockCacheRequest(
            name=self._TOKEN_CACHE,
            shape=(context.geometry.kernel_block_size, 2, self.indexer.head_dim),
            dtype=torch.bfloat16,
            per_row_contiguous=True),)

    def bind_block_cache(self, binding):
        if binding.cache_name != self._TOKEN_CACHE:
            raise ValueError(f'Unexpected MTP token cache: {binding.cache_name}')
        self._token_cache_binding = binding

    def _update_kpool_cache(self, hidden_states, tail_state, state_ids,
                            attn_metadata):
        binding = self._token_cache_binding
        caches = get_step_ctx_manager().current_context().block_caches
        cache = (caches.row(binding.cache_name, binding.consumer_row)
                 if hasattr(caches, 'row') else
                 caches[binding.cache_name][binding.consumer_row])
        block_size = cache.size(1)
        history = (attn_metadata.kv_seqlens - attn_metadata.q_seqlens).long()
        tail_length = history.remainder(self.index_kpool)
        slots = torch.arange(self.index_kpool, device=history.device)
        positions = history[:, None] - tail_length[:, None] + slots
        block_offsets = attn_metadata.block_offsets.long()
        blocks = block_offsets.gather(1, positions.div(block_size, rounding_mode='floor'))
        tails = cache[blocks, positions.remainder(block_size)]
        tails = tails.masked_fill((slots >= tail_length[:, None])[..., None, None], 0)
        state_ids = torch.arange(history.numel(), device=history.device)
        result = super()._update_kpool_cache(
            hidden_states, (tails[:, :, 0].contiguous(), tails[:, :, 1].contiguous()),
            state_ids, attn_metadata)

        # Write raw projected tokens after reading the pre-forward tail.
        # Rejected positions are overwritten on their next visit.
        total_tokens = hidden_states.size(1)
        batch = state_ids.repeat_interleave(attn_metadata.q_seqlens,
                                           output_size=total_tokens)
        token_ids = torch.arange(total_tokens, device=history.device)
        positions = history[batch] + token_ids - attn_metadata.cu_seqlens_q[batch]
        blocks = block_offsets[batch, positions.div(block_size, rounding_mode='floor')]
        key = self.indexer.project_key(hidden_states)[0]
        score = self.indexer.project_compress_score(hidden_states)[0]
        cache[blocks, positions.remainder(block_size)] = torch.stack((key, score), dim=1)
        return result


class Glm5NextMTPDecoderLayer(nn.Module):
    """Checkpoint predictor block: GLM MLA/MoE with plain residuals, no mHC."""

    def __init__(self, config, layer_idx, dtype=None, device=None):
        super().__init__()
        self.self_attn = Glm5NextMTPAttention(config, layer_idx, dtype=dtype, device=device)
        self.mlp = Glm5NextMoE(config, layer_idx, dtype=dtype, device=device)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps,
                                       dtype=dtype, device=device)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps,
                                                dtype=dtype, device=device)

    def forward(self, hidden_states, rotary_pos_emb, past_key_value,
                attn_metadata=None, topk_indices_buffer=None,
                skip_topk=False, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, past_key_value, attn_metadata,
            topk_indices_buffer=topk_indices_buffer, skip_topk=skip_topk)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class Glm5NextMTPModel(GlmMoeDsaMTPModel):
    """Reuse the shared GLM/DeepSeek predictor, proposer and CUDA Graph flow."""

    uses_shared_input_embeddings = True

    def __init__(self, config, ctx_mgr, dtype=None, device=None):
        nn.Module.__init__(self)
        self.config = config.text_config
        self.quantization_config = getattr(config, 'quantization_config', None)
        if self.quantization_config is not None:
            self.config.quantization_config = self.quantization_config
        self.dtype = dtype
        self.ctx_mgr = ctx_mgr
        self.model = GlmMoeDsaMultiTokenPredictor(
            self.config, dtype=dtype, device=device,
            decoder_layer_cls=Glm5NextMTPDecoderLayer)
        self.uses_dsa_topk_buffer = getattr(self.config, 'index_share_for_mtp_iteration', False)
        self.topk_indices_buffer = (
            DSATopKIndicesBuffer(self.config.index_topk + self.config.index_kpool - 1)
            if self.uses_dsa_topk_buffer else None)
        self._load_buffers = {}

    def prepare_inputs_for_generation(self, past_key_values, inputs_embeds=None,
                                      context=None):
        if context.target_inputs_embeds is not None:
            inputs_embeds = context.target_inputs_embeds
        return super().prepare_inputs_for_generation(past_key_values, inputs_embeds, context)

    _layer_idx = staticmethod(Glm5NextForConditionalGeneration._layer_idx)

    def load_weights(self, weights):
        weights = ((name, weight) for name, weight in weights
                   if self._layer_idx(name) == self.config.num_hidden_layers)
        Glm5NextForConditionalGeneration.load_weights(self, weights, is_mtp=True)

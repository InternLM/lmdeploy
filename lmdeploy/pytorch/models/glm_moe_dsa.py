# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from lmdeploy.pytorch import envs as _envs
from lmdeploy.pytorch.model_inputs import StepContextManager, get_step_ctx_manager
from lmdeploy.pytorch.nn import ApplyRotaryEmb
from lmdeploy.pytorch.nn.linear import build_colwise_linear
from lmdeploy.pytorch.nn.nsa import IndexerTopKFP8

from .deepseek_v2 import DeepseekV2MoE
from .deepseek_v32 import (
    DeepseekV32Attention,
    DeepseekV32DecoderLayer,
    DeepseekV32ForCausalLM,
    DeepseekV32Model,
    LayerNorm,
    rotate_activation,
)
from .patch import get_build_model_context


def _get_layer_indexer_type(config: Any, layer_idx: int | None) -> str:
    indexer_types = getattr(config, 'indexer_types', None)
    if indexer_types is None or layer_idx is None or layer_idx >= len(indexer_types):
        return 'full'
    return indexer_types[layer_idx]


def _get_layer_idx_from_weight_name(name: str) -> int | None:
    for marker in ('.layers.', 'layers.'):
        if marker not in name:
            continue
        try:
            return int(name.split(marker, 1)[1].split('.', 1)[0])
        except ValueError:
            return None
    return None


class GlmMoeDsaIndexer(nn.Module):

    def __init__(self, config: Any, layer_idx: int, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        quant_config = getattr(config, 'quantization_config', None)
        self.layer_idx = layer_idx
        self.dim = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.rope_interleave = getattr(config, 'indexer_rope_interleave', False)
        self.index_topk = config.index_topk
        self.wq_b = build_colwise_linear(config.q_lora_rank,
                                         self.n_heads * self.head_dim,
                                         bias=False,
                                         dtype=dtype,
                                         device=device,
                                         is_tp=False,
                                         quant_config=quant_config)
        self.use_fusion = not _envs.disable_dsa_indexer_fusion
        if self.use_fusion:
            self.wk_weights_proj = build_colwise_linear(self.dim,
                                                        self.head_dim + self.n_heads,
                                                        bias=False,
                                                        dtype=torch.bfloat16,
                                                        device=device,
                                                        is_tp=False)
        else:
            self.wk = build_colwise_linear(self.dim,
                                           self.head_dim,
                                           bias=False,
                                           dtype=dtype,
                                           device=device,
                                           is_tp=False,
                                           quant_config=quant_config)
            self.weights_proj = build_colwise_linear(self.dim,
                                                     self.n_heads,
                                                     bias=False,
                                                     dtype=dtype,
                                                     device=device,
                                                     is_tp=False)
        self.k_norm = LayerNorm(self.head_dim, device=device)
        self.softmax_scale = self.head_dim**-0.5
        self.apply_rotary_pos_emb = ApplyRotaryEmb()
        self.indexer_topk = IndexerTopKFP8(self.index_topk,
                                           self.softmax_scale,
                                           self.head_dim,
                                           block_size=128,
                                           fill=-1,
                                           # MTP may reuse its first iteration's indices in later drafts.
                                           allow_short_prefill_scoring_skip=layer_idx
                                           < config.num_hidden_layers)

    def _apply_rotary_pos_emb(self, q_pe: torch.Tensor, k_pe: torch.Tensor,
                              freqs_cis: tuple[torch.Tensor, torch.Tensor]):
        cos, sin = freqs_cis
        if self.rope_interleave:
            half_size = cos.size(-1) // 2
            cos = cos[..., :half_size]
            sin = sin[..., :half_size]
        return self.apply_rotary_pos_emb(q_pe,
                                         k_pe[..., None, :],
                                         cos,
                                         sin,
                                         inplace=False,
                                         complex_mode=self.rope_interleave)

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: Any = None,
    ):
        q = self.wq_b(qr).unflatten(-1, (-1, self.head_dim))
        if self.use_fusion:
            kw = self.wk_weights_proj(x)
            k, weights = kw.split([self.head_dim, self.n_heads], dim=-1)
            cos, sin = freqs_cis
            return self.indexer_topk.forward_fused(q[0],
                                                   k[0],
                                                   weights[0],
                                                   self.k_norm.weight,
                                                   self.k_norm.bias,
                                                   cos,
                                                   sin,
                                                   norm_eps=self.k_norm.eps,
                                                   head_gate_scale=self.n_heads**-0.5,
                                                   rope_interleaved=self.rope_interleave,
                                                   attn_metadata=attn_metadata)

        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k = self.k_norm(self.wk(x))
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        q_pe, k_pe = self._apply_rotary_pos_emb(q_pe, k_pe, freqs_cis)
        q = rotate_activation(torch.cat([q_pe, q_nope], dim=-1))
        k = rotate_activation(torch.cat([k_pe[0], k_nope[0, :, None]], dim=-1))
        weights = self.weights_proj(x) * self.n_heads**-0.5
        return self.indexer_topk(q[0],
                                 k[:, 0],
                                 weights[0],
                                 attn_metadata=attn_metadata)


class DSATopKIndicesBuffer(nn.Module):
    """Share current-forward top-k indices between full and reuse layers."""

    def __init__(self, topk: int):
        super().__init__()
        self.topk = topk
        # None means no full indexer has written the current buffer yet.
        self._has_indices: bool | None = None
        self.register_buffer('indices', None, persistent=False)

    def _target_capacity(self, num_tokens: int) -> int:
        capacity = num_tokens
        ctx_mgr = get_step_ctx_manager()
        if ctx_mgr is None:
            return capacity

        context = ctx_mgr.current_context()
        cache_config = getattr(context, 'cache_config', None)
        max_prefill_token_num = getattr(cache_config, 'max_prefill_token_num', None)
        if max_prefill_token_num is not None:
            capacity = max(capacity, max_prefill_token_num)
        return capacity

    def ensure(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        """Return a stable top-k slice with enough capacity."""
        capacity = self._target_capacity(num_tokens)
        if (self.indices is None or self.indices.size(0) < capacity or self.indices.device != device):
            self.indices = torch.empty(capacity, self.topk, dtype=torch.int32, device=device)
        return self.indices[:num_tokens]

    def write(self, topk_indices: torch.Tensor | None) -> torch.Tensor | None:
        """Store indices, or mark a dense prefill as index-free."""
        self._has_indices = topk_indices is not None
        if topk_indices is None:
            return None
        buffer = self.ensure(topk_indices.size(0), topk_indices.device)
        buffer.copy_(topk_indices)
        return buffer

    def read(self, num_tokens: int, device: torch.device) -> torch.Tensor | None:
        """Read the current indices for a shared indexer layer."""
        if self._has_indices is False:
            return None
        if self.indices is None or self.indices.size(0) < num_tokens or self.indices.device != device:
            raise RuntimeError('DSA top-k indices are reused before the shared buffer is populated.')
        return self.indices[:num_tokens]

    def compact(self, row_indices: torch.Tensor) -> torch.Tensor:
        """Copy selected rows to the prefix for recurrent MTP reuse."""
        selected = self.indices.index_select(0, row_indices)
        self.indices[:selected.size(0)].copy_(selected)
        return self.indices[:selected.size(0)]


class GlmMoeDsaAttention(DeepseekV32Attention):

    def _build_indexer(self, config: Any, layer_idx: int, dtype: torch.dtype, device: torch.device):
        self.indexer_type = _get_layer_indexer_type(config, layer_idx)
        if self.indexer_type == 'shared':
            return None
        return GlmMoeDsaIndexer(config, layer_idx, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Sequence[torch.Tensor] = None,
        attn_metadata: Any = None,
        topk_indices_buffer: DSATopKIndicesBuffer | None = None,
        skip_topk: bool = False,
    ):
        num_heads = self.attn_fwd.num_heads
        nope_size = self.kv_lora_rank
        q_len = hidden_states.size(1)

        query_states, key_states, value_states, q_pe, k_pe, qr = self._qkv_proj(hidden_states,
                                                                                num_heads=num_heads)
        cos, sin = rotary_pos_emb
        q_pe, k_pe = self.apply_rotary_pos_emb(q_pe, k_pe, cos, sin, inplace=False)
        query_states[..., nope_size:] = q_pe
        key_states[..., nope_size:] = k_pe

        if topk_indices_buffer is None:
            raise RuntimeError(f'Layer {self.layer_idx} requires a DSA top-k indices buffer.')
        if self.indexer is not None and not skip_topk:
            topk_indices = topk_indices_buffer.write(
                self.indexer(hidden_states,
                             qr,
                             rotary_pos_emb,
                             attn_metadata=attn_metadata))
        else:
            topk_indices = topk_indices_buffer.read(q_len, hidden_states.device)

        attn_output = self.attn_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[0][..., :nope_size],
            attn_metadata,
            k_scales_zeros=None if len(past_key_value) == 2 else past_key_value[2],
            v_scales_zeros=None if len(past_key_value) == 2 else past_key_value[3],
            nsa_indices=topk_indices,
        )
        attn_bmm_out = attn_output.new_empty(q_len, num_heads, self.v_head_dim)
        self.vc(attn_output, attn_bmm_out)
        return self.o_proj(attn_bmm_out.flatten(-2, -1)[None])


class GlmMoeDsaDecoderLayer(DeepseekV32DecoderLayer):
    attention_cls = GlmMoeDsaAttention

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: list[torch.FloatTensor] | None,
        residual: torch.Tensor | None = None,
        attn_metadata: Any = None,
        topk_indices_buffer: DSATopKIndicesBuffer | None = None,
        skip_topk: bool = False,
        all_routed_experts: torch.Tensor | None = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(hidden_states=hidden_states,
                                       rotary_pos_emb=rotary_pos_emb,
                                       past_key_value=past_key_value,
                                       attn_metadata=attn_metadata,
                                       topk_indices_buffer=topk_indices_buffer,
                                       skip_topk=skip_topk)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if isinstance(self.mlp, DeepseekV2MoE):
            hidden_states = self.mlp(hidden_states, all_routed_experts=all_routed_experts)
        else:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class GlmMoeDsaModel(DeepseekV32Model):
    decoder_layer_cls = GlmMoeDsaDecoderLayer

    def __init__(self, config: Any, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__(config, dtype=dtype, device=device)
        self.topk_indices_buffer = DSATopKIndicesBuffer(config.index_topk)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        attn_metadata: Any = None,
        inputs_embeds: torch.FloatTensor | None = None,
        all_routed_experts: torch.Tensor | None = None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        rotary_pos_emb = (cos[0], sin[0])
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_values[idx],
                residual=residual,
                attn_metadata=attn_metadata,
                topk_indices_buffer=self.topk_indices_buffer,
                all_routed_experts=all_routed_experts,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def forward_microbatch(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        attn_metadata: Any = None,
        inputs_embeds: torch.FloatTensor | None = None,
        all_routed_experts: torch.Tensor | None = None,
    ):
        # Shared top-k indices are model-global, so GLM uses one model forward.
        return self.forward(input_ids=input_ids,
                            position_ids=position_ids,
                            past_key_values=past_key_values,
                            attn_metadata=attn_metadata,
                            inputs_embeds=inputs_embeds,
                            all_routed_experts=all_routed_experts)


class GlmMoeDsaForCausalLM(DeepseekV32ForCausalLM):
    model_cls = GlmMoeDsaModel

    def __init__(self,
                 config: Any,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__(config, ctx_mgr, dtype=dtype, device=device)
        self.enable_return_routed_experts = get_build_model_context().enable_return_routed_experts

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        num_tokens = inputs_embeds.size(1) if inputs_embeds is not None else input_ids.size(1)
        all_routed_experts = None
        if self.enable_return_routed_experts:
            all_routed_experts = position_ids.new_full(
                (num_tokens, self.config.num_hidden_layers, self.config.num_experts_per_tok),
                torch.iinfo(torch.uint16).max,
                dtype=torch.uint16,
            )
        step_ctx = get_step_ctx_manager().current_context()
        forward = self.model.forward_microbatch if step_ctx.enable_microbatch else self.model.forward
        hidden_states = forward(input_ids=input_ids,
                                position_ids=position_ids,
                                past_key_values=past_key_values,
                                attn_metadata=attn_metadata,
                                inputs_embeds=inputs_embeds,
                                all_routed_experts=all_routed_experts)
        if all_routed_experts is None:
            return hidden_states
        return dict(hidden_states=hidden_states, all_routed_experts=all_routed_experts)

    def _load_weight_attention(self, name: str, loaded_weight: torch.Tensor, params_dict: dict[str, nn.Parameter],
                               update_pe_mapping: list):
        if '.self_attn.indexer.' in name and name not in params_dict:
            layer_idx = _get_layer_idx_from_weight_name(name)
            if _get_layer_indexer_type(self.config, layer_idx) == 'shared':
                return
        return super()._load_weight_attention(name, loaded_weight, params_dict, update_pe_mapping)

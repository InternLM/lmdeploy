# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import ApplyRotaryEmb, Attention, RMSNorm, build_rotary_embedding_from_config
from lmdeploy.pytorch.nn.linear import (
    build_colwise_linear,
    build_o_proj,
    build_qkv_proj,
)
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .patch import add_prefix, get_build_model_context
from .qwen3 import Qwen3MLP
from .utils.cudagraph import CudaGraphMixin

_SLIDING_ATTENTION = 'sliding_attention'


def _normalize_dflash_weight_name(name: str) -> str | None:
    """Normalize common DFlash checkpoint weight names to this module
    layout."""
    if 'rotary_emb.inv_freq' in name:
        return None
    if 'rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name:
        return None
    if name.startswith('model.'):
        name = name[len('model.'):]
    if name.startswith('midlayer.'):
        name = name.replace('midlayer.', 'layers.0.', 1)
    return name


def _resolve_dflash_layer_attention(config: PretrainedConfig, layer_idx: int) -> tuple[int | None, bool]:
    """Consume the attention policy already validated by the outer parser."""
    layer_type = config.layer_types[layer_idx]
    sliding_window = config.sliding_window if layer_type == _SLIDING_ATTENTION else None
    default_causal = layer_type == _SLIDING_ATTENTION
    causal_override = config.dflash_config.get('causal', getattr(config, 'causal', None))
    causal = default_causal if causal_override is None else bool(causal_override)
    return sliding_window, causal


class DFlashQwen3Attention(nn.Module):
    """Qwen3 attention used by DFlash draft layers."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None,
                 prefix: str = ''):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        num_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        hidden_size = config.hidden_size
        head_dim = getattr(config, 'head_dim', hidden_size // num_heads)
        num_replicate_kv_heads = getattr(config, 'num_replicate_key_value_heads', 1)
        sliding_window, causal = _resolve_dflash_layer_attention(config, layer_idx)
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.causal = causal
        self.sliding_window = sliding_window

        self.qkv_proj = build_qkv_proj(
            hidden_size,
            num_q_heads=num_heads,
            num_kv_heads=num_key_value_heads,
            head_size=head_dim,
            bias=getattr(config, 'attention_bias', False),
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            num_replicate_kv_heads=num_replicate_kv_heads,
            prefix=add_prefix('qkv_proj', prefix),
        )
        self.apply_rotary_pos_emb = ApplyRotaryEmb()
        self.attn_fwd = Attention(
            num_heads,
            head_dim,
            num_kv_heads=num_key_value_heads,
            v_head_size=head_dim,
            sliding_window=sliding_window,
            causal=causal,
        )
        self.o_proj = build_o_proj(
            num_heads * head_dim,
            hidden_size,
            bias=getattr(config, 'attention_bias', False),
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            is_tp=True,
            prefix=add_prefix('o_proj', prefix),
        )
        self.q_norm = RMSNorm(
            head_dim,
            config.rms_norm_eps,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('q_norm', prefix),
        )
        self.k_norm = RMSNorm(
            head_dim,
            config.rms_norm_eps,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('k_norm', prefix),
        )

    def _split_qkv(self, hidden_states: torch.Tensor):
        """Project and split Q/K/V states."""
        qkv_states = self.qkv_proj(hidden_states)
        qkv_states = qkv_states.flatten(0, -2)
        return self.qkv_proj.split_qkv(qkv_states)

    def kv_proj_only(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project hidden states to K/V for DFlash context materialization."""
        _, key_states, value_states = self._split_qkv(hidden_states)
        return key_states, value_states

    def apply_k_norm(self, key_states: torch.Tensor) -> torch.Tensor:
        """Apply per-head K RMSNorm."""
        return self.k_norm(key_states)

    def apply_k_rope(self, key_states: torch.Tensor,
                     rotary_pos_emb: tuple[torch.FloatTensor, torch.FloatTensor]) -> torch.Tensor:
        """Apply RoPE to K states using a disposable Q tensor."""
        cos, sin = rotary_pos_emb
        dummy_query = torch.empty_like(key_states)
        _, key_states = self.apply_rotary_pos_emb(dummy_query, key_states, cos, sin, inplace=True)
        return key_states

    def project_context_kv(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: tuple[torch.FloatTensor, torch.FloatTensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return projected, normalized, RoPE'd context K/V states."""
        key_states, value_states = self.kv_proj_only(hidden_states)
        key_states = self.apply_k_norm(key_states)
        key_states = self.apply_k_rope(key_states, rotary_pos_emb)
        return key_states, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: tuple[torch.Tensor],
        attn_metadata: Any,
    ):
        """Forward a DFlash query block."""
        query_states, key_states, value_states = self._split_qkv(hidden_states)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = rotary_pos_emb
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            inplace=True,
        )

        attn_output = self.attn_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            attn_metadata,
            k_scales_zeros=None if len(past_key_value) == 2 else past_key_value[2],
            v_scales_zeros=None if len(past_key_value) == 2 else past_key_value[3],
            inplace=True,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)
        return self.o_proj(attn_output)


class DFlashQwen3DecoderLayer(nn.Module):
    """Decoder layer for standard Qwen-family DFlash draft checkpoints."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None,
                 prefix: str = ''):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)
        self.self_attn = DFlashQwen3Attention(
            config,
            layer_idx,
            dtype=dtype,
            device=device,
            prefix=add_prefix('self_attn', prefix),
        )
        self.mlp = Qwen3MLP(config, dtype=dtype, device=device, prefix=add_prefix('mlp', prefix))
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('input_layernorm', prefix),
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('post_attention_layernorm', prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: list[torch.FloatTensor] | None,
        attn_metadata: Any,
        residual: torch.Tensor | None = None,
    ):
        """Forward one decoder layer."""
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

class DFlashDraftModel(nn.Module, CudaGraphMixin):
    """Qwen-family DFlash draft model.

    z-lab DFlash draft checkpoints carry only the draft transformer and feature-fusion weights. Embeddings and logits
    are shared from the target model by the proposer.
    """

    packed_modules_mapping = {
        'qkv_proj': [
            'q_proj',
            'k_proj',
            'v_proj',
        ],
        'gate_up_proj': [
            'gate_proj',
            'up_proj',
        ],
    }

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None,
                 prefix: str = ''):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        self.dtype = dtype
        build_ctx = get_build_model_context()
        self.target_layer_ids = build_ctx.spec_model_ctx.target_aux_hidden_state_layers
        if not self.target_layer_ids:
            raise ValueError('DFlash draft construction requires resolved target_aux_hidden_state_layers metadata.')
        self.mask_token_id = build_ctx.spec_model_ctx.speculative_mask_token_id
        if self.mask_token_id is None:
            raise ValueError('DFlash draft construction requires resolved speculative_mask_token_id metadata.')
        self.num_context_features = len(self.target_layer_ids)
        target_hidden_size = int(getattr(config, 'target_hidden_size', config.hidden_size))
        fc_input_size = target_hidden_size * self.num_context_features
        quantization_config = getattr(config, 'quantization_config', None)

        self.embed_tokens = None
        self.mask_embedding = nn.Parameter(
            torch.zeros(config.hidden_size, dtype=dtype or torch.float16, device=device),
            requires_grad=False,
        )
        self.has_separate_mask_embedding = False

        self.layers = nn.ModuleList([
            DFlashQwen3DecoderLayer(config,
                                    layer_idx,
                                    dtype=dtype,
                                    device=device,
                                    prefix=add_prefix(f'layers.{layer_idx}', prefix))
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.fc = build_colwise_linear(
            fc_input_size,
            config.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=False,
            quant_config=quantization_config,
            check_dist=False,
            prefix=add_prefix('fc', prefix),
        )
        self.hidden_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('hidden_norm', prefix),
        )
        self.norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('norm', prefix),
        )
        self.rotary_emb = build_rotary_embedding_from_config(config, device=device)

    def set_input_embeddings(self, embed_tokens: nn.Module):
        """Set target-shared token embeddings."""
        self.embed_tokens = embed_tokens

    def get_input_embeddings(self):
        """Get target-shared token embeddings."""
        return self.embed_tokens

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed query ids, replacing mask slots when a separate mask vector is
        loaded."""
        if self.embed_tokens is None:
            raise RuntimeError('DFlash draft model requires target-shared input embeddings.')
        embeds = self.embed_tokens(input_ids)
        if self.has_separate_mask_embedding and self.mask_token_id is not None:
            mask = (input_ids == int(self.mask_token_id)).unsqueeze(-1)
            embeds = torch.where(mask, self.mask_embedding.to(dtype=embeds.dtype), embeds)
        return embeds

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Project concatenated target-layer hidden states into draft hidden
        size."""
        expected = int(self.fc.in_features)
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError('DFlash target hidden feature dim mismatch. '
                             f'Expected shape [N, {expected}] from {self.num_context_features} target layers, '
                             f'got {tuple(target_hidden.shape)}.')
        return self.hidden_norm(self.fc(target_hidden))

    def _rotary_pos_emb_for_context(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build rotary embeddings for flattened context states."""
        if position_ids is None:
            raise ValueError('DFlash context KV materialization requires position_ids.')
        if position_ids.dim() == 1:
            rope_position_ids = position_ids.unsqueeze(0)
        else:
            rope_position_ids = position_ids
        cos, sin = self.rotary_emb(hidden_states.unsqueeze(0), rope_position_ids)
        return cos[0], sin[0]

    @torch.inference_mode()
    def precompute_context_kv(
        self,
        target_hidden: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Project target features into per-layer draft context K/V tensors."""
        context_states = self.project_target_hidden(target_hidden)
        rotary_pos_emb = self._rotary_pos_emb_for_context(context_states, position_ids)
        return [layer.self_attn.project_context_kv(context_states, rotary_pos_emb) for layer in self.layers]

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        target_hidden: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[list[torch.Tensor]] | None = None,
        attn_metadata: Any | None = None,
        max_q_seqlen: int | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Project context K/V and optionally write them into the draft KV
        cache.

        When cache arguments are omitted, this returns the materialized K/V tensors for tests, alignment dumps, or
        proposer-side staging.
        """
        per_layer_kv = self.precompute_context_kv(target_hidden, position_ids)
        if past_key_values is None or attn_metadata is None:
            return per_layer_kv
        if max_q_seqlen is None:
            raise ValueError('DFlash draft KV materialization requires CPU max_q_seqlen metadata.')

        from lmdeploy.pytorch.kernels.cuda import fill_kv_cache

        for (key_states, value_states), past_key_value in zip(per_layer_kv, past_key_values, strict=True):
            fill_kv_cache(
                key_states,
                value_states,
                past_key_value[0],
                past_key_value[1],
                attn_metadata.q_start_loc,
                attn_metadata.q_seqlens,
                kv_seq_length=attn_metadata.kv_seqlens,
                max_q_seq_length=int(max_q_seqlen),
                block_offsets=attn_metadata.block_offsets,
                k_scales_zeros=None if len(past_key_value) == 2 else past_key_value[2],
                v_scales_zeros=None if len(past_key_value) == 2 else past_key_value[3],
                quant_policy=attn_metadata.quant_policy,
            )
        return per_layer_kv

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata: Any,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        """Forward a DFlash query block."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)

        hidden_states = inputs_embeds
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        residual = None
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_values[idx],
                residual=residual,
                attn_metadata=attn_metadata,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def prepare_inputs_for_generation(
        self,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        context: StepContext | None = None,
    ):
        """Prepare DFlash draft model inputs."""
        return dict(
            input_ids=context.input_ids,
            position_ids=context.position_ids,
            past_key_values=past_key_values,
            attn_metadata=context.attn_metadata,
            inputs_embeds=inputs_embeds,
        )

    def update_model_metas(
        self,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        context: StepContext | None = None,
    ):
        """Return draft model metadata."""
        return None

    @classmethod
    def rename_weight(cls, name: str) -> str:
        """Rename loaded checkpoint weights."""
        return _normalize_dflash_weight_name(name) or name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load DFlash draft checkpoint weights."""
        stacked_params_mapping = [
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            name = _normalize_dflash_weight_name(name)
            if name is None:
                continue
            if name in ('embed_tokens.weight', 'lm_head.weight'):
                continue
            if name == 'mask_embedding.weight':
                self.mask_embedding.data.copy_(loaded_weight.reshape_as(self.mask_embedding))
                self.has_separate_mask_embedding = True
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, shard_id=shard_id)
                break
            else:
                param = params_dict[name]
                load_weight(param, loaded_weight)

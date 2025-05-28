# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama import LlamaConfig
from transformers.models.mllama.modeling_mllama import MllamaTextConfig, MllamaVisionConfig

from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor, PreprocessInputResult
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.multimodal.data_type import MultiModalTensor
from lmdeploy.pytorch.nn import (ApplyRotaryEmb, Attention, LayerNorm, RMSNorm, RopeType, SiluAndMul,
                                 build_rotary_embedding)
from lmdeploy.pytorch.nn.linear import (build_colwise_linear, build_merged_colwise_linear, build_qkv_proj,
                                        build_rowwise_linear)
from lmdeploy.pytorch.nn.rotary_embedding import Llama3Parameters
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .utils.cudagraph import CudaGraphMeta, CudaGraphMixin, next_power_of_2
from .utils.model import DeployModelMixin

MLLAMA_IMAGE_TOKEN_ID = 128256
MLLAMA_IMAGE_TOKEN = '<|image|>'


def _prepare_aspect_ratio_attention_mask(
    aspect_ratio_mask: torch.Tensor,
    num_patches: int,
    target_length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Expand aspect ratio mask to target_length
    batch_size, max_num_tiles = aspect_ratio_mask.shape
    attention_mask = aspect_ratio_mask.view(batch_size, max_num_tiles, 1, 1).to(dtype)
    attention_mask = attention_mask.repeat(1, 1, target_length, 1)

    # Mask padding patches
    pad_patches = target_length - num_patches
    attention_mask[:, :, -pad_patches:] = 0

    # Invert the mask (0 -> 1, 1 -> 0)
    attention_mask = 1 - attention_mask

    # Reshape to 2D and create 4D attention mask
    # (batch_size, 1, max_num_tiles * target_length,
    # max_num_tiles * target_length)
    attention_mask = attention_mask.reshape(batch_size, max_num_tiles * target_length, 1)
    attention_mask = attention_mask * attention_mask.transpose(-1, -2) * torch.finfo(dtype).min
    attention_mask = attention_mask.unsqueeze(1)

    return attention_mask


class LlamaAttention(nn.Module):
    """Rewrite module of LlamaAttention."""

    def __init__(self, config: LlamaConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        num_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        hidden_size = config.hidden_size
        head_dim = getattr(config, 'head_dim', hidden_size // num_heads)

        # packed qkv
        self.qkv_proj = build_qkv_proj(
            hidden_size,
            num_q_heads=num_heads,
            num_kv_heads=num_key_value_heads,
            head_size=head_dim,
            bias=getattr(config, 'attention_bias', False),
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )

        # rotary embedding
        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention
        self.attn_fwd = Attention(
            num_heads,
            head_dim,
            num_kv_heads=num_key_value_heads,
            v_head_size=head_dim,
        )

        # o_proj
        self.o_proj = build_rowwise_linear(num_heads * head_dim,
                                           hidden_size,
                                           bias=getattr(config, 'attention_bias', False),
                                           quant_config=quantization_config,
                                           dtype=dtype,
                                           device=device,
                                           is_tp=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_metadata: Any = None,
    ):
        """Rewrite of LlamaAttention.forward."""
        # qkv proj
        qkv_states = self.qkv_proj(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        query_states, key_states, value_states = self.qkv_proj.split_qkv(qkv_states)

        # apply rotary embedding
        cos, sin = rotary_pos_emb
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            inplace=True,
        )

        # attention
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

        # o proj
        attn_output = self.o_proj(attn_output)
        return attn_output


class MllamaTextCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(self,
                 config: Optional[MllamaTextConfig] = None,
                 layer_idx: Optional[int] = None,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        quantization_config = getattr(config, 'quantization_config', None)
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // self.num_heads
        self.layer_idx = layer_idx
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # packed qkv
        self.qkv_proj = build_qkv_proj(
            self.hidden_size,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_key_value_heads,
            head_size=self.head_dim,
            bias=getattr(config, 'attention_bias', False),
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )
        self.o_proj = build_rowwise_linear(self.num_heads * self.head_dim,
                                           self.hidden_size,
                                           bias=False,
                                           quant_config=quantization_config,
                                           dtype=dtype,
                                           device=device,
                                           is_tp=True)

        # attention
        self.attn_fwd = Attention(
            self.num_heads,
            self.head_dim,
            num_kv_heads=self.num_key_value_heads,
            v_head_size=self.head_dim,
            causal=False,
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        cross_attention_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        cross_attn_metadata: Any = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel."""
        qkv_states = self.qkv_proj(hidden_states)
        qkv_states = qkv_states.flatten(0, -2)
        query_states, _, _ = self.qkv_proj.split_qkv(qkv_states)
        query_states = query_states.view(-1, query_states.shape[-2], self.head_dim)
        query_states = self.q_norm(query_states)

        if cross_attention_states is not None:
            qkv_states = self.qkv_proj(cross_attention_states)
            qkv_states = qkv_states.flatten(0, -2)
            _, key_states, value_states = self.qkv_proj.split_qkv(qkv_states)
            key_states = key_states.view(-1, key_states.shape[-2], self.head_dim)
            value_states = value_states.view(-1, value_states.shape[-2], self.head_dim)
            key_states = self.k_norm(key_states)
        else:
            key_states = None
            value_states = None

        # attention
        attn_output = self.attn_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            cross_attn_metadata,
            k_scales_zeros=None if len(past_key_value) == 2 else past_key_value[2],
            v_scales_zeros=None if len(past_key_value) == 2 else past_key_value[3],
            inplace=True,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

        # o proj
        attn_output = self.o_proj(attn_output)
        return attn_output


class LlamaMLP(nn.Module):
    """Llama mlp."""

    def __init__(self, config: LlamaConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        # gate up
        self.gate_up_proj = build_merged_colwise_linear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
        )

        # silu and mul
        self.act_fn = SiluAndMul(inplace=True)

        # down
        self.down_proj = build_rowwise_linear(config.intermediate_size,
                                              config.hidden_size,
                                              bias=False,
                                              quant_config=quantization_config,
                                              dtype=dtype,
                                              device=device,
                                              is_tp=True)

    def forward(self, x):
        """forward."""
        gate_up = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        return self.down_proj(act)


class MllamaSelfAttentionDecoderLayer(nn.Module):
    """Llama decoder layer."""

    def __init__(self, config: LlamaConfig, layer_idx: int, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)

        # build attention layer
        self.self_attn = LlamaAttention(config, dtype=dtype, device=device)

        # build MLP
        self.mlp = LlamaMLP(config, dtype=dtype, device=device)

        # build input layer norm
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.rms_norm_eps,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device)

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps,
                                                quant_config=quantization_config,
                                                dtype=dtype,
                                                device=device)

    def forward(self,
                hidden_states: torch.Tensor,
                rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
                past_key_value: Optional[List[torch.FloatTensor]],
                cross_attention_states: Optional[torch.FloatTensor] = None,
                full_text_row_masked_out_mask: Optional[torch.Tensor] = None,
                residual: Optional[torch.Tensor] = None,
                attn_metadata: Any = None,
                cross_attn_metadata: Any = None):

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        outputs = (hidden_states, residual)
        return outputs


class MllamaCrossAttentionDecoderLayer(nn.Module):
    """Llama decoder layer."""

    def __init__(self, config: LlamaConfig, layer_idx: int, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)

        # build attention layer
        self.cross_attn = MllamaTextCrossAttention(config, dtype=dtype, device=device)

        # build MLP
        self.mlp = LlamaMLP(config, dtype=dtype, device=device)

        # build input layer norm
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.rms_norm_eps,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device)

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps,
                                                quant_config=quantization_config,
                                                dtype=dtype,
                                                device=device)
        self.cross_attn_attn_gate = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self.cross_attn_mlp_gate = torch.nn.Parameter(torch.zeros(1, dtype=dtype))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        full_text_row_masked_out_mask: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
        cross_attn_metadata: Any = None,
    ):

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Cross Attention
        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            cross_attention_states=cross_attention_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            cross_attn_metadata=cross_attn_metadata,
        )
        hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states
        residual = hidden_states

        # Fully Connected
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = full_text_row_masked_out_mask * hidden_states
        hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states

        outputs = (hidden_states, None)
        return outputs


class MllamaTextModel(nn.Module):
    """Llama model."""

    def __init__(self, config: LlamaConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size + 8,
                                         config.hidden_size,
                                         self.padding_idx,
                                         dtype=dtype,
                                         device=device)
        self.cross_attention_layers = config.cross_attention_layers

        # build all decode layers
        layers = []
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx in self.cross_attention_layers:
                layers.append(MllamaCrossAttentionDecoderLayer(config, layer_idx, dtype=dtype, device=device))
            else:
                layers.append(MllamaSelfAttentionDecoderLayer(config, layer_idx, dtype=dtype, device=device))
        self.layers = nn.ModuleList(layers)

        # build norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)

        # build rotary embedding in LlamaModel
        rope_dim = config.hidden_size // config.num_attention_heads
        rope_max_pos_emb = config.max_position_embeddings
        rope_base = config.rope_theta
        scaling_factor = 1.0
        llama3_params = None
        rope_scaling = config.rope_scaling
        if rope_scaling is None:
            emb_type = RopeType.LinearScaling
        else:
            if 'scaling_factor' in rope_scaling:
                scaling_factor = rope_scaling['scaling_factor']
            elif 'factor' in rope_scaling:
                scaling_factor = rope_scaling['factor']

            rope_type = rope_scaling['rope_type']
            if rope_type == 'dynamic':
                emb_type = RopeType.DynamicNTKScaling
            elif rope_type == 'linear':
                emb_type = RopeType.LinearScaling
            elif rope_type == 'llama3':
                emb_type = RopeType.Llama3
                low_freq_factor = rope_scaling.get('low_freq_factor', 1.0)
                high_freq_factor = rope_scaling.get('high_freq_factor', 1.0)
                llama3_params = Llama3Parameters(low_freq_factor, high_freq_factor)
            else:
                raise RuntimeError(f'Unsupported rope type: {rope_type}')

        self.rotary_emb = build_rotary_embedding(
            rope_dim,
            rope_max_pos_emb,
            rope_base,
            scaling_factor,
            llama3_params=llama3_params,
            emb_type=emb_type,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        cross_attention_states: Optional[torch.FloatTensor] = None,
        full_text_row_masked_out_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cross_attn_metadata: Any = None,
    ):
        """Rewrite of LlamaModel.forward."""

        # token embedding
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # rotary embedding
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        # decoding
        residual = None
        for idx, decoder_layer in enumerate(self.layers):
            if full_text_row_masked_out_mask is None and idx in self.cross_attention_layers:  # noqa
                continue
            past_key_value = past_key_values[idx]
            hidden_states, residual = decoder_layer(
                hidden_states,
                cross_attention_states=cross_attention_states,
                full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
                cross_attn_metadata=cross_attn_metadata,
            )

        # norm
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.embed_tokens


class MllamaForCausalLM(nn.Module):
    """Llama model."""

    def __init__(self, config: MllamaTextConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.text_config = config.get_text_config()
        self.vocab_size = self.text_config.vocab_size

        self.model = MllamaTextModel(config, dtype=dtype, device=device)
        # build lm_head
        self.lm_head = build_rowwise_linear(config.hidden_size,
                                            config.vocab_size,
                                            bias=False,
                                            dtype=dtype,
                                            device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        cross_attention_states: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        cross_attn_metadata: Any = None,
        **kwargs,
    ):
        """Model forward, return logits."""
        hidden_states = self.model(
            input_ids=input_ids,
            cross_attention_states=cross_attention_states,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            cross_attn_metadata=cross_attn_metadata,
        )
        return hidden_states

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return self.lm_head(hidden_states)


class MllamaPrecomputedPositionEmbedding(nn.Module):
    """Vis position embedding."""

    def __init__(self, config: MllamaVisionConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.config = config
        self.num_patches = (config.image_size // config.patch_size)**2 + 1
        self.hidden_size = config.hidden_size

        self.gate = nn.Parameter(torch.empty(1, dtype=dtype, device=device))

        # position embedding
        self.embedding = nn.Parameter(torch.empty(self.num_patches, self.hidden_size, dtype=dtype, device=device))

        # tile position embedding
        self.tile_embedding = nn.Embedding(self.max_aspect_ratio_id + 1,
                                           self.max_num_tiles * self.num_patches * self.hidden_size,
                                           dtype=dtype,
                                           device=device)

        self._weight_inited = False

    def _init_weight(self):
        """Init weight."""
        if self._weight_inited:
            return

        gate_tanh = self.gate.tanh()
        gated_position_embedding = (1 - gate_tanh) * self.embedding
        self.gate_tanh = gate_tanh
        self.gated_position_embedding = gated_position_embedding.view(1, 1, self.num_patches, self.hidden_size)

        self._weight_inited = True

    def forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        """forward."""
        self._init_weight()

        # position embeddings
        hidden_state = hidden_state + self.gated_position_embedding

        # precomputed tile position embeddings
        tile_position_embedding = self.tile_embedding(aspect_ratio_ids)
        batch_size = hidden_state.shape[0]
        tile_position_embedding = tile_position_embedding.reshape(batch_size, self.max_num_tiles, self.num_patches,
                                                                  self.hidden_size)
        gated_tile_position_embedding = (self.gate_tanh * tile_position_embedding)
        hidden_state = hidden_state + gated_tile_position_embedding

        return hidden_state


class MllamaPrecomputedAspectRatioEmbedding(nn.Module):

    def __init__(self,
                 config: MllamaVisionConfig,
                 is_gated: bool = True,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.is_gated = is_gated

        self.embedding = nn.Embedding(self.max_aspect_ratio_id + 1,
                                      self.max_num_tiles * self.hidden_size,
                                      dtype=dtype,
                                      device=device)
        if is_gated:
            self.gate = nn.Parameter(torch.empty(1, dtype=dtype, device=device))

        self._weight_inited = False

    def _init_weight(self):
        """Init weight."""
        if self._weight_inited:
            return

        gate_tanh = self.gate.tanh()
        self.gate_tanh = gate_tanh

        self._weight_inited = True

    def forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        self._init_weight()
        embeddings = self.embedding(aspect_ratio_ids)
        embeddings = embeddings.reshape(-1, self.max_num_tiles, 1, self.hidden_size)

        if self.is_gated:
            embeddings = embeddings * self.gate_tanh

        hidden_state = hidden_state + embeddings
        return hidden_state


class MllamaVisionAttention(nn.Module):
    """Mllama vision attention."""

    def __init__(self, config: MllamaVisionConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        self.embed_dim = config.hidden_size
        self.num_heads = config.attention_heads
        self.head_dim = config.hidden_size // config.attention_heads

        # packed qkv
        self.qkv_proj = build_qkv_proj(
            self.embed_dim,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_size=self.head_dim,
            bias=False,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )

        # o_proj
        self.o_proj = build_rowwise_linear(self.num_heads * self.head_dim,
                                           self.embed_dim,
                                           bias=False,
                                           quant_config=quantization_config,
                                           dtype=dtype,
                                           device=device,
                                           is_tp=True)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_state.size(0)
        qkv_states = self.qkv_proj(hidden_state)
        qkv_states = qkv_states.flatten(0, -2)
        query, key, value = self.qkv_proj.split_qkv(qkv_states)

        query = query.unflatten(0, (batch_size, -1))
        key = key.unflatten(0, (batch_size, -1))
        value = value.unflatten(0, (batch_size, -1))
        q_seq_len = query.shape[1]

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_seq_len, -1)

        output = self.o_proj(attn_output)

        return output


class MllamaVisionMLP(nn.Module):
    """Mllama vision mlp."""

    def __init__(self, config: MllamaVisionConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        from transformers.activations import ACT2FN
        self.config = config
        quantization_config = getattr(config, 'quantization_config', None)
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = build_colwise_linear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
        )
        self.fc2 = build_rowwise_linear(config.intermediate_size,
                                        config.hidden_size,
                                        bias=True,
                                        quant_config=quantization_config,
                                        dtype=dtype,
                                        device=device,
                                        is_tp=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class MllamaVisionEncoderLayer(nn.Module):
    """Vision encoder layer."""

    def __init__(self,
                 config: MllamaVisionConfig,
                 is_gated: bool,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.is_gated = is_gated
        self.self_attn = MllamaVisionAttention(config, dtype=dtype, device=device)
        self.mlp = MllamaVisionMLP(config, dtype=dtype, device=device)

        self.input_layernorm = LayerNorm(self.hidden_size, eps=config.norm_eps, dtype=dtype, device=device)
        self.post_attention_layernorm = LayerNorm(self.hidden_size, eps=config.norm_eps, dtype=dtype, device=device)

        if is_gated:
            self.gate_attn = nn.Parameter(torch.empty(1, dtype=dtype, device=device))
            self.gate_ffn = nn.Parameter(torch.empty(1, dtype=dtype, device=device))

        self._weight_inited = not is_gated

    def _init_weight(self):
        """Init weight."""
        if self._weight_inited:
            return

        self.gate_attn_tanh = self.gate_attn.tanh()
        self.gate_ffn_tanh = self.gate_ffn.tanh()

        self._weight_inited = True

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """forward."""
        self._init_weight()

        # Self Attention
        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)
        hidden_state = self.self_attn(hidden_state, attention_mask=attention_mask)
        if self.is_gated:
            hidden_state = self.gate_attn_tanh * hidden_state
        hidden_state = residual + hidden_state

        # Feed forward
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        if self.is_gated:
            hidden_state = self.gate_ffn_tanh * hidden_state
        hidden_state = residual + hidden_state

        outputs = hidden_state

        return outputs


class MllamaVisionEncoder(nn.Module):
    """Vision encoder."""

    def __init__(self,
                 config: MllamaVisionConfig,
                 num_layers=32,
                 is_gated=False,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [MllamaVisionEncoderLayer(config, is_gated, dtype=dtype, device=device) for _ in range(num_layers)])
        self.gradient_checkpointing = False
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """forward."""
        encoder_states = ()
        for encoder_layer in self.layers:
            encoder_states = encoder_states + (hidden_states, )
            hidden_states = encoder_layer(
                hidden_state=hidden_states,
                attention_mask=attention_mask,
            )
        encoder_states = encoder_states + (hidden_states, )

        return hidden_states, encoder_states


class MllamaVisionModel(nn.Module):
    """Vision model."""

    def __init__(self, config: MllamaVisionConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()

        self.config = config
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.intermediate_layers_indices = config.intermediate_layers_indices
        self.dtype = dtype

        self.num_patches = (self.image_size // self.patch_size)**2 + 1
        self.scale = config.hidden_size**-0.5

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding='valid',
            bias=False,
            dtype=dtype,
            device=device,
        )

        self.class_embedding = nn.Parameter(torch.empty(self.hidden_size, dtype=dtype, device=device))
        self.gated_positional_embedding = MllamaPrecomputedPositionEmbedding(
            config,
            dtype=dtype,
            device=device,
        )

        self.pre_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(  # noqa: E501
            config,
            is_gated=True,
            dtype=dtype,
            device=device,
        )
        self.post_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(  # noqa: E501
            config,
            is_gated=True,
            dtype=dtype,
            device=device,
        )

        # layer norms
        self.layernorm_pre = nn.LayerNorm(
            self.hidden_size,
            dtype=dtype,
            device=device,
        )
        self.layernorm_post = nn.LayerNorm(
            self.hidden_size,
            dtype=dtype,
            device=device,
        )

        # encoders
        self.transformer = MllamaVisionEncoder(
            config,
            config.num_hidden_layers,
            is_gated=False,
            dtype=dtype,
            device=device,
        )
        self.global_transformer = MllamaVisionEncoder(
            config,
            config.num_global_layers,
            is_gated=True,
            dtype=dtype,
            device=device,
        )

    def apply_class_embedding(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size, _, hidden_size = hidden_state.shape
        class_embedding = self.class_embedding.expand(batch_size, 1, hidden_size)
        hidden_state = torch.cat([class_embedding, hidden_state], dim=1)
        return hidden_state

    def forward(
        self,
        pixel_values: torch.Tensor,
        aspect_ratio_ids: torch.Tensor,
        aspect_ratio_mask: torch.Tensor,
    ):
        """forward."""
        (batch_size, num_concurrent_media, num_tiles, num_channels, height, width) = pixel_values.shape

        pixel_values = pixel_values.reshape(batch_size * num_concurrent_media * num_tiles, num_channels, height, width)
        aspect_ratio_ids = aspect_ratio_ids.reshape(batch_size * num_concurrent_media, -1)

        # Patch embedding
        patch_embeds = self.patch_embedding(pixel_values.to(self.dtype))
        hidden_state = patch_embeds.flatten(2).transpose(1, 2)

        # Tile embeddings
        _, num_patches, dim = hidden_state.shape
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, -1, dim)
        hidden_state = self.pre_tile_positional_embedding(hidden_state, aspect_ratio_ids)

        # Add cls token
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media * num_tiles, num_patches, dim)
        hidden_state = self.apply_class_embedding(hidden_state)
        num_patches += 1

        # Position embeddings
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, num_patches, dim)
        hidden_state = self.gated_positional_embedding(hidden_state, aspect_ratio_ids)

        hidden_state = self.layernorm_pre(hidden_state)

        # Compute the number of tokens to pad
        num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8
        # Compute padding tuple for pad function
        padding = (0, 0, 0, num_padding_patches)  # (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
        # Pad the tensor
        hidden_state = F.pad(hidden_state, padding, mode='constant', value=0)
        slice_index = -num_padding_patches if num_padding_patches > 0 else None

        # Prepare attention mask
        attention_mask = aspect_ratio_mask.reshape(batch_size * num_concurrent_media, -1)
        attention_mask = _prepare_aspect_ratio_attention_mask(
            aspect_ratio_mask=attention_mask,
            num_patches=self.num_patches,
            target_length=hidden_state.shape[2],
            dtype=self.dtype,
        )

        # Apply encoder
        hidden_state = hidden_state.view(batch_size * num_concurrent_media, -1, dim)
        output = self.transformer(
            hidden_state,
            attention_mask=attention_mask,
        )
        hidden_state = output[0]

        hidden_state = self.layernorm_post(hidden_state)

        # Apply global encoder
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles,
                                            num_patches + num_padding_patches, dim)
        hidden_state = self.post_tile_positional_embedding(hidden_state, aspect_ratio_ids)
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media,
                                            num_tiles * (num_patches + num_padding_patches), dim)
        global_output = self.global_transformer(
            hidden_state,
            attention_mask=attention_mask,
        )
        hidden_state = global_output[0]

        # Remove padding form hidden state
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles,
                                            num_patches + num_padding_patches, dim)
        hidden_state = hidden_state[:, :, :slice_index]
        hidden_state = hidden_state.reshape(batch_size, num_concurrent_media, num_tiles, num_patches, dim)

        # Collect intermediate layer outputs from encoder output
        all_intermediate_hidden_states = output[1]
        all_intermediate_hidden_states = [all_intermediate_hidden_states[i] for i in self.intermediate_layers_indices]
        intermediate_hidden_states = torch.stack(all_intermediate_hidden_states, dim=-1)

        # Remove padding from intermediate hidden states
        intermediate_hidden_states = intermediate_hidden_states.reshape(batch_size * num_concurrent_media, num_tiles,
                                                                        num_patches + num_padding_patches, -1)
        intermediate_hidden_states = intermediate_hidden_states[:, :, :slice_index]
        intermediate_hidden_states = intermediate_hidden_states.reshape(batch_size, num_concurrent_media, num_tiles,
                                                                        num_patches, -1)

        # Concatenate final hidden state and intermediate hidden states
        hidden_state = torch.cat([hidden_state, intermediate_hidden_states], dim=-1)

        return hidden_state


class MllamaForConditionalGeneration(nn.Module, CudaGraphMixin, DeployModelMixin):
    """Rewrote model of MllamaForConditionalGeneration."""

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
                 config: LlamaConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr

        self.vision_model = MllamaVisionModel(
            config.vision_config,
            dtype=dtype,
            device=device,
        )
        # build MllamaForCausalLM
        self.language_model = MllamaForCausalLM(config.text_config, dtype=dtype, device=device)

        self.multi_modal_projector = build_rowwise_linear(
            config.vision_config.vision_output_dim,
            config.text_config.hidden_size,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.dtype = dtype

        # preprocessor
        self.input_processor = MLlamaInputProcessor(self.config, dtype)

    def flat_encoder_result(self, attn_metadata: Any, input_ids: torch.LongTensor):
        # since every state share the same shape
        full_text_row_masked_out_mask = torch.ones((attn_metadata.q_seqlens.sum(), 1), dtype=torch.bool)
        start_pos = 0
        img_idx = torch.where(input_ids == MLLAMA_IMAGE_TOKEN_ID)[1]
        for img_id, q_seq_len in zip(img_idx.cpu(), attn_metadata.q_seqlens.cpu()):
            full_text_row_masked_out_mask[start_pos:img_id] = False
            start_pos += q_seq_len
        full_text_row_masked_out_mask = full_text_row_masked_out_mask.to(input_ids.device)

        return full_text_row_masked_out_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        pixel_values: torch.Tensor = None,
        aspect_ratio_ids: torch.Tensor = None,
        aspect_ratio_mask: torch.Tensor = None,
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        cross_attn_metadata: Any = None,
        **kwargs,
    ):
        """Model forward, return logits."""

        if cross_attn_metadata is None:
            full_text_row_masked_out_mask = None
        # FIXME basically, we want to inference
        # text requests and image requests separately
        elif pixel_values is None and (cross_attn_metadata.kv_seqlens is None):
            full_text_row_masked_out_mask = None
        elif cross_attn_metadata.is_decoding:
            full_text_row_masked_out_mask = input_ids.new_ones(input_ids.size(-1), 1)
        else:
            full_text_row_masked_out_mask = self.flat_encoder_result(cross_attn_metadata, input_ids)  # noqa

        cross_attention_states = None
        if pixel_values is not None:
            cross_attention_states = self.vision_model(
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
            )
            cross_attention_states = self.multi_modal_projector(cross_attention_states)
            _, bsz, _, _, image_token_dim = tuple(cross_attention_states.shape)
            cross_attention_states = cross_attention_states.view(bsz, -1, image_token_dim)

        hidden_states = self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            cross_attention_states=cross_attention_states,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            cross_attn_metadata=cross_attn_metadata,
        )
        return hidden_states

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return self.language_model.get_logits(hidden_states)

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.language_model.model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        context: StepContext = None,
    ):
        """Prepare input."""
        # get input_ids, position_ids and attention metadatas
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata
        cross_attn_metadata = context.cross_attn_metadata

        # cross_attn_metadata is None when inputs without image
        if cross_attn_metadata is not None and int(cross_attn_metadata.kv_seqlens.sum()) == 0:
            cross_attn_metadata.kv_seqlens = None

        device = input_ids.device

        # process image input
        pixel_values = None
        aspect_ratio_ids = None
        aspect_ratio_mask = None
        if context.input_multimodals is not None:
            pixel_values = []
            aspect_ratio_ids = []
            aspect_ratio_mask = []
            batched_image_data = [input_mm.get('image', []) for input_mm in context.input_multimodals]
            for image_data in batched_image_data:
                for data in image_data:
                    pixel_values.append(data.data)
                    aspect_ratio_ids.append(data.meta['aspect_ratio_ids'])
                    aspect_ratio_mask.append(data.meta['aspect_ratio_mask'])
            pixel_values = torch.cat(pixel_values, dim=0).to(device)
            aspect_ratio_ids = torch.cat(aspect_ratio_ids, dim=0).to(device)
            aspect_ratio_mask = torch.cat(aspect_ratio_mask, dim=0).to(device)

        # process vision embeddings
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:, vision_embedding_indexing, :] = vision_embeddings.to(inputs_embeds)

        # inputs of forward
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
            cross_attn_metadata=cross_attn_metadata,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights."""
        # modify from vllm
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name):
                continue
            if self.config.text_config.tie_word_embeddings and 'lm_head.weight' in name:  # noqa
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, shard_id=shard_id)
                break
            else:
                param = params_dict[name]
                load_weight(param, loaded_weight)

    def support_cuda_graph(
        self,
        input_ids: torch.Tensor,
        attn_metadata: Any,
        cross_attn_metadata: Any,
        **kwargs,
    ):
        """Support cudagraph."""

        if not attn_metadata.is_decoding:
            return False

        if cross_attn_metadata is None:
            return False

        if cross_attn_metadata.kv_seqlens is None:
            return False

        return True

    def make_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """Make cudagraph buffers from forward inputs."""
        input_buffers = super().make_buffers_cudagraph(graph_meta=graph_meta, **kwargs)

        device = graph_meta.device
        max_batches = graph_meta.max_batchs
        input_buffers['cross_kv_seqlens'] = torch.zeros(max_batches, dtype=torch.int64, device=device)

        return input_buffers

    def fill_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """Fill cudagraph buffers from forward inputs."""
        input_buffers = graph_meta.input_buffers

        new_inputs = super().fill_buffers_cudagraph(graph_meta=graph_meta, **kwargs)

        attn_metadata = new_inputs['attn_metadata']
        cross_attn_metadata = new_inputs['cross_attn_metadata']
        block_offsets = attn_metadata.block_offsets
        batch_size, _ = block_offsets.size()

        kv_seqlens = cross_attn_metadata.kv_seqlens
        if kv_seqlens.data_ptr() != input_buffers['cross_kv_seqlens'].data_ptr():
            input_buffers['cross_kv_seqlens'].zero_()
        input_buffers['cross_kv_seqlens'][:batch_size] = kv_seqlens

        new_batch_size = next_power_of_2(batch_size)
        cross_attn_metadata.block_offsets = input_buffers['block_offsets'][:new_batch_size]
        cross_attn_metadata.q_start_loc = input_buffers['q_start_loc'][:new_batch_size]
        cross_attn_metadata.q_seqlens = input_buffers['q_seqlens'][:new_batch_size]
        cross_attn_metadata.kv_seqlens = input_buffers['cross_kv_seqlens'][:new_batch_size]

        new_inputs['cross_attn_metadata'] = cross_attn_metadata
        return new_inputs

    def update_model_metas(self,
                           past_key_values: List[List[torch.Tensor]],
                           inputs_embeds: Optional[torch.Tensor] = None,
                           context: StepContext = None):
        """Update model meta."""
        model_metas = context.model_metas
        if model_metas is None:
            batch_size = context.q_seqlens.size(0)
            model_metas = [dict(cross_kv_len=0) for _ in range(batch_size)]

        if context.is_decoding:
            return model_metas

        vision_inputs = context.vision_inputs
        if vision_inputs is None:
            return model_metas

        input_mms = vision_inputs.input_multimodals
        if input_mms is None:
            return model_metas

        config = self.config.vision_config
        image_size = config.image_size
        patch_size = config.patch_size
        wh = image_size // patch_size
        img_kv_len = wh * wh + 1
        img_kv_len = img_kv_len * 4

        new_model_metas = []
        for idx, input_mm in enumerate(input_mms):
            if input_mm is None:
                new_model_metas.append(model_metas[idx])
            images = input_mm.get('image', [])
            num_img = len(images)

            cross_kv_len = 0
            if model_metas[idx] is not None:
                cross_kv_len = model_metas[idx].get('cross_kv_len', cross_kv_len)
            cross_kv_len += img_kv_len * num_img
            new_model_metas.append(dict(cross_kv_len=cross_kv_len))

        return model_metas

    def get_input_processor(self) -> BaseModelInputProcessor:
        """Get input processor."""
        return self.input_processor


class MLlamaInputProcessor(BaseModelInputProcessor):
    """Mllama input processor."""

    def __init__(self, config: LlamaConfig, dtype: torch.dtype) -> None:
        self.config = config
        self.dtype = dtype

        vision_config = self.config.vision_config
        image_size = vision_config.image_size
        patch_size = vision_config.patch_size
        wh = image_size // patch_size
        encoder_len = wh * wh + 1
        encoder_len = encoder_len * 4
        self.encoder_len = encoder_len

    def preprocess_input(self, input_ids, input_multimodals, **kwargs):
        """Prepare multimodal input."""
        if input_multimodals is None or len(input_multimodals) == 0:
            return input_ids, input_multimodals

        input_imgs = []
        for input_mm in input_multimodals:
            pixel_values = input_mm['pixel_values']
            aspect_ratio_ids = input_mm['aspect_ratio_ids']
            aspect_ratio_mask = input_mm['aspect_ratio_mask']
            offset = input_mm['offset']

            if pixel_values.dtype != self.dtype:
                pixel_values = pixel_values.to(self.dtype)

            mm_data = MultiModalTensor(data=pixel_values,
                                       start=offset,
                                       end=offset + 1,
                                       encoder_len=self.encoder_len,
                                       meta=dict(aspect_ratio_ids=aspect_ratio_ids,
                                                 aspect_ratio_mask=aspect_ratio_mask))
            input_imgs.append(mm_data)

        result = PreprocessInputResult(
            input_ids=input_ids,
            input_multimodals=dict(image=input_imgs),
        )
        return result

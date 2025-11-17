# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers.models.llama4 import Llama4Config, Llama4TextConfig, Llama4VisionConfig

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor, PreprocessInputResult
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.multimodal.data_type import MultiModalTensor
from lmdeploy.pytorch.nn import ApplyRotaryEmb, Attention, RMSNorm, SiluAndMul, build_rotary_embedding_from_config
from lmdeploy.pytorch.nn.linear import (build_colwise_linear, build_merged_colwise_linear, build_qkv_proj,
                                        build_rowwise_linear)
from lmdeploy.pytorch.nn.moe import build_fused_moe
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .utils.cudagraph import CudaGraphMixin


class Llama4TextAttention(nn.Module):
    """attention."""

    def __init__(self,
                 config: Llama4TextConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attn_scale = config.attn_scale
        self.floor_scale = config.floor_scale
        self.attn_temperature_tuning = config.attn_temperature_tuning
        self.is_causal = True
        self.use_rope = int((layer_idx + 1) % 4 != 0)  # rope unused for dense layers
        self.attn_bias = config.attention_bias

        # qkv
        self.qkv_proj = build_qkv_proj(
            config.hidden_size,
            num_q_heads=self.num_attention_heads,
            num_kv_heads=self.num_key_value_heads,
            head_size=self.head_dim,
            bias=self.attn_bias,
            dtype=dtype,
            device=device,
        )

        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        self.attn_fwd = Attention(
            self.num_attention_heads,
            self.head_dim,
            num_kv_heads=self.num_key_value_heads,
            v_head_size=self.head_dim,
        )

        # o_proj
        self.o_proj = build_rowwise_linear(config.num_attention_heads * self.head_dim,
                                           config.hidden_size,
                                           bias=self.attn_bias,
                                           dtype=dtype,
                                           device=device,
                                           is_tp=True)

        if self.config.use_qk_norm and self.use_rope:
            self.qk_norm = RMSNorm(self.head_dim, eps=1e-6, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_metadata: Any = None,
    ):
        """forward."""
        qkv_states = self.qkv_proj(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        query_states, key_states, value_states = self.qkv_proj.split_qkv(qkv_states)

        if self.use_rope:
            cos, sin = rotary_pos_emb
            # TODO: fuse apply rotary pos emb
            query_states = query_states.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2)
            key_states = key_states.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2)
            query_states, key_states = self.apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            )
            query_states = query_states.unflatten(-1, (2, -1)).transpose(-1, -2).flatten(-2)
            key_states = key_states.unflatten(-1, (2, -1)).transpose(-1, -2).flatten(-2)

        if hasattr(self, 'qk_norm'):
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)

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

        attn_output = self.o_proj(attn_output)

        return attn_output


class Llama4TextMLP(nn.Module):
    """attention."""

    def __init__(self,
                 config: Llama4TextConfig,
                 intermediate_size: int = None,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 is_tp: bool = True,
                 all_reduce: bool = True):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = config.intermediate_size

        self.config = config

        mlp_bias = False
        mlp_args = dict(
            bias=mlp_bias,
            dtype=dtype,
            device=device,
            is_tp=is_tp,
        )
        # gate up
        self.gate_up_proj = build_merged_colwise_linear(
            config.hidden_size,
            [intermediate_size, intermediate_size],
            **mlp_args,
        )

        # silu and mul
        self.act_fn = SiluAndMul(inplace=True)

        # down
        self.down_proj = build_rowwise_linear(
            intermediate_size,
            config.hidden_size,
            all_reduce=all_reduce,
            **mlp_args,
        )

    def forward(self, x):
        """forward."""
        gate_up = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        return self.down_proj(act)


class Llama4TextMoe(nn.Module):
    """attention."""

    def __init__(self, config: Llama4TextConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts

        self.router = build_rowwise_linear(
            self.hidden_dim,
            self.num_experts,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=False,
            quant_config=None,
        )
        self.experts = build_fused_moe(
            self.hidden_dim,
            self.ffn_dim,
            self.num_experts,
            top_k=1,
            renormalize=False,
            dtype=dtype,
            device=device,
            all_reduce=False,
            quant_config=quantization_config,
        )
        self.shared_expert = Llama4TextMLP(config, dtype=dtype, device=device, is_tp=True, all_reduce=False)

        dist_config = dist.get_dist_manager().current_config()
        self.tp = dist_config.tp

    def forward(self, hidden_states: torch.Tensor):
        """forward."""
        batch, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.router(hidden_states)

        topk_weights, topk_ids = torch.topk(router_logits, self.top_k, dim=-1)
        input_weight = topk_weights.float().sigmoid().to(hidden_states.dtype)

        moe_hidden_states = hidden_states[:, None, :] * input_weight[:, :, None]
        moe_hidden_states = moe_hidden_states.view(-1, hidden_dim)
        topk_weights = torch.ones_like(input_weight).reshape(-1, 1)
        topk_ids = topk_ids.reshape(-1, 1)

        out_states = self.experts(
            moe_hidden_states,
            topk_weights,
            topk_ids,
        )

        out_states = out_states.reshape(-1, self.top_k, hidden_dim)
        out_states = out_states.sum(1)

        shared_states = self.shared_expert(hidden_states)
        out_states += shared_states
        out_states = out_states.reshape(batch, seq_len, -1)

        if self.tp > 1:
            dist.all_reduce(out_states)

        return out_states


class Llama4TextDecoderLayer(nn.Module):
    """Decoder layer."""

    def __init__(self,
                 config: Llama4TextConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = Llama4TextAttention(config, layer_idx, dtype=dtype, device=device)
        self.use_chunked_attention = int((layer_idx + 1) % 4 != 0)  # <=> use rope
        self.is_moe_layer = layer_idx in config.moe_layers
        if self.is_moe_layer:  # the 128E model interleaves dense / sparse
            self.feed_forward = Llama4TextMoe(config, dtype=dtype, device=device)
        else:
            self.feed_forward = Llama4TextMLP(config,
                                              intermediate_size=config.intermediate_size_mlp,
                                              dtype=dtype,
                                              device=device)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
    ):
        """forward."""

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
        hidden_states = self.feed_forward(hidden_states)

        outputs = (hidden_states, residual)
        return outputs


class Llama4TextModel(nn.Module):
    """Llama4 text model."""

    def __init__(self, config: Llama4TextConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         self.padding_idx,
                                         dtype=dtype,
                                         device=device)
        self.layers = nn.ModuleList([
            Llama4TextDecoderLayer(config, layer_idx, dtype=dtype, device=device)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device)

        self.rotary_emb = self.build_llama4_rotary_embedding(config)

    @staticmethod
    def build_llama4_rotary_embedding(config: Llama4TextConfig):
        """Build llama4 rotary embedding."""
        return build_rotary_embedding_from_config(config)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        **kwargs,
    ):
        """Model forward."""
        hidden_states = inputs_embeds

        # rotary embedding
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        # decoding
        residual = None
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx]
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
            )

        # norm
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Llama4ForCausalLM(nn.Module):

    def __init__(self,
                 config: Llama4TextConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        self.model = Llama4TextModel(config, dtype=dtype, device=device)
        self.vocab_size = config.vocab_size
        self.lm_head = build_rowwise_linear(config.hidden_size,
                                            config.vocab_size,
                                            bias=False,
                                            device=device,
                                            dtype=dtype)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        **kwargs,
    ):
        """Model forward."""
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            **kwargs,
        )

        return outputs

    def get_input_embeddings(self):
        """Input embeddings."""
        return self.model.embed_tokens

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return self.lm_head(hidden_states)


class Llama4MultiModalProjector(nn.Module):

    def __init__(self, config: Llama4Config, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.vision_output_dim,
            config.text_config.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )

    def forward(self, image_features):
        """forward."""
        hidden_states = self.linear_1(image_features)
        return hidden_states


class Llama4UnfoldConvolution(nn.Module):
    """Llama4 unfold conv."""

    def __init__(self, config: Llama4VisionConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        kernel_size = config.patch_size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=config.patch_size)
        self.linear = nn.Linear(
            config.num_channels * kernel_size[0] * kernel_size[1],
            config.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """forward."""
        hidden_states = self.unfold(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.linear(hidden_states)
        return hidden_states


class Llama4VisionRotaryEmbedding(nn.Module):

    def __init__(self, config: Llama4VisionConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        idx = config.image_size // config.patch_size
        img_idx = torch.arange(idx**2, dtype=torch.int32).reshape(idx**2, 1)
        img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)
        img_idx[-1, -1] = -2  # ID_CLS_TOKEN
        frequencies_x = img_idx % idx  # get the coordinates of the 2d matrix along x
        frequencies_y = img_idx // idx  # get the coordinates of the 2d matrix along y
        freq_dim = config.hidden_size // config.num_attention_heads // 2
        rope_freq = 1.0 / (config.rope_theta**(torch.arange(0, freq_dim, 2)[:(freq_dim // 2)].float() / freq_dim))
        freqs_x = ((frequencies_x + 1)[..., None] * rope_freq[None, None, :]).repeat_interleave(2, dim=-1)
        freqs_y = ((frequencies_y + 1)[..., None] * rope_freq[None, None, :]).repeat_interleave(2, dim=-1)
        freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]
        freqs = freqs.masked_fill(img_idx.reshape(-1, 1, 1) < 0, 0)
        freq_cis = torch.view_as_complex(torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1))
        self.freqs_ci = freq_cis.to(device)  # idx**2, idx**2, idx * 2

    def forward(self, hidden_states):
        return self.freqs_ci


def reshape_for_broadcast(freqs_ci: torch.Tensor, query: torch.Tensor):
    ndim = query.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(query.shape)]
    return freqs_ci.view(*shape)


def vision_apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs_ci: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    query_ = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
    key_ = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
    freqs_ci = reshape_for_broadcast(freqs_ci=freqs_ci, query=query_)  # freqs_ci[:,:,None,:]
    freqs_ci = freqs_ci.to(query_.device)
    query_out = torch.view_as_real(query_ * freqs_ci).flatten(3)
    key_out = torch.view_as_real(key_ * freqs_ci).flatten(3)
    return query_out.type_as(query), key_out.type_as(key)  # but this drops to 8e-3


class Llama4VisionAttention(nn.Module):
    """Vision attn."""

    def __init__(self, config: Llama4VisionConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # qkv
        self.qkv_proj = build_qkv_proj(
            self.embed_dim,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_size=self.head_dim,
            bias=True,
            dtype=dtype,
            device=device,
            is_tp=True,
        )

        # o_proj
        self.o_proj = build_rowwise_linear(self.num_heads * self.head_dim,
                                           self.embed_dim,
                                           bias=True,
                                           dtype=dtype,
                                           device=device,
                                           is_tp=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_ci: torch.Tensor,
    ):
        """forward."""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        qkv_states = self.qkv_proj(hidden_states)
        # (-1, heads, head_dim)
        query_states, key_states, value_states = qkv_states.chunk(3, dim=-1)
        query_states = query_states.reshape(hidden_shape)
        key_states = key_states.reshape(hidden_shape)
        value_states = value_states.reshape(hidden_shape)

        query_states, key_states = vision_apply_rotary_emb(query_states, key_states, freqs_ci=freqs_ci)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        attention_interface = ALL_ATTENTION_FUNCTIONS['sdpa']
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            None,
            dropout=0.0,
            scaling=None,
            is_causal=False,  # HAS TO BE ENFORCED
            output_attentions=False,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class Llama4VisionMLP(nn.Module):
    """Vision mlp."""

    def __init__(self, config: Llama4VisionConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.activation_fn = nn.GELU()
        self.fc1 = build_colwise_linear(config.hidden_size,
                                        config.intermediate_size,
                                        bias=True,
                                        dtype=dtype,
                                        device=device,
                                        is_tp=True)
        self.fc2 = build_rowwise_linear(config.intermediate_size,
                                        config.hidden_size,
                                        bias=True,
                                        dtype=dtype,
                                        device=device,
                                        is_tp=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """forward."""
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Llama4VisionEncoderLayer(nn.Module):
    """Vision encoder layer."""

    def __init__(self, config: Llama4VisionConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config

        self.self_attn = Llama4VisionAttention(config, dtype=dtype, device=device)
        self.mlp = Llama4VisionMLP(config, dtype=dtype, device=device)

        self.input_layernorm = nn.LayerNorm(config.hidden_size, dtype=dtype, device=device)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, dtype=dtype, device=device)

    def forward(
        self,
        hidden_state: torch.Tensor,
        freqs_ci: torch.Tensor,
    ):
        """forward."""
        # Self Attention
        residual = hidden_state

        hidden_state = self.input_layernorm(hidden_state)

        hidden_state = self.self_attn(
            hidden_state,
            freqs_ci=freqs_ci,
        )
        hidden_state = residual + hidden_state

        # Feed forward
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = residual + hidden_state

        return hidden_state


class Llama4VisionEncoder(nn.Module):
    """Vision encoder."""

    def __init__(self, config: Llama4VisionConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Llama4VisionEncoderLayer(config, dtype=dtype, device=device) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_ci: torch.Tensor,
    ):
        """forward."""
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_state=hidden_states,
                freqs_ci=freqs_ci,
            )
        return hidden_states


def pixel_shuffle(input_tensor: torch.Tensor, shuffle_ratio: int):
    # input_tensor: [batch_size, num_patches, channels]
    import math
    batch_size, num_patches, channels = input_tensor.shape
    patch_size = int(math.sqrt(num_patches))

    input_tensor = input_tensor.view(batch_size, patch_size, patch_size, -1)
    batch_size, height, width, channels = input_tensor.size()

    reshaped_tensor = input_tensor.view(batch_size, height, int(width * shuffle_ratio), int(channels / shuffle_ratio))
    reshaped_tensor = reshaped_tensor.permute(0, 2, 1, 3).contiguous()

    reshaped_tensor = reshaped_tensor.view(batch_size, int(height * shuffle_ratio), int(width * shuffle_ratio),
                                           int(channels / (shuffle_ratio**2)))
    reshaped_tensor = reshaped_tensor.permute(0, 2, 1, 3).contiguous()

    output_tensor = reshaped_tensor.view(batch_size, -1, reshaped_tensor.shape[-1])
    return output_tensor


class Llama4VisionMLP2(torch.nn.Module):

    def __init__(self, config: Llama4VisionConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.fc1 = build_colwise_linear(self.intermediate_size,
                                        config.projector_input_dim,
                                        bias=False,
                                        dtype=dtype,
                                        device=device,
                                        is_tp=True)
        self.fc2 = build_rowwise_linear(config.projector_output_dim,
                                        config.projector_output_dim,
                                        bias=False,
                                        dtype=dtype,
                                        device=device,
                                        is_tp=True)
        self.activation_fn = nn.GELU()  # ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        """forward."""
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        return self.activation_fn(self.fc2(hidden_states))


class Llama4VisionPixelShuffleMLP(nn.Module):

    def __init__(self, config: Llama4VisionConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.pixel_shuffle_ratio = config.pixel_shuffle_ratio
        self.inner_dim = int(config.projector_input_dim // (self.pixel_shuffle_ratio**2))
        self.output_dim = config.projector_output_dim
        self.mlp = Llama4VisionMLP2(config, dtype=dtype, device=device)

    def forward(self, encoded_patches: torch.Tensor) -> torch.Tensor:
        encoded_patches = pixel_shuffle(encoded_patches, self.pixel_shuffle_ratio)
        return self.mlp(encoded_patches)


class Llama4VisionModel(nn.Module):
    """Llama4 vision model."""

    def __init__(self, config: Llama4VisionConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels

        self.num_patches = (self.image_size // self.patch_size)**2 + 1
        self.scale = config.hidden_size**-0.5

        self.patch_embedding = Llama4UnfoldConvolution(config, dtype=dtype, device=device)

        self.class_embedding = nn.Parameter(self.scale * torch.empty(self.hidden_size, dtype=dtype, device=device))
        self.positional_embedding_vlm = nn.Parameter(
            self.scale * torch.empty(self.num_patches, self.hidden_size, dtype=dtype, device=device))
        self.rotary_embedding = Llama4VisionRotaryEmbedding(config, dtype=dtype, device=device)

        # layer norms
        self.layernorm_pre = nn.LayerNorm(self.hidden_size, dtype=dtype, device=device)
        self.layernorm_post = nn.LayerNorm(self.hidden_size, dtype=dtype, device=device)

        # encoders
        self.model = Llama4VisionEncoder(config, dtype=dtype, device=device)
        self.vision_adapter = Llama4VisionPixelShuffleMLP(config, dtype=dtype, device=device)

    def get_input_embeddings(self):
        """This function is used to fetch the first embedding layer to activate
        grads on inputs."""
        return self.patch_embedding

    def forward(
        self,
        pixel_values: torch.Tensor,
    ):
        """forward."""
        batch_size_times_num_tiles, num_channels, height, width = pixel_values.shape
        num_concurrent_media = 1
        num_chunks = 1
        hidden_state = self.patch_embedding(pixel_values)
        _, num_patches, hidden_dim = hidden_state.shape

        # Add cls token
        hidden_state = hidden_state.reshape(batch_size_times_num_tiles * num_concurrent_media * num_chunks, num_patches,
                                            hidden_dim)
        class_embedding = self.class_embedding.expand(hidden_state.shape[0], 1, hidden_state.shape[-1])
        hidden_state = torch.cat([hidden_state, class_embedding], dim=1)
        num_patches += 1

        # Position embeddings
        hidden_state = hidden_state.reshape(batch_size_times_num_tiles * num_concurrent_media, num_chunks, num_patches,
                                            hidden_dim)
        positional_embedding = self.positional_embedding_vlm.to(dtype=hidden_state.dtype, device=hidden_state.device)
        hidden_state = hidden_state + positional_embedding

        hidden_state = self.layernorm_pre(hidden_state)

        hidden_state = hidden_state.view(batch_size_times_num_tiles, -1, hidden_dim)
        freqs_ci = self.rotary_embedding(pixel_values)

        output = self.model(
            hidden_state,
            freqs_ci=freqs_ci,
        )

        hidden_state = output

        hidden_state = self.layernorm_post(hidden_state)

        hidden_state = hidden_state[:, :-1, :]

        # now, we use Llama4VisionPixelShuffle + mlp to project embeddings
        hidden_state = self.vision_adapter(hidden_state)

        return hidden_state


class Llama4ForConditionalGeneration(nn.Module, CudaGraphMixin):

    def __init__(self,
                 config: Llama4Config,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr

        self.vision_model = Llama4VisionModel(config.vision_config, dtype=dtype, device=device)

        self.multi_modal_projector = Llama4MultiModalProjector(config, dtype=dtype, device=device)

        self._update_quant_config(config)
        self.language_model = Llama4ForCausalLM(config.text_config, ctx_mgr, dtype=dtype, device=device)
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

        self.input_processor = Llama4InputProcessor(config, dtype)

    @staticmethod
    def _update_quant_config(config: Llama4Config):
        """Update quant config."""
        quant_config = getattr(config, 'quantization_config', None)

        if quant_config is None:
            return config

        quantization_config = dict(
            quant_dtype='float8_e4m3fn',
            quant_method='smooth_quant',
        )
        text_config = config.text_config
        setattr(text_config, 'quantization_config', quantization_config)

        return config

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs,
    ):
        """Get image features."""
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        hidden_state = self.vision_model(pixel_values, **kwargs)
        return hidden_state

    def get_input_embeddings(self):
        """Input embeddings."""
        return self.language_model.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        pixel_values: torch.FloatTensor = None,
        image_mask: torch.Tensor = None,
        **kwargs,
    ):
        """Model forward."""
        image_embeds = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values=pixel_values, )
            vision_flat = image_features.view(-1, image_features.size(-1))
            image_embeds = self.multi_modal_projector(vision_flat)

        lang_embeds: torch.Tensor = self.get_input_embeddings()(input_ids)

        if image_embeds is not None:
            lang_embeds.masked_scatter_(image_mask[..., None], image_embeds)

        inputs_embeds = lang_embeds

        return self.language_model(
            inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
        )

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return self.language_model.get_logits(hidden_states)

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

        # vision inputs
        pixel_values = None
        image_mask = None
        if context.input_multimodals is not None:
            pixel_values = [input_mm.get('image', []) for input_mm in context.input_multimodals]
            # flatten batch
            pixel_values = [data for im_data in pixel_values for data in im_data]
            if len(pixel_values) > 0:
                image_token_id = pixel_values[0].meta['image_token_id']
                image_mask = input_ids == image_token_id
                pixel_values = torch.cat([data.data for data in pixel_values])
            else:
                pixel_values = None
                image_mask = None

        # inputs of forward
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_mask=image_mask,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights."""

        def _load_experts_bf16(name, loaded_weight):
            if '.gate_up_proj' in name:
                loaded_weight = loaded_weight.to(device)
                name = name.replace('.gate_up_proj', '.gate_up.weight')
                param = params_dict[name]
                for exp_id in range(num_experts):
                    weight_gate, weight_up = loaded_weight[exp_id].chunk(2, -1)
                    load_weight(param, weight_gate.t(), expert_id=exp_id, shard_id='gate')
                    load_weight(param, weight_up.t(), expert_id=exp_id, shard_id='up')
            elif '.down_proj' in name:
                loaded_weight = loaded_weight.to(device)
                name = name.replace('.down_proj', '.down.weight')
                param = params_dict[name]
                for exp_id in range(num_experts):
                    weight = loaded_weight[exp_id].t()
                    load_weight(param, weight, expert_id=exp_id, shard_id='down')

        def _load_experts_fp8(name, loaded_weight):
            name = name.replace('.weight_scale', '.scale')
            for (param_name, weight_name, expert_id, shard_id) in expert_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, expert_id=expert_id, shard_id=shard_id)
                break
            else:
                param = params_dict[name]
                load_weight(param, loaded_weight)

        def _load_experts(name, loaded_weight):
            """Load experts weight."""
            quantization_config = getattr(self.config, 'quantization_config', None)
            if quantization_config is None:
                _load_experts_bf16(name, loaded_weight)
            else:
                _load_experts_fp8(name, loaded_weight)

        # modify from vllm
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        num_experts = self.config.text_config.num_local_experts
        expert_params_mapping = []
        for exp_id in range(num_experts):
            gate_param = ('.experts.gate_up', f'.experts.{exp_id}.gate_proj', exp_id, 'gate')
            up_param = ('.experts.gate_up', f'.experts.{exp_id}.up_proj', exp_id, 'up')
            down_param = ('.experts.down', f'.experts.{exp_id}.down_proj', exp_id, 'down')
            expert_params_mapping += [gate_param, up_param, down_param]

        params_dict = dict(self.named_parameters())
        device = next(iter(params_dict.values())).device
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name):
                continue
            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue

            if '.experts' in name:
                _load_experts(name, loaded_weight)
            else:
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

    def get_input_processor(self) -> BaseModelInputProcessor:
        """Get input processor."""
        return self.input_processor


class Llama4InputProcessor(BaseModelInputProcessor):
    """Llama4 input processor."""

    def __init__(self, config: Llama4Config, dtype) -> None:
        self.config = config
        self.dtype = dtype

        self.vision_config = config.vision_config

    def preprocess_input(self,
                         input_ids: List[int],
                         input_multimodals: List[Dict[str, Any]] = None,
                         **kwargs) -> PreprocessInputResult:
        """Prepare multimodal input."""

        if input_multimodals is None or len(input_multimodals) == 0:
            return input_ids, input_multimodals

        input_imgs = []
        for input_mm in input_multimodals:
            pixel_values = input_mm['pixel_values'].to(self.dtype)
            offset = input_mm['offset']
            image_token_id = input_mm['image_token_id']
            num_pad = input_mm['image_tokens']
            if isinstance(num_pad, torch.Tensor):
                num_pad = num_pad.item()

            mm_data = MultiModalTensor(data=pixel_values,
                                       start=offset,
                                       end=offset + num_pad,
                                       meta=dict(image_token_id=image_token_id))
            input_imgs.append(mm_data)

        result = PreprocessInputResult(
            input_ids=input_ids,
            input_multimodals=dict(image=input_imgs),
        )
        return result

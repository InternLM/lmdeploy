# Copyright (c) OpenMMLab. All rights reserved.
# adpated from https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.nn import LayerNorm
from lmdeploy.pytorch.nn.linear import build_colwise_linear, build_qkv_proj, build_rowwise_linear


class WhisperAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config: PretrainedConfig = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> None:
        super().__init__()
        self.config = config
        quantization_config = getattr(config, 'quantization_config', None)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}'
                             f' and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim**-0.5

        # packed qkv
        # TODO, zhouxinyu, hf whisper hard-code k_proj bias = False, may double check
        self.qkv_proj = build_qkv_proj(self.embed_dim,
                                       num_q_heads=self.num_heads,
                                       num_kv_heads=self.num_heads,
                                       head_size=self.head_dim,
                                       bias=bias,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device)

        # o_proj
        self.out_proj = build_rowwise_linear(self.embed_dim,
                                             self.embed_dim,
                                             bias=bias,
                                             quant_config=quantization_config,
                                             dtype=dtype,
                                             device=device,
                                             is_tp=True)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """forward."""
        # qkv proj
        qkv_states = self.qkv_proj(hidden_states)
        q, k, v = self.qkv_proj.split_qkv(qkv_states)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = q * self.scaling

        # attention
        attn_output = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, scale=1.0)

        # o proj
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.flatten(-2, -1)
        attn_output = self.out_proj(attn_output)
        return attn_output


class WhisperEncoderLayer(nn.Module):

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None) -> None:
        super().__init__()
        self.config = config
        quantization_config = getattr(config, 'quantization_config', None)

        self.act = ACT2FN[config.activation_function]
        self.embed_dim = config.d_model

        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            config=config,
            dtype=dtype,
            device=device,
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, dtype=dtype, device=device)
        self.fc1 = build_colwise_linear(
            self.embed_dim,
            config.encoder_ffn_dim,
            bias=True,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )
        self.fc2 = build_rowwise_linear(
            config.encoder_ffn_dim,
            self.embed_dim,
            bias=True,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )
        self.final_layer_norm = LayerNorm(self.embed_dim, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.act(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

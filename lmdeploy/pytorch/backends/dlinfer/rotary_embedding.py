# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from ..default.rotary_embedding import (Llama3RotaryEmbeddingImpl,
                                        LlamaDynamicNTKScalingRotaryEmbedding)
from ..rotary_embedding import (Llama3Parameters, LongRoPEScalingParameters,
                                RopeType, RotaryEmbeddingBuilder,
                                RotaryEmbeddingImpl, YarnParameters)


class DlinferRotaryEmbeddingImpl(RotaryEmbeddingImpl, nn.Module):
    """base rotary embedding."""

    def __init__(self,
                 dim: int,
                 base: int = 10000,
                 scaling_factor: float = 1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base**(
            torch.arange(0, self.dim, 2, dtype=torch.int64).float() /
            self.dim)).float().cuda()
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, x, position_ids):
        """forward."""
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq.device != x.device:
            self.inv_freq = self.inv_freq.to(x.device)

        if self.scaling_factor != 1.0:
            position_ids = position_ids.float() / self.scaling_factor
        else:
            position_ids = position_ids.float()

        inv_freq_expanded = self.inv_freq.view(1, -1, 1)
        position_ids_expanded = position_ids.unsqueeze(1)

        # # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(
            device_type, str) and device_type != 'mps' else 'cpu'
        inv_freq_expanded = inv_freq_expanded
        position_ids_expanded = position_ids_expanded
        tmp = torch.bmm(inv_freq_expanded, position_ids_expanded)
        freqs = tmp.transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DlinferRotaryEmbeddingBuilder(RotaryEmbeddingBuilder):
    """rotary embedding builder."""

    @staticmethod
    def build(
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        scaling_factor: float = 1.0,
        yarn_params: YarnParameters = None,
        longrope_params: LongRoPEScalingParameters = None,
        llama3_params: Llama3Parameters = None,
        emb_type: RopeType = RopeType.Default,
    ):
        """build."""
        if emb_type in (RopeType.Default, RopeType.LinearScaling):
            return DlinferRotaryEmbeddingImpl(dim, base, scaling_factor)
        elif emb_type == RopeType.DynamicNTKScaling:
            return LlamaDynamicNTKScalingRotaryEmbedding(
                dim, base, scaling_factor, max_position_embeddings)
        elif emb_type == RopeType.Llama3:
            return Llama3RotaryEmbeddingImpl(dim, base, scaling_factor,
                                             llama3_params.low_freq_factor,
                                             llama3_params.high_freq_factor,
                                             max_position_embeddings)
        else:
            raise NotImplementedError(
                f'Unsupported embedding type: {emb_type}')

# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from ..default.rotary_embedding import (Llama3RotaryEmbeddingImpl,
                                        LlamaDynamicNTKScalingRotaryEmbedding)
from ..rotary_embedding import (Llama3Parameters, LongRoPEScalingParameters,
                                RopeType, RotaryEmbeddingBuilder,
                                RotaryEmbeddingImpl, YarnParameters)


def _rotary_embedding_fwd(position_ids: torch.Tensor,
                          inv_freq: torch.Tensor,
                          scaling_factor: float,
                          mscale: float = None,
                          dtype: torch.dtype = None,
                          device_type: torch.device = None):
    """rotary embedding forward."""
    if dtype is None:
        dtype = torch.float16

    if scaling_factor != 1.0:
        position_ids = position_ids.float() / scaling_factor
    else:
        position_ids = position_ids.float()

    inv_freq_expanded = inv_freq.view(1, -1, 1)
    position_ids_expanded = position_ids.unsqueeze(1)

    inv_freq_expanded = inv_freq_expanded
    position_ids_expanded = position_ids_expanded
    tmp = torch.bmm(inv_freq_expanded, position_ids_expanded)
    freqs = tmp.transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    if mscale is not None:
        cos = cos * mscale
        sin = sin * mscale

    return cos.to(dtype=dtype), sin.to(dtype=dtype)


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
        device_type = x.device.type
        dtype = x.dtype
        if self.inv_freq.device != x.device:
            self.inv_freq = self.inv_freq.to(x.device)
        return _rotary_embedding_fwd(position_ids,
                                     self.inv_freq,
                                     scaling_factor=self.scaling_factor,
                                     dtype=dtype,
                                     device_type=device_type)


class DlinferLlamaDynamicNTKScalingRotaryEmbedding(
        LlamaDynamicNTKScalingRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling.

    Credits to the Reddit users /u/bloc97 and /u/emozilla
    """

    def __init__(self,
                 dim: int,
                 base: int = 10000,
                 scaling_factor: float = 1.0,
                 max_position_embeddings: int = 2048):
        super().__init__(dim, base, scaling_factor, max_position_embeddings)
        self.exponent_1 = self.dim / (self.dim - 2)
        self.exponent_2 = torch.arange(
            0, self.dim, 2, dtype=torch.int64).float().cuda() / self.dim
        self.sub = self.scaling_factor - 1
        self.div = self.scaling_factor / self.max_position_embeddings

    def _ntk_inv_freq(self, seq_len: torch.Tensor):
        """ntk_inv_freq."""
        base = self.base * ((self.div * seq_len) - self.sub)**self.exponent_1
        inv_freq = 1.0 / (base**self.exponent_2)
        return inv_freq

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """forward."""
        device_type = x.device.type
        dtype = x.dtype
        seq_len = torch.max(position_ids) + 1
        ntk_inv_freq = self._ntk_inv_freq(seq_len)
        if self.inv_freq.device != x.device:
            self.inv_freq = self.inv_freq.to(x.device)
        inv_freq = torch.where(seq_len > self.max_position_embeddings,
                               ntk_inv_freq, self.inv_freq)

        cos, sin = _rotary_embedding_fwd(position_ids,
                                         inv_freq,
                                         scaling_factor=1.0,
                                         dtype=dtype,
                                         device_type=device_type)
        return cos, sin


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
            return DlinferLlamaDynamicNTKScalingRotaryEmbedding(
                dim, base, scaling_factor, max_position_embeddings)
        elif emb_type == RopeType.Llama3:
            return Llama3RotaryEmbeddingImpl(dim, base, scaling_factor,
                                             llama3_params.low_freq_factor,
                                             llama3_params.high_freq_factor,
                                             max_position_embeddings)
        else:
            raise NotImplementedError(
                f'Unsupported embedding type: {emb_type}')

# Copyright (c) OpenMMLab. All rights reserved.

import math

import torch
from torch import nn

from ..default.rotary_embedding import LlamaDynamicNTKScalingRotaryEmbedding, YarnRotaryEmbeddingImpl
from ..rotary_embedding import (Llama3Parameters, LongRoPEScalingParameters, RopeType, RotaryEmbeddingBuilder,
                                RotaryEmbeddingImpl, YarnParameters)


def _rotary_embedding_fwd(position_ids: torch.Tensor,
                          inv_freq: torch.Tensor,
                          scaling_factor: float,
                          mscale: float = None,
                          dtype: torch.dtype = None):
    """Rotary embedding forward."""
    if dtype is None:
        dtype = torch.float16

    if scaling_factor != 1.0:
        position_ids = position_ids.float() / scaling_factor
    else:
        position_ids = position_ids.float()

    position_ids = position_ids.unsqueeze(-1)
    angles = position_ids * inv_freq.view(1, 1, -1)
    angles = torch.cat((angles, angles), dim=-1)

    sin = angles.sin()
    cos = angles.cos()

    if mscale is not None:
        cos = cos * mscale
        sin = sin * mscale
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


class DlinferRotaryEmbeddingImpl(RotaryEmbeddingImpl, nn.Module):
    """Base rotary embedding."""

    def __init__(self, dim: int, base: int = 10000, scaling_factor: float = 1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.base = base
        # yapf: disable
        inv_freq = 1.0 / (self.base**(torch.arange(0, self.dim, 2, dtype=torch.float, device='cuda') / self.dim))
        # yapf: enable
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, x, position_ids):
        """forward."""
        # x: [bs, num_attention_heads, seq_len, head_size]
        dtype = x.dtype
        if self.inv_freq.device != x.device:
            self.inv_freq = self.inv_freq.to(x.device)
        return _rotary_embedding_fwd(position_ids, self.inv_freq, scaling_factor=self.scaling_factor, dtype=dtype)


class DlinferLlamaDynamicNTKScalingRotaryEmbedding(LlamaDynamicNTKScalingRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling.

    Credits to the Reddit users /u/bloc97 and /u/emozilla
    """

    def __init__(self, dim: int, base: int = 10000, scaling_factor: float = 1.0, max_position_embeddings: int = 2048):
        super().__init__(dim, base, scaling_factor, max_position_embeddings)
        self.dim_scale_ratio = self.dim / (self.dim - 2)
        self.pos_freq_scaling = torch.arange(0, self.dim, 2, dtype=torch.int64).float().cuda() / self.dim
        self.scale_offset = self.scaling_factor - 1
        self.pos_scale_factor = self.scaling_factor / \
            self.max_position_embeddings

    def _ntk_inv_freq(self, seq_len: torch.Tensor):
        """Calculate inverse frequency with NTK scaling."""
        base = self.base * ((self.pos_scale_factor * seq_len) - self.scale_offset)**self.dim_scale_ratio
        inv_freq = 1.0 / (base**self.pos_freq_scaling)
        return inv_freq

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """forward."""
        dtype = x.dtype
        seq_len = torch.max(position_ids) + 1
        ntk_inv_freq = self._ntk_inv_freq(seq_len)
        if self.inv_freq.device != x.device:
            self.inv_freq = self.inv_freq.to(x.device)
        inv_freq = torch.where(seq_len > self.max_position_embeddings, ntk_inv_freq, self.inv_freq)

        cos, sin = _rotary_embedding_fwd(position_ids, inv_freq, scaling_factor=1.0, dtype=dtype)
        return cos, sin


class DlinferLlama3RotaryEmbeddingImpl(DlinferRotaryEmbeddingImpl):
    """Llama3 rotary embedding implementation."""

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        scaling_factor: float = 1.0,
        low_freq_factor: float = 1.0,
        high_freq_factor: float = 4.0,
        original_max_position_embeddings: int = 8194,
    ):
        super().__init__(dim, base, scaling_factor)
        old_context_len = original_max_position_embeddings
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        inv_freq = self.inv_freq
        factor = self.scaling_factor

        wavelen = 2 * math.pi / inv_freq
        # wavelen < high_freq_wavelen: do nothing
        # wavelen > low_freq_wavelen: divide by factor
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        # otherwise: interpolate between the two, using a smooth factor
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        self.scaling_factor = 1.0
        self.register_buffer('inv_freq', inv_freq_llama)


class DlinferYarnRotaryEmbeddingImpl(YarnRotaryEmbeddingImpl):
    """Yarn rotary embedding implementation."""

    def __init__(self,
                 dim: int,
                 base: int = 10000,
                 scaling_factor: float = 1.0,
                 original_max_position_embeddings: int = 4096,
                 yarn_params: YarnParameters = None):
        super().__init__(dim, base, scaling_factor, original_max_position_embeddings, yarn_params)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """forward."""
        dtype = x.dtype
        if self.inv_freq.device != x.device:
            self.inv_freq = self.inv_freq.to(x.device)
        return _rotary_embedding_fwd(position_ids, self.inv_freq, scaling_factor=1.0, mscale=self.mscale, dtype=dtype)


class DlinferRotaryEmbeddingBuilder(RotaryEmbeddingBuilder):
    """Rotary embedding dlinfer builder."""

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
            return DlinferLlamaDynamicNTKScalingRotaryEmbedding(dim, base, scaling_factor, max_position_embeddings)
        elif emb_type == RopeType.Llama3:
            return DlinferLlama3RotaryEmbeddingImpl(dim, base, scaling_factor, llama3_params.low_freq_factor,
                                                    llama3_params.high_freq_factor, max_position_embeddings)
        elif emb_type == RopeType.Yarn:
            return DlinferYarnRotaryEmbeddingImpl(dim,
                                                  base,
                                                  scaling_factor,
                                                  max_position_embeddings,
                                                  yarn_params=yarn_params)
        else:
            raise NotImplementedError(f'Unsupported embedding type: {emb_type}')

# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from ..rotary_embedding import (EmbeddingType, RotaryEmbeddingBuilder,
                                RotaryEmbeddingImpl)


def _rotary_embedding_fwd(position_ids: torch.Tensor, inv_freq: torch.Tensor,
                          scaling_factor: float, dtype: torch.dtype,
                          device_type: torch.device):
    """rotary embedding forward."""
    position_ids = position_ids.float() / scaling_factor
    inv_freq_expanded = inv_freq[None, :,
                                 None].float().expand(position_ids.shape[0],
                                                      -1, 1)
    position_ids_expanded = position_ids[:, None, :]
    # Force float32 since bfloat16 loses precision on long contexts
    # See https://github.com/huggingface/transformers/pull/29285
    device_type = device_type if isinstance(
        device_type, str) and device_type != 'mps' else 'cpu'
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float()
                 @ position_ids_expanded.float()).transpose(1, 2)
        emb = freqs.repeat(1, 1, 2)
        cos = emb.cos()
        sin = emb.sin()
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


class RotaryEmbeddingImpl(RotaryEmbeddingImpl, nn.Module):
    """base rotary embedding."""

    def __init__(self,
                 dim: int,
                 base: int = 10000,
                 scaling_factor: float = 1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base**(torch.arange(
            0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """forward."""
        device_type = x.device.type
        dtype = x.dtype
        if self.inv_freq.device != x.device:
            self.inv_freq = self.inv_freq.to(x.device)
        return _rotary_embedding_fwd(position_ids,
                                     self.inv_freq,
                                     scaling_factor=self.scaling_factor,
                                     dtype=dtype,
                                     device_type=device_type)


class LlamaDynamicNTKScalingRotaryEmbedding(RotaryEmbeddingImpl):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling.

    Credits to the Reddit users /u/bloc97 and /u/emozilla
    """

    def __init__(self,
                 dim: int,
                 base: int = 10000,
                 scaling_factor: float = 1.0,
                 max_position_embeddings: int = 2048):
        super().__init__(dim, base, scaling_factor)
        self.max_position_embeddings = max_position_embeddings

    def _ntk_inv_freq(self, seq_len: torch.Tensor):
        """ntk_inv_freq."""
        device = seq_len.device
        base = self.base * (
            (self.scaling_factor * seq_len / self.max_position_embeddings) -
            (self.scaling_factor - 1))**(self.dim / (self.dim - 2))
        inv_freq = 1.0 / (base**(torch.arange(
            0, self.dim, 2, dtype=torch.int64, device=device).float() /
                                 self.dim))
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
                                         scaling_factor=self.scaling_factor,
                                         dtype=dtype,
                                         device_type=device_type)
        return cos, sin


class DefaultRotaryEmbeddingBuilder(RotaryEmbeddingBuilder):
    """rotary embedding builder."""

    @staticmethod
    def build(
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        scaling_factor: float = 1.0,
        emb_type: EmbeddingType = EmbeddingType.Default,
    ):
        """build."""
        if emb_type in (EmbeddingType.Default, EmbeddingType.LinearScaling):
            return RotaryEmbeddingImpl(dim, base, scaling_factor)
        elif emb_type == EmbeddingType.DynamicNTKScaling:
            return LlamaDynamicNTKScalingRotaryEmbedding(
                dim, base, scaling_factor, max_position_embeddings)
        else:
            raise NotImplementedError(
                f'Unsupported embedding type: {emb_type}')

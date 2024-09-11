# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from torch import nn

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
    if device_type is None:
        device_type = 'cuda'
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

        if mscale is not None:
            cos = cos * mscale
            sin = sin * mscale

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
                                         scaling_factor=1.0,
                                         dtype=dtype,
                                         device_type=device_type)
        return cos, sin


class Llama3RotaryEmbeddingImpl(RotaryEmbeddingImpl):
    """llama3 rotary embedding implementation."""

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
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen,
                                     inv_freq / factor, inv_freq)
        # otherwise: interpolate between the two, using a smooth factor
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (
            1 - smooth_factor
        ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen >
                                                            low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq,
                                     inv_freq_llama)
        self.scaling_factor = 1.0
        self.register_buffer('inv_freq', inv_freq_llama)


def yarn_find_correction_dim(num_rotations,
                             dim,
                             base=10000,
                             max_position_embeddings=2048):
    """yarn_find_correction_dim."""
    return (dim * math.log(max_position_embeddings /
                           (num_rotations * 2 * math.pi))) / (2 *
                                                              math.log(base))


# Find dim range bounds based on rotations
def yarn_find_correction_range(low_rot,
                               high_rot,
                               dim,
                               base=10000,
                               max_position_embeddings=2048):
    """yarn_find_correction_range."""
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    """yarn_get_mscale."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    """yarn_linear_ramp_mask."""
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class YarnRotaryEmbeddingImpl(RotaryEmbeddingImpl):
    """yarn rotary embedding implementation."""

    def __init__(self,
                 dim: int,
                 base: int = 10000,
                 scaling_factor: float = 1.0,
                 original_max_position_embeddings: int = 4096,
                 yarn_params: YarnParameters = None):
        super().__init__(dim, base, scaling_factor)
        self.original_max_position_embeddings = \
            original_max_position_embeddings
        assert yarn_params is not None
        self.beta_fast = yarn_params.beta_fast
        self.beta_slow = yarn_params.beta_slow
        self.mscale = yarn_params.mscale
        self.mscale_all_dim = yarn_params.mscale_all_dim

        # get inv_freq
        freq_extra = 1.0 / (self.base**(
            torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freq_inter = 1.0 / (self.scaling_factor * self.base**(
            torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(
            low, high, dim // 2).to(dtype=torch.float32)
        inv_freq = freq_inter * (1 -
                                 inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # get mscale
        self.mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale) /
            yarn_get_mscale(self.scaling_factor, self.mscale_all_dim))
        if self.mscale == 1.0:
            self.mscale = None

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """forward."""
        device_type = x.device.type
        dtype = x.dtype
        if self.inv_freq.device != x.device:
            self.inv_freq = self.inv_freq.to(x.device)
        return _rotary_embedding_fwd(position_ids,
                                     self.inv_freq,
                                     scaling_factor=1.0,
                                     mscale=self.mscale,
                                     dtype=dtype,
                                     device_type=device_type)


class LongRoPEScalingRotaryEmbeddingImpl(RotaryEmbeddingImpl):
    """yarn rotary embedding implementation."""

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        max_position_embeddings: int = 4096,
        longrope_params: LongRoPEScalingParameters = None,
    ):
        super().__init__(dim, base)
        short_factor = torch.tensor(longrope_params.short_factor,
                                    dtype=torch.float32)
        long_factor = torch.tensor(longrope_params.long_factor,
                                   dtype=torch.float32)
        self.register_buffer('short_factor', short_factor, persistent=False)
        self.register_buffer('long_factor', long_factor, persistent=False)
        self.original_max_position_embeddings = \
            longrope_params.original_max_position_embeddings
        self.mscale = None
        self.short_mscale = longrope_params.short_mscale
        self.long_mscale = longrope_params.long_mscale
        if self.short_mscale is None and self.long_mscale is None:
            scale = (max_position_embeddings /
                     self.original_max_position_embeddings)
            if scale <= 1.0:
                self.mscale = 1.0
            else:
                self.mscale = math.sqrt(
                    1 + math.log(scale) /
                    math.log(self.original_max_position_embeddings))

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """rope forward."""
        dtype = x.dtype
        device = position_ids.device
        if self.short_factor.device != device:
            self.register_buffer('short_factor',
                                 self.short_factor.to(device),
                                 persistent=False)
            self.register_buffer('long_factor',
                                 self.long_factor.to(device),
                                 persistent=False)

        max_pos_ids = position_ids.max() + 1
        mask = max_pos_ids > self.original_max_position_embeddings
        ext_factors = torch.where(mask, self.long_factor, self.short_factor)

        mscale = self.mscale
        if mscale is None:
            mscale = torch.where(mask, self.long_mscale, self.short_mscale)

        inv_freq = self.inv_freq * (1.0 / ext_factors)
        return _rotary_embedding_fwd(position_ids,
                                     inv_freq,
                                     scaling_factor=1.0,
                                     mscale=mscale,
                                     dtype=dtype,
                                     device_type=device)


class DefaultRotaryEmbeddingBuilder(RotaryEmbeddingBuilder):
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
            return RotaryEmbeddingImpl(dim, base, scaling_factor)
        elif emb_type == RopeType.DynamicNTKScaling:
            return LlamaDynamicNTKScalingRotaryEmbedding(
                dim, base, scaling_factor, max_position_embeddings)
        elif emb_type == RopeType.Llama3:
            return Llama3RotaryEmbeddingImpl(dim, base, scaling_factor,
                                             llama3_params.low_freq_factor,
                                             llama3_params.high_freq_factor,
                                             max_position_embeddings)
        elif emb_type == RopeType.Yarn:
            return YarnRotaryEmbeddingImpl(dim,
                                           base,
                                           scaling_factor,
                                           max_position_embeddings,
                                           yarn_params=yarn_params)
        elif emb_type == RopeType.LongRoPEScaling:
            return LongRoPEScalingRotaryEmbeddingImpl(
                dim,
                base,
                max_position_embeddings=max_position_embeddings,
                longrope_params=longrope_params,
            )
        else:
            raise NotImplementedError(
                f'Unsupported embedding type: {emb_type}')

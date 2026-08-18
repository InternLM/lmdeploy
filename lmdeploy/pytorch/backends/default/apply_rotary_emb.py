# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

from ..apply_rotary_emb import ApplyRotaryEmbBuilder, ApplyRotaryEmbImpl


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    half_size = x.shape[-1] // 2
    x1 = x[..., :half_size]
    x2 = x[..., half_size:]
    out = torch.empty_like(x)
    out[..., :half_size] = -x2
    out[..., half_size:] = x1
    return out


def rotate_complex(x):
    """Rotates adjacent element pairs for complex-number RoPE."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = -x_odd
    out[..., 1::2] = x_even
    return out


def _prepare_cos_sin(query: Tensor, cos: Tensor, sin: Tensor, complex_mode: bool):
    """Prepare cos/sin tables for broadcasting over attention heads."""
    if complex_mode:
        feature_dim = query.size(-1)
        if cos.size(-1) * 2 == feature_dim:
            cos = cos.repeat_interleave(2, dim=-1)
        if sin.size(-1) * 2 == feature_dim:
            sin = sin.repeat_interleave(2, dim=-1)
        if cos.size(-1) != feature_dim or sin.size(-1) != feature_dim:
            raise ValueError('complex RoPE expects cos/sin width to be head_dim or head_dim // 2, '
                             f'but got cos={cos.size(-1)}, sin={sin.size(-1)}, head_dim={feature_dim}.')

    if cos.dim() == query.dim() - 1:
        cos = cos.unsqueeze(-2)
    if sin.dim() == query.dim() - 1:
        sin = sin.unsqueeze(-2)
    return cos, sin


class DefaultApplyRotaryEmbImpl(ApplyRotaryEmbImpl):
    """Apply rotary embedding implementation."""

    def forward(self,
                query: Tensor,
                key: Tensor,
                cos: Tensor,
                sin: Tensor,
                inplace: bool = True,
                complex_mode: bool = False):
        """forward."""
        if complex_mode:
            rotate_fn = rotate_complex
        else:
            rotate_fn = rotate_half
        cos, sin = _prepare_cos_sin(query, cos, sin, complex_mode)
        if inplace:
            q_embed = query
            k_embed = key
            q_sin = rotate_fn(query) * sin
            q_embed.mul_(cos)
            q_embed.add_(q_sin)
            k_sin = rotate_fn(key) * sin
            k_embed.mul_(cos)
            k_embed.add_(k_sin)
        else:
            q_embed = (query * cos) + (rotate_fn(query) * sin)
            k_embed = (key * cos) + (rotate_fn(key) * sin)
        return q_embed, k_embed


class DefaultApplyRotaryEmbBuilder(ApplyRotaryEmbBuilder):
    """Apply rotary embedding implementation builder."""

    @staticmethod
    def build():
        """Build implementation."""
        return DefaultApplyRotaryEmbImpl()

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


class DefaultApplyRotaryEmbImpl(ApplyRotaryEmbImpl):
    """Apply rotary embedding implementation."""

    def forward(self,
                query: Tensor,
                key: Tensor,
                cos: Tensor,
                sin: Tensor,
                inplace: bool = True):
        """forward."""
        unsqueeze_dim = -2
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        if inplace:
            q_embed = query
            k_embed = key
            q_sin = rotate_half(query) * sin
            q_embed.mul_(cos)
            q_embed.add_(q_sin)
            k_sin = rotate_half(key) * sin
            k_embed.mul_(cos)
            k_embed.add_(k_sin)
        else:
            q_embed = (query * cos) + (rotate_half(query) * sin)
            k_embed = (key * cos) + (rotate_half(key) * sin)
        return q_embed, k_embed


class DefaultApplyRotaryEmbBuilder(ApplyRotaryEmbBuilder):
    """Apply rotary embedding implementation builder."""

    @staticmethod
    def build():
        """build implementation."""
        return DefaultApplyRotaryEmbImpl()

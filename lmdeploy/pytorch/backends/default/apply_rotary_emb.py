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
            # cos/sin are (seq_len, dim//2), broadcast over heads
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)
            rotate_fn = rotate_complex
        else:
            unsqueeze_dim = -2
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)
            rotate_fn = rotate_half
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

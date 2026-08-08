# Copyright (c) OpenMMLab. All rights reserved.
"""Shared tensor and weight transforms for native vision models."""
from __future__ import annotations

import _turbomind as _tm
import torch

from ..linear import Linear, transform_input_dim, transform_output_dim
from ..weight_format import TrivialFormat


def to_tm_tensor(tensor: torch.Tensor, *, dtype: torch.dtype | None = None):
    """Convert a PyTorch tensor to a contiguous TurboMind tensor.

    Floating tensors are converted to ``dtype`` when one is provided.
    Non-floating tensors retain their original dtype.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            'TurboMind multimodal data should be a torch.Tensor, '
            f'got {type(tensor).__name__}')
    if (dtype is not None and tensor.is_floating_point()
            and tensor.dtype != dtype):
        tensor = tensor.to(dtype)
    return _tm.from_dlpack(tensor.contiguous())


def to_tm_norm_type(norm_type: str):
    """Map a vision norm name to its TurboMind enum value."""
    if norm_type == 'layer_norm':
        return _tm.NormType.LAYER_NORM
    if norm_type == 'rms_norm':
        return _tm.NormType.RMS_NORM
    raise ValueError(f'Unsupported vision norm_type: {norm_type!r}')


@transform_output_dim
def split_packed_qkv(tensor: torch.Tensor):
    """Split a packed vision projection laid out as ``[Q | K | V]``."""
    if tensor.shape[-1] % 3 != 0:
        raise ValueError(
            'packed vision qkv output dim is not divisible by 3: '
            f'{tuple(tensor.shape)}')
    return tuple(x.contiguous() for x in tensor.chunk(3, dim=-1))


@transform_output_dim
def _pad_head_dim_out(tensor: torch.Tensor, *, num_heads: int,
                      src_head_dim: int, dst_head_dim: int) -> torch.Tensor:
    rest = tensor.shape[:-1]
    tensor = tensor.reshape(rest + (num_heads, src_head_dim))
    pad = tensor.new_zeros(
        rest + (num_heads, dst_head_dim - src_head_dim))
    return torch.cat([tensor, pad], dim=-1).reshape(
        rest + (num_heads * dst_head_dim, ))


@transform_input_dim
def _pad_head_dim_in(tensor: torch.Tensor, *, num_heads: int,
                     src_head_dim: int, dst_head_dim: int) -> torch.Tensor:
    rest = tensor.shape[1:]
    tensor = tensor.reshape((num_heads, src_head_dim) + rest)
    pad = tensor.new_zeros(
        (num_heads, dst_head_dim - src_head_dim) + rest)
    return torch.cat([tensor, pad], dim=1).reshape(
        (num_heads * dst_head_dim, ) + rest)


def pad_attn_head_dim(q: Linear, k: Linear, v: Linear,
                      proj: Linear, *, num_heads: int,
                      src_head_dim: int,
                      dst_head_dim: int):
    """Pad ViT attention projection weights to a supported head dimension.

    Q/K/V are padded on their output dimension and the output projection is padded on its input dimension. Equal source
    and destination dimensions return the original weights unchanged. Padding currently requires trivial floating-point
    weights.
    """
    if dst_head_dim < src_head_dim:
        raise ValueError(
            f'dst_head_dim={dst_head_dim} is smaller than '
            f'src_head_dim={src_head_dim}')
    if dst_head_dim == src_head_dim:
        return q, k, v, proj

    linears = (q, k, v, proj)
    for linear, name in zip(linears, ('q', 'k', 'v', 'proj')):
        if not isinstance(linear.weight_format, TrivialFormat):
            raise NotImplementedError(
                f'ViT {name} weight is '
                f'{type(linear.weight_format).__name__}; head_dim padding '
                'currently supports TrivialFormat only')

    kwargs = dict(
        num_heads=num_heads,
        src_head_dim=src_head_dim,
        dst_head_dim=dst_head_dim,
    )
    return (
        _pad_head_dim_out(q, **kwargs),
        _pad_head_dim_out(k, **kwargs),
        _pad_head_dim_out(v, **kwargs),
        _pad_head_dim_in(proj, **kwargs),
    )

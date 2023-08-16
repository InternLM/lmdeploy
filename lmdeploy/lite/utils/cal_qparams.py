# Copyright (c) OpenMMLab. All rights reserved.
from typing import NamedTuple, Optional

import torch


class QParams(NamedTuple):
    """A class to hold the quantization parameters."""

    scales: torch.Tensor
    zero_points: Optional[torch.Tensor]


@torch.no_grad()
def cal_qparams_per_channel_absmax(w: torch.Tensor,
                                   n_bits: int,
                                   return_stats: bool = False) -> QParams:
    """Calculate quantization parameters for each channel using absolute max
    value."""

    absmax = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits - 1) - 1
    scales = absmax.clamp(min=1e-5).div(q_max)

    if return_stats:
        return QParams(scales=scales, zero_points=None), absmax
    else:
        return QParams(scales=scales, zero_points=None)


@torch.no_grad()
def cal_qparams_per_channel_minmax(w: torch.Tensor,
                                   n_bits: int,
                                   return_stats: bool = False) -> QParams:
    """Calculate quantization parameters for each channel using min and max
    values."""

    w_min = w.min(dim=-1, keepdim=True)[0]
    w_max = w.max(dim=-1, keepdim=True)[0]

    q_max = 2**n_bits - 1
    scales = (w_max - w_min)
    scales = scales.clamp_(min=1e-5).div_(q_max)

    zero_points = (-w_min / scales).round()

    if return_stats:
        return QParams(scales=scales, zero_points=zero_points), (w_min, w_max)
    else:
        return QParams(scales=scales, zero_points=zero_points)


@torch.no_grad()
def cal_qparams_per_group_absmax(w: torch.Tensor,
                                 n_bits: int,
                                 group_size: int,
                                 return_stats: bool = False) -> QParams:
    """Calculate quantization parameters for each group using absolute max
    value."""

    outc, inc = w.shape
    assert inc >= group_size, \
        'Input channels should be greater than or equal to group_size.'
    assert inc % group_size == 0, \
        'Input channels should be divisible by group_size.'
    absmax = w.abs().reshape(outc, -1, group_size).max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits - 1) - 1
    scales = absmax.clamp(min=1e-5).div(q_max)
    if return_stats:
        return QParams(scales=scales, zero_points=None), absmax
    else:
        return QParams(scales=scales, zero_points=None)


@torch.no_grad()
def cal_qparams_per_group_minmax(w: torch.Tensor,
                                 n_bits: int,
                                 group_size: int,
                                 return_stats: bool = False) -> QParams:
    """Calculate quantization parameters for each group using min and max
    values."""

    outc, inc = w.shape
    assert inc >= group_size, \
        'Input channels should be greater than or equal to group_size.'
    assert inc % group_size == 0, \
        'Input channels should be divisible by group_size.'
    w_group_wise = w.reshape(outc, -1, group_size)
    w_min = w_group_wise.min(dim=-1, keepdim=True)[0]
    w_max = w_group_wise.max(dim=-1, keepdim=True)[0]

    q_max = 2**n_bits - 1
    scales = (w_max - w_min)
    scales = scales.clamp_(min=1e-5).div_(q_max)
    zero_points = (-w_min / scales).round()
    if return_stats:
        return QParams(scales=scales, zero_points=zero_points), (w_min, w_max)
    else:
        return QParams(scales=scales, zero_points=zero_points)


@torch.no_grad()
def cal_qparams_per_tensor_minmax(w: torch.Tensor,
                                  n_bits: int,
                                  return_stats: bool = False) -> QParams:
    """Calculate quantization parameters for the entire tensor using min and
    max values."""

    w_min = w.min()
    w_max = w.max()

    q_max = 2**n_bits - 1
    scales = (w_max - w_min)
    scales = scales.clamp_(min=1e-5).div_(q_max)
    zero_points = (-w_min / scales).round()
    if return_stats:
        return QParams(scales=scales, zero_points=zero_points), (w_min, w_max)
    else:
        return QParams(scales=scales, zero_points=zero_points)


@torch.no_grad()
def cal_qparams_per_tensor_absmax(w: torch.Tensor,
                                  n_bits: int,
                                  return_stats: bool = False) -> QParams:
    """Calculate quantization parameters for the entire tensor using absolute
    max value."""
    absmax = w.abs().max()
    q_max = 2**(n_bits - 1) - 1
    scales = absmax.clamp(min=1e-5).div(q_max)

    if return_stats:
        return QParams(scales=scales, zero_points=None), absmax
    else:
        return QParams(scales=scales, zero_points=None)

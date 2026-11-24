# Copyright (c) 2024, OpenMMLab and DeepLink. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import torch
from torch import nn

if TYPE_CHECKING:
    from lmdeploy.lite.quantization.modules.linear import WeightOnlyQLinear
from lmdeploy.lite.utils.cal_qparams import QParams


def weight_only_qlinear_from_linear(
    cls: type[WeightOnlyQLinear],
    linear: nn.Linear,
    quantizer: TypeVar('Quantizer'),
    awq_layout: bool = True,
    qparams: QParams | None = None,
) -> WeightOnlyQLinear:
    """Create a WeightOnlyQLinear object from a PyTorch Linear object.

    Args:
        linear (nn.Linear): PyTorch Linear object.
        quantizer (Quantizer): Object that handles quantization.
        awq_layout (bool): AWQ layout. Defaults to True.

    Returns:
        WeightOnlyQLinear: A WeightOnlyQLinear object.
    """
    device = linear.weight.device

    w_bit = quantizer.bits
    pack_num = 32 // w_bit
    if awq_layout:
        assert w_bit == 4
        weight_pack_order = [0, 1, 2, 3, 4, 5, 6, 7]
    else:
        raise ValueError('ascend only support awq')
    group_size = quantizer.group_size
    symmetry = quantizer.symmetry

    in_features = linear.in_features
    out_features = linear.out_features
    bias = False if linear.bias is None else True

    qlinear = cls(in_features, out_features, bias, w_bit, symmetry, group_size)
    qlinear.bias = linear.bias

    if qparams is None:
        qparams = quantizer.calculate_qparams(linear.weight)
        i32_w = quantizer.quant(linear.weight, qparams, real=True)
    else:
        i32_w = linear.weight.to(torch.int32)
    i32_w = i32_w.t().contiguous()

    pack_int_w = torch.zeros_like(qlinear.qweight).to(device)

    for col in range(pack_int_w.shape[1]):
        for i in range(pack_num):
            pack_int_w_col = i32_w[:, col * pack_num + weight_pack_order[i]]
            pack_int_w[:, col] |= pack_int_w_col << (i * w_bit)

    qlinear.qweight = torch.bitwise_xor(pack_int_w, 0x88888888)
    qlinear.scales = qparams.scales.squeeze(-1).t().contiguous()

    if qparams.zero_points is not None:
        zeros = qparams.zero_points.to(torch.int32).to(device)
        zeros = zeros.squeeze(-1).t().contiguous()
        qlinear.qzeros = 8.0 - zeros.to(torch.float16)
    qlinear.to('cpu')

    return qlinear

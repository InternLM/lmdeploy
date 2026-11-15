# Copyright (c) OpenMMLab. All rights reserved.

from torch import Tensor

from . import ops as ext_ops


def awq_linear(
    x: Tensor,
    qweight: Tensor,
    scales: Tensor,
    qzeros: Tensor,
    bias: Tensor | None = None,
    all_reduce: bool = False,
    group_size: int = 0,
):
    return ext_ops.weight_quant_matmul(
        x.squeeze(0), qweight, scales, offset=qzeros, bias=bias, all_reduce=all_reduce, group_size=group_size
    ).unsqueeze(0)

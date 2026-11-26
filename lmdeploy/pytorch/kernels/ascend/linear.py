# Copyright (c) OpenMMLab. All rights reserved.

from torch import Tensor

from . import ops as ext_ops


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None, all_reduce: bool = False, group: str = ''):
    return ext_ops.linear(x, weight, bias=bias, all_reduce=all_reduce, group=group)

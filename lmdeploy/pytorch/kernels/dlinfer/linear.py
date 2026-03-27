# Copyright (c) OpenMMLab. All rights reserved.

import dlinfer.ops as ext_ops
from torch import Tensor


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None, all_reduce: bool = False, group: str = ''):
    return ext_ops.linear(x, weight, bias=bias, all_reduce=all_reduce, group=group)

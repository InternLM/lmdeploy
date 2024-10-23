# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import dlinfer.ops as ext_ops
from torch import Tensor


def awq_linear(x: Tensor,
               qweight: Tensor,
               scales: Tensor,
               qzeros: Tensor,
               bias: Optional[Tensor] = None,
               all_reduce: bool = False,
               group_size: int = 0):
    return ext_ops.weight_quant_matmul(x.squeeze(0),
                                       qweight,
                                       scales,
                                       offset=qzeros,
                                       bias=bias,
                                       all_reduce=all_reduce,
                                       group_size=group_size).unsqueeze(0)

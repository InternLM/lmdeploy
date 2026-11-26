# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from . import ops as ext_ops


def silu_and_mul(
    input_tensor: Tensor,
) -> Tensor:
    return ext_ops.silu_and_mul(input_tensor)

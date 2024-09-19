# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops

from torch import Tensor

def silu_and_mul(x: Tensor):
    return ext_ops.silu_and_mul(x)

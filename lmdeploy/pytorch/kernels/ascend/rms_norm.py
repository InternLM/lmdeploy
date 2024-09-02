# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops
from torch import Tensor


def rms_norm(hidden_states: Tensor,
             weight: Tensor,
             eps: float = 1e-6,
             out: Tensor = None):
    rms_norm_out = ext_ops.rms_norm(hidden_states, weight, eps)
    if out is None:
        out = rms_norm_out
    else:
        out.copy_(rms_norm_out)
    return out

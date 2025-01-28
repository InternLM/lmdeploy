# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import dlinfer.ops as ext_ops
from torch import Tensor, dtype


def rms_norm(hidden_states: Tensor,
             weight: Tensor,
             epsilon: float = 1e-6,
             residual: Tensor = None,
             quant_dtype: Optional[dtype] = None,
             out: Tensor = None):
    if residual is None:
        rms_norm_out = ext_ops.rms_norm(hidden_states, weight, epsilon, quant_dtype)
        if out is None:
            out = rms_norm_out
        else:
            out.copy_(rms_norm_out)
        return out
    else:
        return ext_ops.add_rms_norm(hidden_states, residual, weight, epsilon, quant_dtype)

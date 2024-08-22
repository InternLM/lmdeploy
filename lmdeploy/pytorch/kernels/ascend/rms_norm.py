# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops
from torch import Tensor


def rms_norm(hidden_states: Tensor, weight: Tensor, epsilon: float = 1e-6):
    return ext_ops.rms_norm(hidden_states, weight, epsilon)

# Copyright (c) OpenMMLab. All rights reserved.
import deeplink_ext.cpp_extensions as ext
import torch
from torch import Tensor


def rms_norm(hidden_states: Tensor, weight: Tensor, eps: float = 1e-6):
    output = torch.empty_like(hidden_states)
    inv_rms_shape = list(hidden_states.shape[:-1]) + [1]
    inv_rms = torch.empty(inv_rms_shape,
                          dtype=torch.float32,
                          device=hidden_states.device)
    ext.rms_norm(output, inv_rms, hidden_states, weight.shape, weight, None,
                 eps)
    return output

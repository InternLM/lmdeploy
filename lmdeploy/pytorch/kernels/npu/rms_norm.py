# Copyright (c) OpenMMLab. All rights reserved.
import torch_npu
from torch import Tensor


def rms_norm(hidden_states: Tensor, weight: Tensor, eps: float = 1e-6):
    return torch_npu.npu_rms_norm(hidden_states, weight, eps)[0]

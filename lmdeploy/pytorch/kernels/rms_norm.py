# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from .dispatcher import FunctionDispatcher


def _rms_norm_interface(hidden_states: Tensor,
                        weight: Tensor,
                        eps: float = 1e-6):
    """rms_norm."""
    ...


rms_norm = FunctionDispatcher('rms_norm').make_caller(_rms_norm_interface)

# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

from .dispatcher import FunctionDispatcher


def _fused_rotary_emb_api(q: Tensor,
                          k: Tensor,
                          position_ids: torch.LongTensor,
                          inv_freq: Tensor,
                          scaling_factor: float,
                          out_q: Tensor = None,
                          out_k: Tensor = None):
    """Fuse `rotary_embedding` and `apply_rotary_pos_emb`."""
    ...


fused_rotary_emb = FunctionDispatcher('fused_rotary_emb').make_caller(
    _fused_rotary_emb_api)

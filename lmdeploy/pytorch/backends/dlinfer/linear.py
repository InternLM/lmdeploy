# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import List, Optional

import torch
import torch.distributed as dist

from lmdeploy.pytorch.kernels.dlinfer import linear

from ..linear import LinearBuilder, LinearImpl


class DlinferLinearImpl(LinearImpl):
    """Dlinfer linear implementation api."""

    def update_weights(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Update weights."""
        if os.getenv('DLINFER_LINEAR_USE_NN_LAYOUT', '0') == '1':
            weight = weight.data.t().contiguous()
        return weight, bias

    def forward(self,
                x,
                weight: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False,
                group: dist.ProcessGroup = None,
                rank: int = 0,
                scatter_size: List[int] = None):
        """forward."""
        out = linear(x, weight, bias, False)
        if all_reduce:
            dist.all_reduce(out, group=group)
        return out


class DlinferLinearBuilder(LinearBuilder):
    """Dlinfer linear implementation builder."""

    @staticmethod
    def build(in_features: int, out_features: int, bias: bool = True, dtype: torch.dtype = None):
        """build."""
        return DlinferLinearImpl()

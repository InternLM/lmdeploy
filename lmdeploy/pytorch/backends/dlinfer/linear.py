# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch

from lmdeploy.pytorch.kernels.dlinfer import linear

from ..linear import LinearBuilder, LinearImpl


class DlinferLinearImpl(LinearImpl):
    """Dlinfer linear implementation api."""

    def forward(self,
                x,
                weight: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False):
        """forward."""
        return linear(x, weight, bias, all_reduce)


class DlinferLinearBuilder(LinearBuilder):
    """Dlinfer linear implementation builder."""

    @staticmethod
    def build(in_features: int,
              out_features: int,
              bias: bool = True,
              dtype: torch.dtype = None):
        """build."""
        return DlinferLinearImpl()

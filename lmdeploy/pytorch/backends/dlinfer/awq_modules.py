# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch

from lmdeploy.pytorch.kernels.dlinfer import awq_linear

from ..awq_modules import LinearW4A16Builder, LinearW4A16Impl


class AwqLinearW4A16Impl(LinearW4A16Impl):
    """Awq kernel linear."""

    def __init__(self, in_features: int, out_features: int, w_bit: int, group_size: int):
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size

    def forward(self,
                x,
                qweight: torch.Tensor,
                scales: torch.Tensor,
                qzeros: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False,
                group: Optional[torch.distributed.ProcessGroup] = None):
        """forward."""
        out = awq_linear(x, qweight, scales, qzeros, bias, all_reduce, self.group_size)
        return out


class AwqLinearW4A16Builder(LinearW4A16Builder):
    """Awq linear builder."""

    @staticmethod
    def build(in_features: int,
              out_features: int,
              w_bit: int,
              group_size: int,
              bias: bool = False,
              dtype: torch.dtype = None):
        """build."""
        return AwqLinearW4A16Impl(in_features, out_features, w_bit, group_size)

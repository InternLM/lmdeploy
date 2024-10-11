# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import dlinfer.ops as ext_ops
import torch

from ..awq_modules import LinearW4A16Builder, LinearW4A16Impl


class AwqLinearW4A16Impl(LinearW4A16Impl):
    """awq kernel linear."""

    def __init__(self, in_features: int, out_features: int, w_bit: int,
                 group_size: int):
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
                all_reduce: bool = False):
        """forward."""
        out = ext_ops.weight_quant_matmul(
            x.squeeze(0),
            qweight,
            scales,
            offset=qzeros,
            bias=bias,
            group_size=self.group_size).unsqueeze(0)
        return out


class AwqLinearW4A16Builder(LinearW4A16Builder):
    """awq linear builder."""

    @staticmethod
    def build(in_features: int,
              out_features: int,
              w_bit: int,
              group_size: int,
              bias: bool = False,
              dtype: torch.dtype = None):
        """build."""
        return AwqLinearW4A16Impl(in_features, out_features, w_bit, group_size)

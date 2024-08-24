# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from torch import distributed as dist

from ..awq_modules import LinearW4A16Builder, LinearW4A16Impl


class DefaultLinearW4A16Impl(LinearW4A16Impl):
    """w4a16 linear implementation."""

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
        from awq.utils.packing_utils import dequantize_gemm
        out_shape = x.shape[:-1] + (self.out_features, )
        input_dtype = x.dtype
        if input_dtype != torch.float16:
            x = x.half()
        out = dequantize_gemm(qweight, qzeros, scales, self.w_bit,
                              self.group_size)
        out = torch.matmul(x, out)

        out = out + bias if bias is not None else out
        out = out.reshape(out_shape)

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)
        if all_reduce:
            dist.all_reduce(out)
        return out


class DefaultLinearW4A16Builder(LinearW4A16Builder):
    """w4a16 linear implementation builder."""

    @staticmethod
    def build(in_features: int,
              out_features: int,
              w_bit: int,
              group_size: int,
              bias: bool = False,
              dtype: torch.dtype = None):
        """build."""
        return DefaultLinearW4A16Impl(in_features, out_features, w_bit,
                                      group_size)
